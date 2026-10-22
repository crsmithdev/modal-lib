"""Run functions on Modal GPUs and record the wall-clock time of each phase.

Import this module first in the app file. The container imports the app file
when it starts, so the import time is the container start (`t_start`).

In the container, a `Clock` records `t_<name>` at each `mark(name)`. On the
local side, `run` records `t_call` before each call and writes one JSON row
per call. The time from `t_call` to `t_start` is the cold start, and Modal
bills from `t_start`.
"""

import time

T_START = time.time()

import concurrent.futures
import contextlib
import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Callable, Iterable, Iterator

import modal

# Modal GPU prices in dollars per second, from modal.com/pricing on 2026-10-21.
# CPU and memory add under 1% at the default container size and are left out.
PRICE = {
    "T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542,
    "A100-40GB": 0.000583, "A100-80GB": 0.000694, "RTX-PRO-6000": 0.000842,
    "H100": 0.001097, "H200": 0.001261, "B200": 0.001736,
}

CACHE_MOUNT = "/cache"


def torch_image(torch: str = "2.11.0", torchvision: str = "0.26.0", cuda: str = "cu128",
                python: str = "3.12", packages: Iterable[str] = ()) -> modal.Image:
    """A Debian image with torch from the PyTorch index and this module.

    Put every build step in `packages`. Modal accepts only more `add_local_*`
    calls after this image.
    """
    image = modal.Image.debian_slim(python_version=python).uv_pip_install(
        f"torch=={torch}", f"torchvision=={torchvision}",
        index_url=f"https://download.pytorch.org/whl/{cuda}",
    )
    if packages:
        image = image.uv_pip_install(*packages)
    return image.add_local_python_source("modal_clock")


class Clock:
    """Wall-clock marks in one container, as `t_<name>` keys of one record."""

    def __init__(self) -> None:
        self.record = {"t_start": T_START}

    def mark(self, name: str) -> float:
        t = self.record[f"t_{name}"] = time.time()
        return t

    def done(self, **extra) -> dict:
        """Mark `end` and return the record with the device name and `extra`."""
        self.mark("end")
        return dict(gpu_name=gpu_name(), **self.record, **extra)


def gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def cache_volume(app: str, gpu: str) -> modal.Volume:
    """The Volume that holds the compile cache of `app` on one GPU type."""
    return modal.Volume.from_name(f"{app}-cache-{gpu.lower()}", create_if_missing=True)


def cache_filled(volume: modal.Volume) -> bool:
    return "cache.tar" in [e.path for e in volume.listdir("/")]


@contextlib.contextmanager
def compile_cache(fill: modal.Volume | None = None) -> Iterator[None]:
    """Use the Inductor and Triton caches from the Volume at CACHE_MOUNT.

    Enter it before `import torch`, because Inductor and Triton read the cache
    paths at import. The cache is one tar file: a Modal Volume takes about
    0.4 s per file, and a compile cache can have thousands of files. With
    `fill`, the exit writes the cache to that Volume.
    """
    tar = f"{CACHE_MOUNT}/cache.tar"
    if os.path.exists(tar):
        shutil.unpack_archive(tar, "/tmp/cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/cache/inductor"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/cache/triton"
    yield
    if fill:
        shutil.make_archive("/tmp/cache", "tar", "/tmp/cache")
        shutil.copy("/tmp/cache.tar", tar)
        fill.commit()


def run(calls: Iterable[tuple[dict, Callable[[], dict]]], out: pathlib.Path,
        concurrency: int = 8) -> Iterator[dict]:
    """Make each call and append its row to the JSON Lines file `out`.

    Each call is (fields, thunk). The row is the fields, `t_call`, and the
    thunk's record, or `error` if the call raised. Keep `concurrency` under
    the account's GPU limit, so that no call waits in a queue and the time
    from `t_call` to `t_start` is the cold start only.
    """
    def one(call):
        fields, thunk = call
        t_call = time.time()
        try:
            return dict(**fields, t_call=t_call, **thunk())
        except Exception as e:
            return dict(**fields, t_call=t_call, error=repr(e))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f, concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        for row in pool.map(one, calls):
            f.write(json.dumps(row) + "\n")
            f.flush()
            yield row


def cost(seconds: float, gpu: str) -> float:
    """Dollars for `seconds` of one `gpu` at PRICE."""
    return seconds * PRICE[gpu]
