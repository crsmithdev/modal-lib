"""Wall-clock marks in a container, and timed calls from the local side."""

import concurrent.futures
import json
import pathlib
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator

from ._start import T_START
from .recipes import gpu_spec

# Modal GPU prices in dollars per second per GPU, from modal.com/pricing on 2026-10-21.
# CPU and memory add under 1% at the default container size and are left out.
PRICE = {
    "T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542,
    "A100-40GB": 0.000583, "A100-80GB": 0.000694, "RTX-PRO-6000": 0.000842,
    "H100": 0.001097, "H200": 0.001261, "B200": 0.001736,
}


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


def cost(seconds: float, gpu: str) -> float:
    """Dollars for `seconds` on `gpu`, a spec such as "H100" or "H100:8", at PRICE."""
    kind, count = gpu_spec(gpu)
    return seconds * PRICE[kind] * count


def measure(calls: Iterable[tuple[dict, Callable[[], dict]]], out: pathlib.Path,
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
