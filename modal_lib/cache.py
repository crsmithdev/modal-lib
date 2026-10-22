"""A torch.compile cache on a Modal Volume, one per app and GPU type."""

import contextlib
import os
import shutil
from collections.abc import Iterator

import modal

CACHE_MOUNT = "/cache"


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
