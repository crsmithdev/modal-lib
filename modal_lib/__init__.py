"""Run training jobs on Modal GPUs, and record the wall-clock time and cost of each phase.

Import this package first in the app file: its import time is the container
start (`t_start`). See README.md.
"""

from ._start import T_START  # noqa: I001  (first, before modal is imported)

from .cache import CACHE_MOUNT, cache_filled, cache_volume, compile_cache
from .clock import PRICE, Clock, cost, gpu_name, measure
from .image import torch_image
from .recipes import gpu_spec, load, override, resolve
from .runs import RUNS_MOUNT, Run, launch, runs_volume

__all__ = [
    "T_START", "CACHE_MOUNT", "RUNS_MOUNT", "PRICE",
    "Clock", "Run", "cache_filled", "cache_volume", "compile_cache", "cost", "gpu_name",
    "gpu_spec", "launch", "load", "measure", "override", "resolve", "runs_volume", "torch_image",
]
