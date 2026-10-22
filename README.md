# modal-clock

Run functions on [Modal](https://modal.com) GPUs and record the wall-clock
time of each phase, from the call to the result. It also keeps a
`torch.compile` cache on a Modal Volume per GPU type, and prices each call.

A benchmark usually times only the loop it cares about. A cloud GPU call also
spends time waiting for a container, importing torch, starting CUDA and
compiling, and Modal bills from container start. This module records all of
it.

## Use

```sh
uv add "modal-clock @ git+https://github.com/crsmithdev/modal-clock"
```

Import `modal_clock` first in the app file. The container imports the app
file when it starts, so the import time is the container start.

```python
import modal_clock as mc

import contextlib
import pathlib

import modal

APP = "my-bench"
app = modal.App(APP, image=mc.torch_image(packages=["numpy"]))


@app.function(timeout=3600, single_use_containers=True)
def bench(gpu: str, cached: bool, fill: bool = False) -> dict:
    clock = mc.Clock()
    clock.mark("enter")
    cache = mc.compile_cache(fill=mc.cache_volume(APP, gpu) if fill else None)
    with cache if cached else contextlib.nullcontext():
        clock.mark("copy")
        import torch                # after the cache: Inductor reads its paths at import
        clock.mark("torch")
        ...                         # compile and warm up
        clock.mark("warm")
        ...                         # the timed work
        return clock.done(acc=0.94)  # extra fields go into the record


@app.local_entrypoint()
def main(gpu: str = "H100"):
    volumes = {mc.CACHE_MOUNT: mc.cache_volume(APP, gpu)}
    calls = [(dict(gpu=gpu, cached=False), lambda: bench.with_options(gpu=gpu).remote(gpu, False))]
    for row in mc.run(calls, pathlib.Path("results/bench.jsonl")):
        print(row)
```

[`examples/matmul.py`](examples/matmul.py) is a complete app that fills the
cache and times a cold call and a warm call:

```sh
uv run modal run examples/matmul.py --gpus L4,H100
```

## What it records

Each call writes one JSON row:

| Field | Set by | Time |
| - | - | - |
| `t_call` | `run`, local | before the call |
| `t_start` | the import, in the container | container start |
| `t_<name>` | `Clock.mark(name)` | wherever the function marks it |
| `t_end` | `Clock.done()` | the end of the function |
| `gpu_name` | `Clock.done()` | the device from `nvidia-smi` |
| `error` | `run` | if the call raised |

The row also has the fields that you give to `run` and to `Clock.done()`. All
times are Unix seconds from `time.time()`, so durations between the local
side and the container depend on the clocks agreeing.

- `t_start - t_call` is the cold start. Modal does not bill it.
- `t_end - t_start` is the billed time. `mc.cost(seconds, gpu)` prices it at
  `mc.PRICE`, Modal's GPU prices on 2026-10-21. CPU and memory are left out.

## Compile cache

`compile_cache()` points `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` at
`/tmp/cache` and unpacks `/cache/cache.tar` there if the Volume has it. Enter
it before `import torch`. With `fill=<volume>`, the exit packs the cache and
commits it to the Volume.

The cache is one tar file because a Modal Volume reads about 0.4 s per file,
and an Inductor cache can have thousands of files. In one case, 5,165 files
took 33 minutes to read one by one and 1 to 5 s to unpack as one tar.

`cache_volume(app, gpu)` keys the Volume by the GPU type that you ask for.
Modal sometimes gives a different variant of that type (an A100 80GB for an
A100 40GB, an A10G for an A10), and the cache misses on a different device.
Record `gpu_name` and group the results by it.

## Used by

- [speedrun-rta](https://github.com/crsmithdev/speedrun-rta): real time and
  cost of ML training speedruns.
