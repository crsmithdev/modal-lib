"""Time a small compiled function on Modal GPUs, cold and with a warm compile cache.

    uv run modal run examples/clock_matmul.py
    uv run modal run examples/clock_matmul.py --gpus L4,H100 --repeats 3
"""

import modal_lib as ml

import contextlib
import pathlib

import modal

APP = "modal-lib-clock-example"
app = modal.App(APP, image=ml.torch_image())


@app.function(timeout=600, single_use_containers=True)
def matmul(gpu: str, cached: bool, fill: bool = False) -> dict:
    clock = ml.Clock()
    clock.mark("enter")
    cache = ml.compile_cache(fill=ml.cache_volume(APP, gpu) if fill else None)
    with cache if cached else contextlib.nullcontext():
        clock.mark("copy")
        import torch
        clock.mark("torch")
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        clock.mark("cuda")
        f = torch.compile(lambda x: torch.relu(x @ x).sum(), mode="max-autotune")
        f(x)
        torch.cuda.synchronize()
        clock.mark("warm")
        for _ in range(10):
            f(x)
        torch.cuda.synchronize()
        return clock.done()


@app.local_entrypoint()
def main(gpus: str = "L4", repeats: int = 1):
    gpus = gpus.split(",")

    def fn(gpu, cached):
        volumes = {ml.CACHE_MOUNT: ml.cache_volume(APP, gpu)} if cached else {}
        return matmul.with_options(gpu=gpu, volumes=volumes)

    # Fill each GPU's cache once. These calls are not measured.
    for g in gpus:
        if not ml.cache_filled(ml.cache_volume(APP, g)):
            fn(g, True).remote(g, True, fill=True)

    calls = [(dict(gpu=g, cached=c), lambda g=g, c=c: fn(g, c).remote(g, c))
             for _ in range(repeats) for c in (False, True) for g in gpus]
    for row in ml.measure(calls, pathlib.Path("results/matmul.jsonl")):
        if "error" in row:
            print(row["gpu"], row["error"])
            continue
        print(f"{row['gpu']} {'warm' if row['cached'] else 'cold'}: "
              f"call to start {row['t_start'] - row['t_call']:.1f} s, "
              f"setup {row['t_cuda'] - row['t_start']:.1f} s, "
              f"compile {row['t_warm'] - row['t_cuda']:.1f} s, "
              f"${ml.cost(row['t_end'] - row['t_start'], row['gpu']):.4f} in container")
