"""Time a small compiled function on Modal GPUs, cold and with a warm compile cache.

    uv run modal run examples/matmul.py
    uv run modal run examples/matmul.py --gpus L4,H100 --repeats 3
"""

import modal_clock as mc

import contextlib
import pathlib

import modal

APP = "modal-clock-example"
app = modal.App(APP, image=mc.torch_image())


@app.function(timeout=600, single_use_containers=True)
def matmul(gpu: str, cached: bool, fill: bool = False) -> dict:
    clock = mc.Clock()
    clock.mark("enter")
    cache = mc.compile_cache(fill=mc.cache_volume(APP, gpu) if fill else None)
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
        volumes = {mc.CACHE_MOUNT: mc.cache_volume(APP, gpu)} if cached else {}
        return matmul.with_options(gpu=gpu, volumes=volumes)

    # Fill each GPU's cache once. These calls are not measured.
    for g in gpus:
        if not mc.cache_filled(mc.cache_volume(APP, g)):
            fn(g, True).remote(g, True, fill=True)

    calls = [(dict(gpu=g, cached=c), lambda g=g, c=c: fn(g, c).remote(g, c))
             for _ in range(repeats) for c in (False, True) for g in gpus]
    for row in mc.run(calls, pathlib.Path("results/matmul.jsonl")):
        if "error" in row:
            print(row["gpu"], row["error"])
            continue
        print(f"{row['gpu']} {'warm' if row['cached'] else 'cold'}: "
              f"call to start {row['t_start'] - row['t_call']:.1f} s, "
              f"setup {row['t_cuda'] - row['t_start']:.1f} s, "
              f"compile {row['t_warm'] - row['t_cuda']:.1f} s, "
              f"${mc.cost(row['t_end'] - row['t_start'], row['gpu']):.4f} in container")
