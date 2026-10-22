# modal-lib

Run training jobs on [Modal](https://modal.com) GPUs from a catalog of
recipes, and record the wall-clock time and cost of each phase.

- **Recipes:** one JSON file of named job settings, with `--set` overrides
  on the command line.
- **Runs:** each call gets an ID, a folder on a Modal Volume, and a record
  of when each phase started and what it cost. Runs outlive the terminal, and
  `mlib runs` and `mlib pull` find them later.
- **Clock:** the time from the call to the container start, through setup
  and compile, to the result. Modal bills from container start, so the
  record also prices each run.
- **Compile cache:** a `torch.compile` cache on a Volume per GPU type, as
  one tar file.

## Install

```sh
uv add "modal-lib @ git+https://github.com/crsmithdev/modal-lib"
uv run modal setup   # once
```

## A job

A job is a Modal function that takes `(recipe, run_id)`. Import `modal_lib`
first in the app file: the container imports the app file when it starts, so
the import time is the container start.

```python
import modal_lib as ml

import modal

APP = "my-jobs"
app = modal.App(APP, image=ml.torch_image(packages=["numpy"]).add_local_dir("src", "/root/src"))


@app.function(timeout=3600, single_use_containers=True)
def train(recipe: dict, run_id: str) -> dict:
    with ml.Run(APP, run_id, recipe) as run:
        import torch
        run.clock.mark("torch")
        ...                                   # set up
        run.clock.mark("compile")
        ...                                   # train, and save files in run.dir
        run.result = {"loss": 0.1}
    return run.record
```

`single_use_containers=True` gives each run a fresh container. A reused
container started before the run's call, so its record has `reused: true`
and no cold start.

Recipes go in one JSON file:

```json
{
  "default": "smoke",
  "recipes": {
    "smoke": {"gpu": "L4", "steps": 500, "lr": 0.001, "note": "a minute on an L4"},
    "full": {"gpu": "H100:8", "steps": 20000, "lr": 0.0003, "timeout": 21600}
  }
}
```

The job gets the whole recipe. modal-lib reads four keys: `gpu` (a Modal
GPU spec), `timeout` (seconds, default 3600), `env` (environment variables
for the container) and `function` (a job other than the default).

Point `mlib` at the app in `pyproject.toml`:

```toml
[tool.modal-lib]
app = "my-jobs"
file = "jobs.py"
function = "train"
recipes = "recipes.json"
```

## Commands

```sh
mlib recipes                      # the catalog; * is the default
mlib show full --set steps=100    # one recipe after overrides
mlib deploy                       # deploy the app; runs need a deployed app
mlib run smoke                    # spawn a run and print its ID
mlib run full --set lr=1e-4 --set env.SEED=3 --wait
mlib runs                         # every run: status, time to start, run time, dollars
mlib pull smoke-20261022-095704   # download the run's folder to runs/<id>/
```

`mlib run` appends each run to `.modal-lib/runs.jsonl` in the project. A
`--set` value is JSON if it parses, and a string if not.

`mlib runs` output:

```
smoke-20261022-095704                L4         done       start   6.0 s  run     7.9 s  $0.002
smoke-20261022-095734                T4         done       start   3.9 s  run    24.0 s  $0.004
smoke-20261022-100021                L4         failed     start   8.5 s  run     4.8 s  $0.001
```

## The run folder

`/runs/<id>/` on the Volume `<app>-runs` holds:

- `recipe.json`: the recipe after overrides.
- `record.json`: `t_start`, a `t_<name>` for each `run.clock.mark(name)`,
  `t_end`, `gpu_name` (the device from `nvidia-smi`), `result`, and `error`
  if the job raised.
- the files that the job wrote to `run.dir`.

All times are Unix seconds from `time.time()`. `t_start` minus the local
`t_call` is the cold start, which Modal does not bill. `t_end` minus
`t_start` is the billed time, priced at `ml.PRICE`, Modal's GPU prices on
2026-10-21, times the GPU count. CPU and memory are left out.

## Benchmarks across GPUs

For a benchmark that makes many calls and compares them, use the pieces
directly. `ml.Clock` records the marks, `ml.measure` makes the calls
concurrently and writes one JSON row per call with `t_call`, and
`ml.compile_cache` keeps the compile cache.
[`examples/clock_matmul.py`](examples/clock_matmul.py) times a cold call and
a warm call:

```sh
uv run modal run examples/clock_matmul.py --gpus L4,H100
```

`compile_cache()` points `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` at
`/tmp/cache` and unpacks `/cache/cache.tar` there if the Volume has it. Enter
it before `import torch`. With `fill=<volume>`, the exit packs the cache and
commits it.

The cache is one tar file because a Modal Volume reads about 0.4 s per file,
and an Inductor cache can have thousands of files. In one case, 5,165 files
took 33 minutes to read one by one and 1 to 5 s to unpack as one tar.

`cache_volume(app, gpu)` keys the Volume by the GPU type that you ask for.
Modal sometimes gives a different variant of that type (an A100 80GB for an
A100 40GB, an A10G for an A10), and the cache misses on a different device.
Group results by `gpu_name`.

## The examples in this repo

This repo's `pyproject.toml` points `mlib` at
[`examples/jobs.py`](examples/jobs.py), a small MLP job, and
[`examples/recipes.json`](examples/recipes.json):

```sh
uv sync
uv run mlib deploy
uv run mlib run smoke --wait
```

## Used by

- [speedrun-rta](https://github.com/crsmithdev/speedrun-rta): real time and
  cost of ML training speedruns.
