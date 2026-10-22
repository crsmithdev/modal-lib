"""Runs: one call of a job, with an ID, a folder on the runs Volume and a clock record.

In the container, `Run` makes `/runs/<id>/`, writes `recipe.json` there, and
on exit writes `record.json` (the clock marks, the result, and any error) and
commits the Volume. On the local side, `launch` spawns the call and appends a
row to `.modal-lib/runs.jsonl`.
"""

import json
import pathlib
import time

import modal

from .clock import Clock

RUNS_MOUNT = "/runs"
REGISTRY = pathlib.Path(".modal-lib/runs.jsonl")

# Runs in this container so far. A container that Modal reuses for a second
# run started before that run's call, so its t_start is not that run's start.
_runs_here = 0


def runs_volume(app: str) -> modal.Volume:
    return modal.Volume.from_name(f"{app}-runs", create_if_missing=True)


class Run:
    """The context of one run in the container. Set `result` before the exit."""

    def __init__(self, app: str, run_id: str, recipe: dict) -> None:
        self.app, self.id, self.recipe = app, run_id, recipe
        self.dir = pathlib.Path(RUNS_MOUNT) / run_id
        self.clock = Clock()
        self.result: dict = {}
        self.record: dict = {}

    def __enter__(self) -> "Run":
        global _runs_here
        _runs_here += 1
        self.clock.mark("enter")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "recipe.json").write_text(json.dumps(self.recipe, indent=2))
        return self

    def __exit__(self, kind, error, tb) -> None:
        extra = {"reused": True} if _runs_here > 1 else {}
        if error:
            extra["error"] = repr(error)
        self.record = self.clock.done(id=self.id, result=self.result, **extra)
        (self.dir / "record.json").write_text(json.dumps(self.record, indent=2))
        runs_volume(self.app).commit()


def new_id(key: str) -> str:
    return f"{key}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"


def launch(fn: modal.Function, app: str, key: str, recipe: dict) -> dict:
    """Spawn `fn(recipe, run_id)` with the recipe's GPU, timeout and env, and register it.

    `fn` must run in a deployed app, so that the call outlives this process.
    """
    run_id = new_id(key)
    fn = fn.with_options(gpu=recipe.get("gpu"), timeout=recipe.get("timeout", 3600),
                         env=recipe.get("env"), volumes={RUNS_MOUNT: runs_volume(app)})
    t_call = time.time()
    call = fn.spawn(recipe, run_id)
    row = dict(id=run_id, recipe=key, app=app, gpu=recipe.get("gpu"), call=call.object_id, t_call=t_call)
    REGISTRY.parent.mkdir(exist_ok=True)
    with REGISTRY.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row
