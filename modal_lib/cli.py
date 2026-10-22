"""mlib: run a project's recipes on Modal, and list, check and download the runs.

    mlib recipes                     the catalog
    mlib show KEY [--set a.b=v ...]  one recipe, with overrides applied
    mlib deploy                      deploy the project's app
    mlib run KEY [--set ...] [--wait]
    mlib runs                        every run, with status, time and cost
    mlib pull ID [DEST]              download a run's folder

Run it in a project root whose pyproject.toml has:

    [tool.modal-lib]
    app = "my-jobs"              # the deployed app name
    file = "jobs.py"             # the app file, for deploy
    function = "train"           # the default job
    recipes = "recipes.json"
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tomllib

from . import recipes
from .clock import cost
from .runs import REGISTRY, launch, runs_volume


def config() -> dict:
    return tomllib.loads(pathlib.Path("pyproject.toml").read_text())["tool"]["modal-lib"]


def recipe(args) -> tuple[str, dict]:
    key, r = recipes.resolve(recipes.load(config()["recipes"]), args.key)
    return key, recipes.override(r, args.set)


def cmd_recipes(args) -> None:
    cat = recipes.load(config()["recipes"])
    for key, r in cat["recipes"].items():
        mark = "*" if key == cat["default"] else " "
        print(f"{mark} {key:<24} {r.get('gpu', '-'):<12} {r.get('note', '')}")


def cmd_show(args) -> None:
    key, r = recipe(args)
    print(key)
    print(json.dumps(r, indent=2))


def cmd_deploy(args) -> None:
    raise SystemExit(subprocess.run(["modal", "deploy", config()["file"]]).returncode)


def cmd_run(args) -> None:
    import modal

    cfg = config()
    key, r = recipe(args)
    fn = modal.Function.from_name(cfg["app"], r.get("function", cfg["function"]))
    row = launch(fn, cfg["app"], key, r)
    print(row["id"])
    if args.wait:
        try:
            modal.FunctionCall.from_id(row["call"]).get()
        except Exception as e:
            print(f"error: {e!r}"[:200])
        print(summary(row, status(row)))


def status(row: dict) -> dict:
    """The run's record from the Volume, or {"status": ...} if it has none yet."""
    import modal

    try:
        record = json.loads(b"".join(runs_volume(row["app"]).read_file(f"{row['id']}/record.json")))
        return dict(status="failed" if "error" in record else "done", **record)
    except FileNotFoundError:
        pass
    try:
        modal.FunctionCall.from_id(row["call"]).get(timeout=0)
        return dict(status="done, no record")
    except TimeoutError:
        return dict(status="running")
    except Exception as e:
        return dict(status=f"crashed: {e!r}"[:80])


def summary(row: dict, s: dict) -> str:
    line = f"{row['id']:<36} {row['gpu'] or '-':<10} {s['status']:<10}"
    if "t_end" in s and s.get("reused"):
        line += f" reused container, run {s['t_end'] - s['t_enter']:7.1f} s"
    elif "t_end" in s:
        billed = s["t_end"] - s["t_start"]
        line += f" start {s['t_start'] - row['t_call']:5.1f} s  run {billed:7.1f} s"
        if row["gpu"]:
            line += f"  ${cost(billed, row['gpu']):.3f}"
    return line


def cmd_runs(args) -> None:
    if not REGISTRY.exists():
        return
    for line in REGISTRY.read_text().splitlines():
        row = json.loads(line)
        print(summary(row, status(row)))


def cmd_pull(args) -> None:
    rows = [json.loads(line) for line in REGISTRY.read_text().splitlines()]
    row = next(r for r in rows if r["id"] == args.id)
    vol = runs_volume(row["app"])
    dest = pathlib.Path(args.dest or pathlib.Path("runs") / args.id)
    for entry in vol.listdir(args.id, recursive=True):
        if entry.type.name != "FILE":
            continue
        path = dest / pathlib.Path(entry.path).relative_to(args.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            vol.read_file_into_fileobj(entry.path, f)
        print(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mlib", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("recipes").set_defaults(fn=cmd_recipes)
    for name, fn in [("show", cmd_show), ("run", cmd_run)]:
        s = sub.add_parser(name)
        s.add_argument("key", nargs="?")
        s.add_argument("--set", action="append", default=[], metavar="A.B=VALUE")
        s.set_defaults(fn=fn)
        if name == "run":
            s.add_argument("--wait", action="store_true")
    sub.add_parser("deploy").set_defaults(fn=cmd_deploy)
    sub.add_parser("runs").set_defaults(fn=cmd_runs)
    s = sub.add_parser("pull")
    s.add_argument("id")
    s.add_argument("dest", nargs="?")
    s.set_defaults(fn=cmd_pull)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
