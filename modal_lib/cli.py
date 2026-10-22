"""mlib: run a project's recipes on Modal GPUs, and list, read, stop and download the runs.

    mlib run RECIPE [--set a.b=v ...] [--wait]   a job; --wait exits with its exit code
    mlib serve RECIPE [--set a.b=v ...]          deploy a server and time its cold start
    mlib ps                                      every run, from the runs Volume
    mlib logs ID                                 a run's log.txt
    mlib stop ID                                 cancel a job, or stop a server's app
    mlib pull ID                                 download a run's folder to runs/ID

Run it in a project root with a modal.json.
"""

import argparse
import io
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import modal

from . import project, remote


def function(p: dict, name: str, r: dict, **kw) -> dict:
    """The keyword arguments of app.function for a recipe."""
    return dict(image=project.image(p), gpu=r.get("gpu"), volumes=project.volumes(p, r),
                secrets=project.secrets(p), serialized=True, **kw)


def cmd_run(args) -> None:
    p = project.load()
    r = project.recipe(p, args.recipe, args.set)
    app_name, name, t_call = p["app"], args.recipe, time.time()

    def job() -> int:
        return remote.job(app_name, name, r, t_call)

    app = modal.App(f"{app_name}-{name}")
    fn = app.function(name="job", timeout=r.get("timeout", 3600), single_use_containers=True,
                      **function(p, name, r))(job)
    with modal.enable_output(), app.run(detach=True):
        call = fn.spawn()
    print(call.object_id)
    if not args.wait:
        return
    try:
        code = call.get()
    except Exception as e:
        print(f"error: {e!r}"[:200], file=sys.stderr)
        code = 1
    for _ in range(15):  # the container commits its last record just after the call returns
        rec = record(runs(p), call.object_id)
        if rec and rec["status"] != "running":
            break
        time.sleep(2)
    if rec:
        print(summary(rec))
    sys.exit(code)


def cmd_serve(args) -> None:
    p = project.load()
    r = project.recipe(p, args.recipe, args.set)
    app_name, name, t_call = p["app"], args.recipe, time.time()
    run_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"

    def web() -> None:
        remote.serve(app_name, run_id, name, r, t_call)

    app = modal.App(f"{app_name}-{name}")
    fn = app.function(name="web", timeout=r.get("timeout", 86400), max_containers=1,
                      scaledown_window=r.get("idle", 300), **function(p, name, r))(
        modal.web_server(r["port"], startup_timeout=r.get("startup", 600))(web))
    with modal.enable_output():
        app.deploy()
    t_deployed = time.time()
    url = fn.get_web_url()
    print(f"{run_id}  {url}", flush=True)
    while True:
        try:
            with urllib.request.urlopen(url + r.get("ready", "/"), timeout=60) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    t_ready = time.time()
    ready = dict(url=url, t_call=t_call, t_deployed=t_deployed, t_ready=t_ready)
    with runs(p).batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(ready).encode()), f"{run_id}/ready.json")
    print(f"ready: deploy {t_deployed - t_call:.1f} s, cold start {t_ready - t_deployed:.1f} s")


def runs(p: dict) -> modal.Volume:
    return modal.Volume.from_name(f"{p['app']}-runs", create_if_missing=True)


def record(vol: modal.Volume, run_id: str) -> dict | None:
    """The run's record.json, with status `lost` if a job's heartbeat is stale."""
    try:
        rec = json.loads(b"".join(vol.read_file(f"{run_id}/record.json")))
    except FileNotFoundError:
        return None
    if rec["status"] == "running" and time.time() - rec["t_beat"] > 3 * remote.BEAT:
        rec["status"] = "lost" if rec["kind"] == "job" else "idle"
    return rec


def summary(rec: dict) -> str:
    d = rec.get("durations", {})
    gpu = f"{rec.get('gpu_type') or rec.get('gpu_name') or '-'}" + (
        f":{rec['gpu_count']}" if rec.get("gpu_count", 1) > 1 else "")
    cost = f"${rec['cost']:.3f}" if rec.get("cost") is not None else "-"
    return (f"{rec['id']:<30} {rec['name']:<16} {gpu:<12} {rec['status']:<8} "
            f"start {d.get('start', 0):6.1f} s  setup {d.get('setup', 0):6.1f} s  "
            f"cmd {d.get('cmd', 0):8.1f} s  {cost}")


def cmd_ps(args) -> None:
    vol = runs(project.load())
    recs = [rec for e in vol.listdir("/") if (rec := record(vol, e.path))]
    for rec in sorted(recs, key=lambda rec: rec["t_call"]):
        print(summary(rec))


def cmd_logs(args) -> None:
    vol = runs(project.load())
    for chunk in vol.read_file(f"{args.id}/log.txt"):
        sys.stdout.buffer.write(chunk)


def cmd_stop(args) -> None:
    p = project.load()
    rec = record(runs(p), args.id)
    if args.id.startswith("fc-"):
        modal.FunctionCall.from_id(args.id).cancel()
    else:
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", f"{p['app']}-{rec['name']}"],
                       check=True)


def cmd_pull(args) -> None:
    p = project.load()
    pathlib.Path("runs").mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", f"{p['app']}-runs",
                    args.id, "runs/"], check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mlib", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("run", cmd_run), ("serve", cmd_serve)]:
        s = sub.add_parser(name)
        s.add_argument("recipe")
        s.add_argument("--set", action="append", default=[], metavar="A.B=VALUE")
        s.set_defaults(fn=fn)
    sub.choices["run"].add_argument("--wait", action="store_true")
    sub.add_parser("ps").set_defaults(fn=cmd_ps)
    for name, fn in [("logs", cmd_logs), ("stop", cmd_stop), ("pull", cmd_pull)]:
        s = sub.add_parser(name)
        s.add_argument("id")
        s.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
