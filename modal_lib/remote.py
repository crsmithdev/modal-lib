"""The container side: run a recipe's command, and keep its log, events, GPU samples and record."""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from importlib import metadata

import modal

from .project import PRICE, device_type

BEAT = 30  # seconds between heartbeats; each one commits the runs Volume


class Run:
    """One run of a recipe's command, with its folder at /runs/<id>."""

    def __init__(self, app: str, run_id: str, kind: str, name: str, r: dict, t_call: float) -> None:
        now = time.time()
        uptime = float(open("/proc/uptime").read().split()[0])
        self.app, self.r = app, r
        self.dir = f"/runs/{run_id}"
        os.makedirs(self.dir, exist_ok=True)
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.record = dict(id=run_id, app=app, kind=kind, name=name, recipe=r, t_call=t_call,
                           t_container=now - uptime, t_enter=now, phases={}, status="running")
        self.gpu()
        self.event("call", t=t_call)
        self.event("container", t=self.record["t_container"])

    def gpu(self) -> None:
        """Record the GPU type and count that the container actually has."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.split("\n")
        except FileNotFoundError:
            return
        names = [n.strip() for n in out if n.strip()]
        if names:
            self.record.update(gpu_name=names[0], gpu_type=device_type(names[0]), gpu_count=len(names))

    def event(self, event: str, **fields) -> None:
        fields.setdefault("t", time.time())
        with self.lock, open(f"{self.dir}/events.jsonl", "a") as f:
            f.write(json.dumps(dict(event=event, **fields)) + "\n")

    def save(self, status: str) -> None:
        """Write record.json with the durations and cost up to now, and commit the Volume."""
        rec = self.record
        with self.lock:
            if status == "running" and self.done.is_set():
                return  # a late heartbeat must not overwrite the exit
            now = time.time()
            rec.update(status=status, t_beat=now, seconds=now - rec["t_container"])
            price = PRICE.get(rec.get("gpu_type"))
            rec["cost"] = rec["seconds"] * price * rec["gpu_count"] if price else None
            rec["durations"] = durations(rec)
            with open(f"{self.dir}/record.json", "w") as f:
                json.dump(rec, f, indent=2)
        modal.Volume.from_name(f"{self.app}-runs").commit()

    def beat(self) -> None:
        self.save("running")
        while not self.done.wait(BEAT):
            self.event("heartbeat")
            self.save("running")

    def start(self) -> subprocess.Popen:
        """Start the command, the GPU sampler and the heartbeat."""
        env = dict(os.environ, PYTHONUNBUFFERED="1", **self.r.get("env", {}))
        if self.r.get("cache"):
            env.update(self.cache_in())
        if shutil.which("nvidia-smi"):
            with open(f"{self.dir}/gpu.csv", "w") as f:
                f.write("time,index,util_pct,memory_mib,power_w\n")
                f.flush()
                self.smi = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits", "-l", "5"], stdout=f)
        self.record["t_cmd"] = time.time()
        self.event("cmd", cmd=self.r["cmd"])
        proc = subprocess.Popen(self.r["cmd"], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self.beat, daemon=True).start()
        return proc

    def follow(self, proc: subprocess.Popen) -> int:
        """Log each line of the command with its time, mark phases, and record the exit."""
        pending = {name: re.compile(rx) for name, rx in self.r.get("phases", {}).items()}
        with open(f"{self.dir}/log.txt", "a") as log:
            for line in proc.stdout:
                t = time.time()
                log.write(f"{t:.3f} {line}")
                log.flush()
                sys.stdout.write(line)
                for name, rx in list(pending.items()):
                    if rx.search(line):
                        with self.lock:
                            self.record["phases"][name] = t
                        self.event("phase", name=name, t=t)
                        del pending[name]
        code = proc.wait()
        self.done.set()
        if hasattr(self, "smi"):
            self.smi.terminate()
        self.record.update(t_exit=time.time(), exit_code=code)
        self.event("exit", code=code, t=self.record["t_exit"])
        if code == 0 and self.r.get("cache"):
            self.cache_out()
        self.save("ok" if code == 0 else "failed")
        return code

    def cache_key(self) -> str:
        try:
            torch = metadata.version("torch")
        except metadata.PackageNotFoundError:
            torch = "none"
        return re.sub(r"[^A-Za-z0-9._-]", "_", f"{torch}-{self.record.get('gpu_name', 'cpu')}")

    def cache_in(self) -> dict:
        """Unpack the compile cache for this torch and device, and point the command at it.

        The cache is one tar file: a Volume takes about 0.4 s per file, and a
        compile cache can have thousands.
        """
        tar = f"/cache/{self.cache_key()}.tar"
        if os.path.exists(tar):
            shutil.unpack_archive(tar, "/tmp/cache")
        self.event("cache", hit=os.path.exists(tar))
        return dict(TORCHINDUCTOR_CACHE_DIR="/tmp/cache/inductor", TRITON_CACHE_DIR="/tmp/cache/triton")

    def cache_out(self) -> None:
        """Save the compile cache if this torch and device have none yet."""
        tar = f"/cache/{self.cache_key()}.tar"
        if os.path.exists(tar) or not os.path.exists("/tmp/cache"):
            return
        shutil.make_archive("/tmp/cache", "tar", "/tmp/cache")
        shutil.copy("/tmp/cache.tar", tar + ".tmp")
        os.replace(tar + ".tmp", tar)
        modal.Volume.from_name(f"{self.app}-cache").commit()


def durations(rec: dict) -> dict:
    """Seconds from the call to the container, to the command, and in each phase.

    A phase starts at the first line that matches its pattern, and ends at
    the start of the next phase or at the exit.
    """
    d = dict(start=rec["t_container"] - rec["t_call"])
    if "t_cmd" not in rec:
        return d
    d["setup"] = rec["t_cmd"] - rec["t_container"]
    marks = sorted(rec["phases"].items(), key=lambda kv: kv[1])
    end = rec.get("t_exit", rec["t_beat"])
    for (name, t), nxt in zip(marks, [t for _, t in marks[1:]] + [end]):
        d[name] = nxt - t
    d["cmd"] = end - rec["t_cmd"]
    return d


def job(app: str, name: str, r: dict, t_call: float) -> int:
    run = Run(app, modal.current_function_call_id(), "job", name, r, t_call)
    try:
        return run.follow(run.start())
    except BaseException as e:
        run.event("error", error=repr(e)[:200])
        run.save("failed")
        raise


def serve(app: str, run_id: str, name: str, r: dict, t_call: float) -> None:
    """Start the command in the background; Modal routes the port to it."""
    run = Run(app, run_id, "serve", name, r, t_call)
    threading.Thread(target=run.follow, args=(run.start(),), daemon=True).start()
