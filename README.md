# modal-lib

Run and serve commands on [Modal](https://modal.com) GPUs from recipes in one
project file, and record the time, GPU use and cost of each run.

- **Jobs:** a command, such as a training script, runs in a fresh container
  and keeps running after the terminal exits.
- **Servers:** a command that listens on a port, such as ComfyUI, gets a
  Modal URL. `mlib serve` records the time to the first good response.
- **Runs:** each run has a folder on a Modal Volume with its log, events,
  GPU samples and a record of phase times and cost. `mlib ps` lists the runs
  from any machine.

## Install

```sh
uv add "modal-lib @ git+https://github.com/crsmithdev/modal-lib"
uv run modal setup   # once
```

## The project file

Put `modal.json` in the project root and run `mlib` there:

```json
{
  "app": "imagegen",
  "image": "image.py:image",
  "volumes": {"/models": "comfy-models"},
  "secrets": ["huggingface"],
  "recipes": {
    "lora": {"kind": "job", "gpu": "H100", "timeout": 7200,
             "cmd": ["accelerate", "launch", "train.py"], "env": {"SEED": "1"},
             "phases": {"load": "^loading", "train": "^step 1/"}, "cache": true},
    "comfy": {"kind": "serve", "gpu": "L40S", "port": 8188, "idle": 300,
              "cmd": ["comfy", "launch", "--", "--listen"], "ready": "/system_stats"}
  }
}
```

- `image` is `file.py:attr`, a `modal.Image`. Its Python version must be the
  local one, because modal-lib sends its functions to the container
  serialized. Set the working directory of the command with `.workdir()`.
- `volumes` and `secrets` are names. Every recipe gets them.
- A recipe has `cmd` and `gpu`, and optionally `env` (strings), `timeout`
  in seconds, `retries`, `phases` and `cache`.
- `retries` is the count of times that Modal starts a job again in a new
  container when its container stops, for example on a lost machine. A
  command that exits with a code that is not 0 is not retried.
- `phases` maps a phase name to a regular expression. A phase starts at
  the first output line that matches, and ends at the next phase or the exit.
- `cache: true` keeps the `torch.compile` cache on the Volume `<app>-cache`,
  as one tar file per torch version and device. A Modal Volume reads about
  0.4 s per file, and an Inductor cache can have thousands of files.
- A `serve` recipe has `port`, and optionally `ready` (the path to poll,
  `/` by default) and `idle` (seconds before the container stops, 300 by
  default).

## Commands

```sh
mlib run lora                      # start a job and print its ID
mlib run lora --set env.SEED=2 --set gpu=H200 --wait   # wait; exit with the job's exit code
mlib serve comfy                   # deploy, print the URL, and time the cold start
mlib ps                            # every run, with status, times and cost
mlib logs fc-01M5...               # a run's log.txt
mlib stop fc-01M5...               # cancel a job, or stop a server's app
mlib pull fc-01M5...               # download the run's folder to runs/<id>
```

`--set a.b=v` changes one recipe field. A value is JSON if it parses, else a
string. Values under `env` are always strings.

## The run folder

A job's ID is its Modal function call ID. A server's ID is the recipe name
and the deploy time. The folder is `/<id>/` on the Volume `<app>-runs`:

| File | Content |
| - | - |
| `log.txt` | each output line of the command, after its time |
| `events.jsonl` | the call, the container start, the command start, each phase, a heartbeat every 30 s, and the exit |
| `gpu.csv` | `nvidia-smi` every 5 s: utilization, memory and power |
| `record.json` | the durations, the GPU that the container got, the status and the cost |
| `ready.json` | for a server: the URL, and the deploy and ready times |

The durations are `start` (call to container start), `setup` (container
start to command start), each phase, and `cmd` (command start to exit).

A retry writes to the same folder. The log and the events continue, and
`record.json` keeps each earlier attempt under `attempts`, with the status
`lost` if its container stopped. `restart` is the time from the last
heartbeat of the earlier attempt to the new container, and the cost is the
sum of all the attempts. The command gets `MLIB_RUN_DIR`, its run folder,
and `MLIB_ATTEMPT`, 0 for the first attempt. Files that the command writes
to the run folder are on the Volume at the next heartbeat, so a new
attempt can read a checkpoint that the earlier one wrote.

The status is `running`, `ok`, `failed` (an exit code that is not 0, which
includes a job that times out or that `mlib stop` cancels), or `lost` (no
heartbeat for 90 s: the container stopped before it could record the exit).
A server with no heartbeat is `idle`. The
heartbeat commits the Volume, so the log and the record are at most 30 s old
while the run is running.

The cost is the container time from its start × the price of the GPU that it
got × the count, at the Modal prices of 2026-10-21. CPU and memory are left
out. The cost is empty for a GPU that is not in the price table.

## The example

```sh
cd examples
uv run mlib run smoke --wait   # 20 one-second steps on an L4, about $0.006
uv run mlib run fail --wait    # exits 3
uv run mlib run lost           # stops its container at step 35; Modal retries it once
uv run mlib serve web          # python -m http.server on an L4
```

## Used by

- [speedrun-rta](https://github.com/crsmithdev/speedrun-rta): real time and
  cost of ML training speedruns.
