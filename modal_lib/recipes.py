"""A catalog of recipes in one JSON file, resolved on the local side.

    {"default": "smoke", "recipes": {"smoke": {"gpu": "L4", "env": {"TRIALS": "2"}}, ...}}

A recipe is a plain dict. The library reads four keys and passes the whole
dict to the job:

    gpu       a Modal GPU spec, "H100" or "H100:8"
    timeout   seconds, default 3600
    env       environment variables for the container
    function  the Modal function to call, if not the project's default
"""

import json
import pathlib


def load(path: str | pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def resolve(catalog: dict, key: str | None = None) -> tuple[str, dict]:
    """The recipe named `key`, a unique prefix of one, or the catalog default."""
    names = catalog["recipes"]
    key = key or catalog["default"]
    if key not in names:
        matches = [k for k in names if k.startswith(key)]
        if len(matches) != 1:
            raise SystemExit(f"recipe {key!r} matches {matches or 'nothing'}; known: {', '.join(names)}")
        key = matches[0]
    return key, names[key]


def override(recipe: dict, sets: list[str]) -> dict:
    """A copy of `recipe` with each "a.b=value" applied. A value is JSON if it parses."""
    out = json.loads(json.dumps(recipe))
    for s in sets:
        path, _, raw = s.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, leaf = path.split(".")
        d = out
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = value
    return out


def gpu_spec(gpu: str) -> tuple[str, int]:
    """("H100", 8) for "H100:8", and ("L4", 1) for "L4"."""
    kind, _, count = gpu.partition(":")
    return kind, int(count or 1)
