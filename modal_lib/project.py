"""The project file modal.json, recipe overrides, and GPU prices."""

import copy
import importlib.util
import json
import pathlib

import modal

# Modal GPU prices in dollars per second per GPU, from modal.com/pricing on 2026-10-21.
# CPU and memory are left out.
PRICE = {
    "T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542,
    "A100-40GB": 0.000583, "A100-80GB": 0.000694, "RTX-PRO-6000": 0.000842,
    "H100": 0.001097, "H200": 0.001261, "B200": 0.001736,
}

# nvidia-smi names to PRICE keys. The first substring that matches wins.
DEVICES = [
    ("T4", "T4"), ("L40S", "L40S"), ("L4", "L4"), ("A10", "A10"),
    ("A100-SXM4-40GB", "A100-40GB"), ("A100-PCIE-40GB", "A100-40GB"), ("A100", "A100-80GB"),
    ("RTX PRO 6000", "RTX-PRO-6000"), ("H100", "H100"), ("H200", "H200"), ("B200", "B200"),
]


def device_type(name: str) -> str | None:
    """The PRICE key of an nvidia-smi device name, or None if it is unknown."""
    return next((key for part, key in DEVICES if part in name), None)


def load(path: str = "modal.json") -> dict:
    return json.loads(pathlib.Path(path).read_text())


def recipe(project: dict, name: str, sets: list[str]) -> dict:
    """The recipe `name` with each `a.b=v` in `sets` applied.

    A value is JSON if it parses, else a string. Values under `env` are
    always strings.
    """
    r = copy.deepcopy(project["recipes"][name])
    for s in sets:
        path, value = s.split("=", 1)
        keys = path.split(".")
        if keys[0] != "env":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        d = r
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return r


def image(project: dict) -> modal.Image:
    """The project's image from `path.py:attr`, with this package added."""
    path, attr = project["image"].split(":")
    spec = importlib.util.spec_from_file_location("modal_lib_image", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr).add_local_python_source("modal_lib")


def volumes(project: dict, r: dict) -> dict:
    """The project's Volumes, the runs Volume at /runs, and the cache Volume at /cache."""
    vs = {path: modal.Volume.from_name(name, create_if_missing=True)
          for path, name in project.get("volumes", {}).items()}
    vs["/runs"] = modal.Volume.from_name(f"{project['app']}-runs", create_if_missing=True)
    if r.get("cache"):
        vs["/cache"] = modal.Volume.from_name(f"{project['app']}-cache", create_if_missing=True)
    return vs


def secrets(project: dict) -> list:
    return [modal.Secret.from_name(name) for name in project.get("secrets", [])]
