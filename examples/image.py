import pathlib

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .workdir("/root/example")
    .add_local_file(pathlib.Path(__file__).parent / "train.py", "/root/example/train.py")
)
