from collections.abc import Iterable

import modal


def torch_image(torch: str = "2.11.0", torchvision: str = "0.26.0", cuda: str = "cu128",
                python: str = "3.12", packages: Iterable[str] = ()) -> modal.Image:
    """A Debian image with torch from the PyTorch index and this package.

    Put every build step in `packages`. Modal accepts only more `add_local_*`
    calls after this image.
    """
    image = modal.Image.debian_slim(python_version=python).uv_pip_install(
        f"torch=={torch}", f"torchvision=={torchvision}",
        index_url=f"https://download.pytorch.org/whl/{cuda}",
    )
    if packages:
        image = image.uv_pip_install(*packages)
    return image.add_local_python_source("modal_lib")
