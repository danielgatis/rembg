import os
import pathlib
import sys

sys.path.append(os.path.dirname(__file__))
from setuptools import find_packages, setup

import versioneer

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open(here / "requirements.txt") as f:
    requireds = f.read().splitlines()

with open(here / "requirements-gpu.txt") as f:
    gpu_requireds = f.read().splitlines()

setup(
    name="rembg",
    description="Remove image background",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielgatis/rembg",
    author="Daniel Gatis",
    author_email="danielgatis@gmail.com",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    keywords="remove, background, u2net",
    packages=["rembg"],
    python_requires=">=3.7",
    install_requires=requireds,
    entry_points={
        "console_scripts": [
            "rembg=rembg.cli:main",
        ],
    },
    extras_require={
        "gpu": gpu_requireds,
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
