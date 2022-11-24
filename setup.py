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
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="remove, background, u2net",
    packages=["rembg"],
    python_requires=">=3.7, <3.11",
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
