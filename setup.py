import os
import pathlib
import sys

sys.path.append(os.path.dirname(__file__))
from setuptools import find_packages, setup

import versioneer

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    requireds = f.read().splitlines()

if os.getenv("GPU") is None:
    with open("requirements-cpu.txt") as f:
        requireds += f.read().splitlines()
else:
    with open("requirements-gpu.txt") as f:
        requireds += f.read().splitlines()

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
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="remove, background, u2net",
    packages=["rembg"],
    python_requires=">=3.8, <4",
    install_requires=requireds,
    entry_points={
        "console_scripts": [
            "rembg=rembg.cli:main",
        ],
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
