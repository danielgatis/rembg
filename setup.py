import os
import pathlib
import sys

sys.path.append(os.path.dirname(__file__))
from setuptools import find_packages, setup

import versioneer

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="remove, background, u2net",
    packages=["rembg"],
    python_requires=">3.7, <3.11",
    install_requires=[
        "aiohttp~=3.8.1",
        "asyncer~=0.0.2",
        "click~=8.1.3",
        "fastapi~=0.87.0",
        "filetype~=1.2.0",
        "pooch~=1.6.0",
        "imagehash~=4.3.1",
        "numpy~=1.23.5",
        "onnxruntime~=1.13.1",
        "opencv-python-headless~=4.6.0.66",
        "pillow~=9.3.0",
        "pymatting~=1.1.8",
        "python-multipart~=0.0.5",
        "scikit-image~=0.19.3",
        "scipy~=1.9.3",
        "tqdm~=4.64.1",
        "uvicorn~=0.20.0",
        "watchdog~=2.1.9",
    ],
    entry_points={
        "console_scripts": [
            "rembg=rembg.cli:main",
        ],
    },
    extras_require={
        "gpu": ["onnxruntime-gpu~=1.13.1"],
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
