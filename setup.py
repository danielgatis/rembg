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
        "Programming Language :: Python :: 3.11",
    ],
    keywords="remove, background, u2net",
    packages=["rembg", "rembg.sessions", "rembg.commands"],
    python_requires=">=3.8, <3.12",
    install_requires=[
        "aiohttp",
        "asyncer",
        "click",
        "fastapi",
        "filetype",
        "gradio",
        "imagehash",
        "numpy",
        "onnxruntime",
        "opencv-python-headless",
        "pillow",
        "pooch",
        "pymatting",
        "python-multipart",
        "scikit-image",
        "scipy",
        "tqdm",
        "uvicorn",
        "watchdog",
    ],
    entry_points={
        "console_scripts": [
            "rembg=rembg.cli:main",
        ],
    },
    extras_require={
        "gpu": ["onnxruntime-gpu"],
    },
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
