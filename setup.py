import setuptools
import sys


if "apex" in sys.argv:
    sys.argv.remove("apex")

    # install requirements for mixed precision training
    import subprocess
    import torch

    TORCH_MAJOR = int(torch.__version__.split(".")[0])

    if TORCH_MAJOR == 0:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/NVIDIA/apex",
                "-v",
                "--no-cache-dir",
            ]
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/NVIDIA/apex",
                "-v",
                "--no-cache-dir",
                "--global-option=--cpp_ext",
                "--global-option=--cuda_ext",
            ]
        )

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="torch-lr-finder",
    version="0.2.0",
    author="David Silva",
    author_email="davidtvs10@gmail.com",
    description="Pytorch implementation of the learning rate range test",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidtvs/pytorch-lr-finder",
    packages=setuptools.find_packages(exclude=["examples", "images"]),
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.5.9",
    install_requires=["matplotlib", "numpy", "torch>=0.4.1", "tqdm", "packaging"],
    extras_require={
        "tests": ["pytest", "pytest-cov", "pytest-mock"],
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "flake8",
            "black",
            "pep8-naming",
            "torchvision",
            "ipywidgets",
        ],
    },
)
