import setuptools
import sys


# install requirements for mixed precision training
if "amp" in sys.argv:
    sys.argv.remove("amp")

    from pip import __version__ as PIP_VERSION

    PIP_MAJOR, PIP_MINOR = [int(v) for v in PIP_VERSION.split(".")[:2]]

    if PIP_MAJOR <= 9:
        raise RuntimeError(
            "Current version of pip is not compatible with `apex`,"
            "you may need to install `apex` manually."
        )
    elif 10 <= PIP_MAJOR <= 19 and PIP_MINOR < 3:
        from pip._internal import main as pipmain
    else:
        from pip._internal.main import main as pipmain

    pipmain(
        [
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
    version="0.1.5",
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
        "tests": ["pytest", "pytest-cov"],
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "pep8-naming",
            "torchvision",
            "ipywidgets",
        ],
    },
)
