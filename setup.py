import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="torch-lr-finder",
    version="0.0.3",
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
    python_requires=">=2.7",
    install_requires=["matplotlib", "numpy", "torch>=0.4.1", "tqdm"],
)
