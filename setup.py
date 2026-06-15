from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="pandatools",
    version="2.1.0",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sơn Lê",
    url="https://github.com/sonbuwin-beep/pandatools",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
