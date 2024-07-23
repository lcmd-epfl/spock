import io

# Read the contents of your README file
from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="spock",
    packages=["navicat_spock"],
    version="0.0.2",
    description="Volcano Plot fitting tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="rlaplaza, lcmd-epfl, ace-ethz, dcl-ethz",
    author_email="laplazasolanas@gmail.com",
    url="https://github.com/rlaplaza/spock",
    keywords=["compchem"],
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "setuptools",
        "scikit-learn",
        "piecewise-regression",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "navicat_spock = navicat_spock.spock:run_spock",
        ],
    },
)
