# setup.py
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="src",
    version="1.0",
    description="ELECTRA",
    author="Jonas Elsborg",
    author_email="jels@dtu.dk",
    url="https://github.com/Jotels/ELECTRA_UM",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["lightning"],
    packages=find_packages(),
)