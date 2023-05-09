#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="ml",
    version="0.1",
    description="Hacky benchmarking repo to test MPS with pytorch (and lightning).",
    author="Mizaimao",
    packages=find_packages(include=["ml", "ml.*"]),
)
