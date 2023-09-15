#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="gcbf",
    version="0.0.0",
    description='PyTorch Official Implementation of CoRL 2023 Paper: : S Zhang, K Garg, C Fan: '
                '"Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control"',
    author="Songyuan Zhang",
    author_email="szhang21@mit.edu",
    url="https://github.com/MIT-REALM/gcbf",
    install_requires=[],
    packages=find_packages(),
)
