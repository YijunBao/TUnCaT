#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='tuncat',
    version='1.1.1',
    description=(
        'Temporal Unmixing of Calcium Traces (TUnCaT) is an automatic algorithm to decontaminate false transients from temporal traces generated from fluorescent calcium imaging videos. '
    ),
    long_description=open('README.md').read(),
    # long_description=long_description,
    long_description_content_type="text/markdown",
    author='Yijun Bao',
    author_email='yijun.bao@duke.edu',
    maintainer='Yijun Bao',
    maintainer_email='yijun.bao@duke.edu',
    license='the GNU License, Version 2.0',
    packages=find_packages(), # ['tuncat'], # 
    platforms=["all"],
    url='https://github.com/YijunBao/TUnCaT',
    project_urls={
        "Source Code": "https://github.com/YijunBao/TUnCaT",
        "Citation": "https://doi.org/10.5281/zenodo.5764576",
        "Paper": "https://doi.org/10.3389/fnins.2021.797421"
    },
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn <1.2',
        'numba',
        'h5py',
        'matplotlib'
    ],
    python_requires='>=3'
    )