#!/usr/bin/env python

import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

extension = [Extension(name="comparative_feret_algorithms.fast.dimensionality_reduction", sources=["comparative_feret_algorithms/fast/dimensionality_reduction.pyx"], include_dirs=[numpy.get_include()])]

setup(
    name="comparative_feret_algorithms",
    version="1.0",
    description=("A module containing Machine Learning algorithms "
                 "used in the comparative FERET study project."),
    license="MIT",
    author="Mihai Ionut Deaconu",
    author_email="mihai.ionut.deaconu@gmail.com",
    packages=["comparative_feret_algorithms", "comparative_feret_algorithms.fast"],
    ext_modules=cythonize(extension)
)