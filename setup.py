#!/usr/bin/env python

import numpy
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extension = [Extension(name="machine_learning.fast.dimensionality_reduction", sources=["machine_learning/fast/dimensionality_reduction.pyx"], include_dirs=[numpy.get_include()]),
             Extension(name="machine_learning.fast.k_nearest_neighbours", sources=["machine_learning/fast/k_nearest_neighbours.pyx"], include_dirs=[numpy.get_include()])]

setup(
    name="comparative_feret_algorithms",
    version="1.0",
    description=("A module containing Machine Learning algorithms "
                 "used in the comparative FERET study project."),
    license="MIT",
    author="Mihai Ionut Deaconu",
    author_email="mihai.ionut.deaconu@gmail.com",
    packages=["machine_learning", "machine_learning.fast"],
    ext_modules=cythonize(extension)
)