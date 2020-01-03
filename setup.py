#!/usr/bin/env python

import numpy
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extension = [Extension(name="machine_learning.decomposition", 
                       sources=["machine_learning/decomposition.pyx"], 
                       include_dirs=[numpy.get_include()]),
             Extension(name="machine_learning.neighbours", 
                       sources=["machine_learning/neighbours.pyx"], 
                       include_dirs=[numpy.get_include()]), 
             Extension(name="machine_learning.data_structures", 
                       sources=["data_structures/priority_queue.pyx", 
                                "data_structures/KeyedPriorityQueue.cpp"], 
                       language='c++', 
                       include_dirs=[numpy.get_include()])]

setup(
    name="comparative_feret_algorithms",
    version="1.0",
    description=("A module containing Machine Learning algorithms "
                 "used in the comparative FERET study project."),
    license="MIT",
    author="Mihai Ionut Deaconu",
    author_email="mihai.ionut.deaconu@gmail.com",
    packages=["machine_learning"],
    ext_modules=cythonize(extension)
)