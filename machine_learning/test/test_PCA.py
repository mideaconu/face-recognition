import sys

import unittest
import numpy as np

from machine_learning.decomposition import PCA

rng = np.random.RandomState(42)
data = np.dot(rng.normal(0, 1, size=(5000, 2)), np.array([[1, 0], [2, 2]]))

class PCATest(unittest.TestCase):

    """ Constructor input parameter test """

    def test_n_components_type(self):
        with self.assertRaises(TypeError):
            self.pca = PCA(n_components="1")

    def test_n_components_value(self):
        with self.assertRaises(ValueError):
            self.pca = PCA(n_components=-1)

    def test_solver_value(self):
        with self.assertRaises(ValueError):
            self.pca = PCA(n_components=1, solver="qr")

    def test_n_oversamples_type(self):
        with self.assertRaises(TypeError):
            self.pca = PCA(n_components=1, n_oversamples="1")

    def test_n_oversamples_value(self):
        with self.assertRaises(ValueError):
            self.pca = PCA(n_components=1, n_oversamples=-1)

    def test_n_iter_type(self):
        with self.assertRaises(TypeError):
            self.pca = PCA(n_components=1, n_iter="1")

    def test_n_iter_value(self):
        with self.assertRaises(ValueError):
            self.pca = PCA(n_components=1, n_iter=-1)

if __name__ == "__main__":
    unittest.main() 