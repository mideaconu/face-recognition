import sys

import unittest
import numpy as np

from comparative_feret_algorithms.fast import dimensionality_reduction as dr

rng = np.random.RandomState(42)
data = np.dot(rng.normal(0, 1, size=(5000, 2)), np.array([[1, 0], [2, 2]]))


class PCATest(unittest.TestCase):

    """ Input parameter test: data """

    def test_data_null(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(None, 2)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(np.array([]), 2)

    """ Input parameter test: n_components """

    def test_n_components_negative(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(data, -1)

    def test_n_components_zero(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(data, 0)

    def test_n_components_n_1(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(data, data.shape[1]+1)

if __name__ == "__main__":
    unittest.main()