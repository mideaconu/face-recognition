import sys

import unittest
import numpy as np

from comparative_feret_algorithms import dimensionality_reduction as dr

rng = np.random.RandomState(42)
data = np.dot(rng.normal(0, 1, size=(5000, 2)), np.array([[1, 0], [2, 2]]))


class PCATest(unittest.TestCase):

    """ Input parameter test: data """

    def test_data_float(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(0., 2)

    def test_data_list(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA([], 2)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(np.array([]), 2)

    def test_data_nonnumeric(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(np.array([[0, 0], [0, "a"]]), 2)

    def test_data_dimensionality(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(np.array([[0, 0], [0]]), 2)

    """ Input parameter test: n_components """

    def test_n_components_float(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(data, 2.)

    def test_n_components_complex(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(data, 2j)

    def test_n_components_string(self):
        with self.assertRaises(ValueError):
            self.pca = dr.PCA(data, "a")

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