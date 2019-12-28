import sys

import unittest
import numpy as np

from machine_learning.decomposition import PCA

rng = np.random.RandomState(42)
data = rng.normal(0, 1, size=(5000, 2))

class PCATest(unittest.TestCase):

    def setUp(self):
        self.pca = PCA(n_components=3, method="svd", n_oversamples=10, n_iter=2)

    """ Constructor input parameter test """

    def test_n_components_type(self):
        with self.assertRaises(TypeError):
            self.pca = PCA(n_components="1")

    def test_n_components_value(self):
        with self.assertRaises(ValueError):
            self.pca = PCA(n_components=-1)

    def test_method_value(self):
        with self.assertRaises(ValueError):
            self.pca = PCA(n_components=1, method="qr")

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

    """ Fit input parameter test """

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            self.pca.fit(np.array([]))

    def test_n_components_v_features(self):
        with self.assertRaises(ValueError):
            self.pca.fit(data)

    """ Setter input parameter test """

    def test_n_components_setter_type(self):
        with self.assertRaises(TypeError):
            self.pca.n_components = "1"

    def test_n_components_setter_value(self):
        with self.assertRaises(ValueError):
            self.pca.n_components = -1

    def test_method_setter_value(self):
        with self.assertRaises(ValueError):
            self.pca.method = "qr"

    def test_n_oversamples_setter_type(self):
        with self.assertRaises(TypeError):
            self.pca.n_oversamples = "1"

    def test_n_oversamples_setter_value(self):
        with self.assertRaises(ValueError):
            self.pca.n_oversamples = -1

    def test_n_iter_setter_type(self):
        with self.assertRaises(TypeError):
            self.pca.n_iter = "1"

    def test_n_iter_setter_value(self):
        with self.assertRaises(ValueError):
            self.pca.n_iter = -1

if __name__ == "__main__":
    unittest.main() 