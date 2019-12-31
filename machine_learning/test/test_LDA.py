import sys

import unittest
import numpy as np

from machine_learning.decomposition import LDA

rng = np.random.RandomState(42)
data = rng.normal(0, 1, size=(5000, 2))
labels = np.asarray([0] * 2500 + [1] * 2500)

class LDATest(unittest.TestCase):

    def setUp(self):
        self.lda = LDA(n_components=3)

    """ Constructor input parameter test """

    def test_n_components_type(self):
        with self.assertRaises(TypeError):
            self.lda = LDA(n_components="1")

    def test_n_components_value(self):
        with self.assertRaises(ValueError):
            self.lda = LDA(n_components=-1)

    """ Fit input parameter test """

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            self.lda.fit(np.array([]), labels)

    def test_n_components_v_features(self):
        with self.assertRaises(ValueError):
            self.lda.fit(data, labels)

    def test_empty_labels(self):
        with self.assertRaises(ValueError):
            self.lda.fit(data, np.array([]))

    def test_mismached_labels(self):
        with self.assertRaises(ValueError):
            self.lda.fit(data, np.array([0]))

    """ Setter input parameter test """

    def test_n_components_setter_type(self):
        with self.assertRaises(TypeError):
            self.lda.n_components = "1"

    def test_n_components_setter_value(self):
        with self.assertRaises(ValueError):
            self.lda.n_components = -1

if __name__ == "__main__":
    unittest.main() 