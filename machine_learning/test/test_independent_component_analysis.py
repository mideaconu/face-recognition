import sys

import unittest
import numpy as np

from machine_learning import dimensionality_reduction as dr

rng = np.random.RandomState(42)
data = np.dot(rng.standard_t(1.5, size=(15000, 2)), np.array([[1, 0], [1, 2]]))


class PCATest(unittest.TestCase):

    """ Input parameter test: data """

    def test_data_null(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(None, 2)

    def test_data_float(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(0., 2)

    def test_data_list(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA([], 2)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(np.array([]), 2)

    """ Input parameter test: n_components """

    def test_n_components_float(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(data, 2.)

    def test_n_components_complex(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(data, 2j)

    def test_n_components_string(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(data, "a")

    def test_n_components_negative(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(data, -1)

    def test_n_components_zero(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(data, 0)

    def test_n_components_n_1(self):
        with self.assertRaises(ValueError):
            self.ica = dr.ICA(data, data.shape[1]+1)

if __name__ == "__main__":
    unittest.main()