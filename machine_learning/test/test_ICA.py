#
# Author: Mihai-Ionut Deaconu
#

import sys

import unittest
import numpy as np

from machine_learning.decomposition import ICA

rng = np.random.RandomState(42)
data = rng.normal(0, 1, size=(5000, 2))

class ICATest(unittest.TestCase):

    def setUp(self):
        self.ica = ICA(n_components=3, method="symmetric")

    """ Constructor input parameter test """

    def test_n_components_type(self):
        with self.assertRaises(TypeError):
            self.ica = ICA(n_components="1")

    def test_n_components_value(self):
        with self.assertRaises(ValueError):
            self.ica = ICA(n_components=-1)

    def test_method_value(self):
        with self.assertRaises(ValueError):
            self.ica = ICA(n_components=1, method="parallel")

    """ Fit input parameter test """

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            self.ica.fit(np.array([]))

    def test_n_components_v_features(self):
        with self.assertRaises(ValueError):
            self.ica.fit(data)

    """ Setter input parameter test """

    def test_n_components_setter_type(self):
        with self.assertRaises(TypeError):
            self.ica.n_components = "1"

    def test_n_components_setter_value(self):
        with self.assertRaises(ValueError):
            self.ica.n_components = -1

    def test_method_setter_value(self):
        with self.assertRaises(ValueError):
            self.ica.method = "parallel"

if __name__ == "__main__":
    unittest.main() 