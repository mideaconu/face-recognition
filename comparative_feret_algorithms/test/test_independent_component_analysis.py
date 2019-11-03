import unittest
import numpy as np
import sys
sys.path.append("..")
from dimensionality_reduction import ICA

rng = np.random.RandomState(23)
data = np.dot(rng.standard_t(1.5, size=(15000, 2)), np.array([[1, 0], [1, 2]]))

class PCATest(unittest.TestCase):

    def setUp(self):
        self.ica = ICA(data, 2)

    def test_data_int(self):
        with self.assertRaises(ValueError):
            self.ica_ = ICA(0, 2)

    def test_data_float(self):
        with self.assertRaises(ValueError):
            self.ica_ = ICA(0., 2)

    def test_data_list(self):
        with self.assertRaises(ValueError):
            self.ica_ = ICA([], 2)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            self.ica_ = ICA(np.array([]), 2)

    def test_n_components_negative(self):
        with self.assertRaises(ValueError):
            self.ica_ = ICA(data, -1)

    def test_n_components_zero(self):
        with self.assertRaises(ValueError):
            self.ica_ = ICA(data, 0)

    def test_n_components_n_1(self):
        with self.assertRaises(ValueError):
            self.ica_ = ICA(data, data.shape[1]+1)

    def test_whitening_matrix(self):
        np.testing.assert_allclose(np.linalg.inv(np.cov(data.T)), np.dot(self.ica.get_whitening_matrix(), self.ica.get_whitening_matrix()))

    def test_whitened_data(self):
        np.testing.assert_allclose(np.cov(self.ica.get_whitened_data()), np.eye(data.shape[1]), atol=1e-10)

    def test_components(self):
        np.testing.assert_allclose(np.dot(self.ica.get_raw_components().T, self.ica.get_raw_components()), np.eye(data.shape[1]), atol=1e-10)

    def test_components_shape(self):
        assert self.ica.get_components().shape == (2, 2)

if __name__ == "__main__":
    unittest.main()