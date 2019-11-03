import unittest
import numpy as np
import sys
sys.path.append("..")
from dimensionality_reduction import PCA

rng = np.random.RandomState(23)
data = np.dot(rng.normal(0, 1, size=(5000, 2)), np.array([[1, 0], [2, 2]]))

class PCATest(unittest.TestCase):

    def setUp(self):
        self.pca = PCA(data, 2)

    def test_data_int(self):
        with self.assertRaises(ValueError):
            self.pca_ = PCA(0, 2)

    def test_data_float(self):
        with self.assertRaises(ValueError):
            self.pca_ = PCA(0., 2)

    def test_data_list(self):
        with self.assertRaises(ValueError):
            self.pca_ = PCA([], 2)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            self.pca_ = PCA(np.array([]), 2)

    def test_n_components_negative(self):
        with self.assertRaises(ValueError):
            self.pca_ = PCA(data, -1)

    def test_n_components_zero(self):
        with self.assertRaises(ValueError):
            self.pca_ = PCA(data, 0)

    def test_n_components_n_1(self):
        with self.assertRaises(ValueError):
            self.pca_ = PCA(data, data.shape[1]+1)

    def test_scatter_matrix(self):
        data_ = data - np.mean(data, axis=0)
        np.testing.assert_allclose(self.pca.get_scatter_matrix(), np.dot(np.transpose(data_), data_) / (data_.shape[0]-1))

    def test_eigen(self):
        for i in range(data.shape[1]):
            np.testing.assert_allclose(np.dot(np.cov(np.transpose(data)), self.pca.get_eigvec()[i]), np.dot(self.pca.get_eigval()[i], self.pca.get_eigvec()[i]))

    def test_sorted_eigval(self):
        np.testing.assert_equal(self.pca.get_eigval(), sorted(self.pca.get_eigval(), reverse=True))

    def test_components_shape(self):
        assert self.pca.get_components().shape == (2, 2)

    def test_components_variance(self):
        assert self.pca.get_explained_variance() >= 0 and self.pca.get_explained_variance() <= 100

if __name__ == "__main__":
    unittest.main()