import unittest
import numpy as np
import sys
sys.path.append("..")
from dimensionality_reduction import LDA

rng = np.random.RandomState(23)
data = np.dot(np.concatenate((rng.normal(-7, 2, size=(5000, 2)), rng.normal(7, 2, size=(5000, 2))), axis=0), np.array([[-1, -2], [2, 1]]))
labels = [0] * 5000 + [1] * 5000

class LDATest(unittest.TestCase):

    def setUp(self):
        self.lda = LDA(data, labels, 1)

    def test_data_int(self):
        with self.assertRaises(ValueError):
            self.lda_ = LDA(0, labels, 1)

    def test_data_float(self):
        with self.assertRaises(ValueError):
            self.lda_ = LDA(0., labels, 1)

    def test_data_list(self):
        with self.assertRaises(ValueError):
            self.lda_ = LDA([], labels, 1)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            self.lda_ = LDA(np.array([]), labels, 1)

    def test_n_components_negative(self):
        with self.assertRaises(ValueError):
            self.lda_ = LDA(data, labels, -1)

    def test_n_components_zero(self):
        with self.assertRaises(ValueError):
            self.lda_ = LDA(data, labels, 0)

    def test_n_components_n_1(self):
        with self.assertRaises(ValueError):
            self.lda_ = LDA(data, labels, data.shape[1]+1)

    def test_eigen(self):
        for i in range(data.shape[1]):
            np.testing.assert_allclose(np.dot(np.cov(self.lda.get_transformation_matrix()), self.lda.get_eigvec()[i]), np.dot(self.lda.get_eigval()[i], self.lda.get_eigvec()[i]), atol=1e-10)

    def test_sorted_eigval(self):
        np.testing.assert_equal(self.lda.get_eigval(), sorted(self.lda.get_eigval(), reverse=True))

    def test_components_shape(self):
        assert self.lda.get_components().shape == (1, 2)

if __name__ == "__main__":
    unittest.main()