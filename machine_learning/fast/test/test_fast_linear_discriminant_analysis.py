import sys

import unittest
import numpy as np

from machine_learning.fast import dimensionality_reduction as dr

rng = np.random.RandomState(42)
data = np.dot(np.concatenate((rng.normal(-7, 2, size=(5000, 2)), rng.normal(7, 2, size=(5000, 2))), axis=0), np.array([[-1, -2], [2, 1]]))
labels = [0] * 5000 + [1] * 5000


class LDATest(unittest.TestCase):

    """ Input parameter test: data """

    def test_data_null(self):
        with self.assertRaises(ValueError):
            self.lda = dr.LDA(None, [], 2)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            self.lda = dr.LDA(np.array([]), [], 2)

    """ Input parameter test: labels """

    def test_labels_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            self.lda = dr.LDA(data, [0, 1], 2)

    """ Input parameter test: n_components """

    def test_n_components_negative(self):
        with self.assertRaises(ValueError):
            self.lda = dr.LDA(data, labels, -1)

    def test_n_components_zero(self):
        with self.assertRaises(ValueError):
            self.lda = dr.LDA(data, labels, 0)

    def test_n_components_n_1(self):
        with self.assertRaises(ValueError):
            self.lda = dr.LDA(data, labels, data.shape[1]+1)

if __name__ == "__main__":
    unittest.main()