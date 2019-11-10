import sys

import unittest
import collections
import numpy as np

from machine_learning import k_nearest_neighbours as knn

def euclidean(a, b):
    return np.linalg.norm(a - b)


class kNNTest(unittest.TestCase):

    """ Input parameter test: data """
    
    def test_data_null(self):
        with self.assertRaises(ValueError):
            knn_ = knn.kNN(None, ['a', 'b', 'c'], euclidean, 1)

    def test_data_float(self):
        with self.assertRaises(ValueError):
            knn_ = knn.kNN(0., ['a', 'b', 'c'], euclidean, 1)

    def test_data_list(self):
        with self.assertRaises(ValueError):
            knn_ = knn.kNN([], ['a', 'b', 'c'], euclidean, 1)

    def test_data_empty(self):
        with self.assertRaises(ValueError):
            knn_ = knn.kNN(np.array([]), ['a', 'b', 'c'], euclidean, 1)

    def test_data_nonnumeric(self):
        with self.assertRaises(ValueError):
            knn_ = knn.kNN(np.array([0, 0, 'a']), ['a', 'b', 'c'], euclidean, 1)

    def test_data_dimensionality(self):
        knn_ = kNN(np.array([[0, 0], [0]]), ['a', 'b', 'c'], euclidean, 1)

    """ Input parameter test: labels """

    def test_labels_dimension_mismatch(self):
        knn_ = kNN(np.array([0, 0, 0]), ['a', 'b'], euclidean, 1)
    
    def test_null_labels(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), None, euclidean, 1)

    def test_int_type_labels(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), 0, euclidean, 1)

    def test_float_type_labels(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), 0., euclidean, 1)

    def test_str_type_labels(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), "0", euclidean, 1)

    def test_labels_length(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), ['a', 'b'], euclidean, 1)

    def test_noncallable_dist(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), ['a', 'b', 'c'], "euclidean", 1)

    def test_zero_leafsize(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), ['a', 'b', 'c'], euclidean, 0)

    def test_negative_leafsize(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 2]), ['a', 'b', 'c'], euclidean, -1)

    def test_knn_negative_nn(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(-1, np.array([9, 4]))

    def test_knn_n_1_nn(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(data.shape[0]+1, np.array([1, 1]))

    def test_knn_dimension(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(1, np.array([1, 1, 1]))

if __name__ == "__main__":
    unittest.main()