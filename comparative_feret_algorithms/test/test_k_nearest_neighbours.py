import sys

import unittest
import collections
import numpy as np

from comparative_feret_algorithms import k_nearest_neighbours as knn

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

    def test_knn_existing_point(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([9, 5])]]), self.knn.find_knn(1, np.array([9, 5])))

    def test_knn_out_of_range(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([1, 1])]]), self.knn.find_knn(1, np.array([-10, -10])))

    def test_knn_pivot(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([4, 6])]]), self.knn.find_knn(1, np.array([4.9375, 5.25])))

    def test_knn_equidistant_nn(self):
        np.testing.assert_array_equal(np.sort(np.array([labels[data.tolist().index([1, 1])], labels[data.tolist().index([2, 2])]]), axis=0), np.sort(self.knn.find_knn(2, np.array([2, 1])), axis=0))

    def test_knn_negative_nn(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(-1, np.array([9, 4]))

    def test_knn_0_nn(self):
        np.testing.assert_array_equal(np.array([]), self.knn.find_knn(0, np.array([9, 4])))

    def test_knn_1_nn(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([9, 5])]]), self.knn.find_knn(1, np.array([9, 4])))

    def test_knn_3_nn(self):
        np.testing.assert_array_equal(np.sort(np.array([labels[data.tolist().index([1, 1])], labels[data.tolist().index([1, 2])], labels[data.tolist().index([2, 2])]]), axis=0), np.sort(self.knn.find_knn(3, np.array([2, 1])), axis=0))

    def test_knn_n_nn(self):
        np.testing.assert_array_equal(np.array(labels), np.sort(self.knn.find_knn(data.shape[0], np.array([1, 1])), axis=0))

    def test_knn_n_1_nn(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(data.shape[0]+1, np.array([1, 1]))

    def test_knn_dimension(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(1, np.array([1, 1, 1]))

    def test_knn_existing_point_leafsize_2(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([9, 5])]]), self.knn_2.find_knn(1, np.array([9, 5])))

    def test_knn_equidistant_nn_leafsize_2(self):
        np.testing.assert_array_equal(np.sort(np.array([labels[data.tolist().index([1, 1])], labels[data.tolist().index([2, 2])]]), axis=0), np.sort(self.knn_2.find_knn(2, np.array([2, 1])), axis=0))

    def test_knn_1_nn_leafsize_2(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([9, 5])]]), self.knn_2.find_knn(1, np.array([9, 4])))
    
    def test_knn_n_nn_leafsize_2(self):
        np.testing.assert_array_equal(np.array(labels), np.sort(self.knn_2.find_knn(data.shape[0], np.array([1, 1])), axis=0))

    def test_knn_1_nn_leafsize_n(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([9, 5])]]), self.knn_n.find_knn(1, np.array([9, 4])))

    def test_knn_n_nn_leafsize_n(self):
        np.testing.assert_array_equal(np.array(labels), np.sort(self.knn_n.find_knn(data.shape[0], np.array([1, 1])), axis=0))

    def test_knn_1_nn_leafsize_n_1(self):
        np.testing.assert_array_equal(np.array([labels[data.tolist().index([9, 5])]]), self.knn_n_1.find_knn(1, np.array([9, 4])))

    def test_knn_n_nn_leafsize_n_1(self):
        np.testing.assert_array_equal(np.array(labels), np.sort(self.knn_n_1.find_knn(data.shape[0], np.array([1, 1])), axis=0))

if __name__ == "__main__":
    unittest.main()