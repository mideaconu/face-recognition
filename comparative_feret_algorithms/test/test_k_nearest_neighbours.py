import sys

import unittest
import collections
import numpy as np

sys.path.append("..")
from k_nearest_neighbours import kNN

data = np.array([[1, 1], [1, 2], [2, 2], [2, 8], [3, 5], [3, 6], [3, 10], 
                [4, 6], [6, 3], [6, 6], [6, 9], [7, 1], [8, 1], [9, 5], 
                [9, 9], [9, 10]])

labels = [chr(i) for i in range(ord('a'), ord('p')+1)]

def euclidean(a, b):
    return np.linalg.norm(a - b)


class kNNTest(unittest.TestCase):

    def setUp(self):
        self.knn = kNN(data, labels, euclidean, 1)
        self.knn.construct_tree()

        self.knn_2 = kNN(data, labels, euclidean, 2)
        self.knn_2.construct_tree()

        self.knn_n = kNN(data, labels, euclidean, data.shape[0])
        self.knn_n.construct_tree()

        self.knn_n_1 = kNN(data, labels, euclidean, data.shape[0]+1)
        self.knn_n_1.construct_tree()
    
    def test_null_data(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(None, ['a', 'b', 'c'], euclidean, 1)

    def test_int_type_data(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(0, ['a', 'b', 'c'], euclidean, 1)

    def test_float_type_data(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(0., ['a', 'b', 'c'], euclidean, 1)

    def test_str_type_data(self):
        with self.assertRaises(ValueError):
            knn_ = kNN("0", ['a', 'b', 'c'], euclidean, 1)

    def test_list_data(self):
        with self.assertRaises(ValueError):
            knn_ = kNN([0, 1, 2], ['a', 'b', 'c'], euclidean, 1)

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([]), ['a', 'b', 'c'], euclidean, 1)

    def test_nonnumeric_data(self):
        with self.assertRaises(ValueError):
            knn_ = kNN(np.array([0, 1, 'c']), ['a', 'b', 'c'], euclidean, 1)

    def test_nd_data(self):
        knn_ = kNN(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]), ['a', 'b', 'c'], euclidean, 1)
    
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