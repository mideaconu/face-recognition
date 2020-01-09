#
# Author: Mihai-Ionut Deaconu
#

import sys

import unittest
import numpy as np
from scipy.spatial.distance import cityblock

from machine_learning.neighbours import kNN

data = np.ndarray([[1, 1], [2, 3], [4, 2]])
labels = np.ndarray(['a', 'a', 'b'])

class kNNTest(unittest.TestCase):

    def setUp(self):
        self.knn = kNN(data, labels, cityblock, 1)

    """ Constructor input parameter test """

    def test_data(self):
        with self.assertRaises(ValueError):
            knn = kNN(np.array([]), labels, cityblock, 1)

    def test_labels_empty(self):
        with self.assertRaises(ValueError):
            knn = kNN(data, np.array([]), cityblock, 1)

    def test_labels_length(self):
        with self.assertRaises(ValueError):
            knn = kNN(data, np.array(['a', 'a']), cityblock, 1)

    def test_distance(self):
        with self.assertRaises(ValueError):
            knn = kNN(data, labels, "cityblock", 1)

    def test_leafsize(self):
        with self.assertRaises(ValueError):
            knn = kNN(data, labels, cityblock, 0)

    """ Find kNN parameter test """

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(0, np.ndarray([2, 2]))

    def test_invalid_k_(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(data.shape[1]+1, np.ndarray([2, 2]))

    def test_point_shape(self):
        with self.assertRaises(ValueError):
            self.knn.find_knn(1, np.ndarray([2, 2, 2]))

    """ Setter input parameter test """

    def test_data_setter(self):
        with self.assertRaises(ValueError):
            self.knn.data = np.array([])

    def test_labels_setter_empty(self):
        with self.assertRaises(ValueError):
            self.knn.labels = np.array([])

    def test_labels_setter_length(self):
        with self.assertRaises(ValueError):
            self.knn.labels = np.array(['a', 'a'])

    def test_distance_setter(self):
        with self.assertRaises(ValueError):
            self.knn.distance = "cityblock"

    def test_leafsize_setter(self):
        with self.assertRaises(ValueError):
            self.knn.leafsize = 0

if __name__ == "__main__":
    unittest.main() 