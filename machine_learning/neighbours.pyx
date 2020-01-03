import collections
cimport cython

import numpy as np
cimport numpy as cnp

from machine_learning.data_structures import priority_queue


""" k-Nearest Neighbours (kNN)
:param data: Numpy training data (n_samples x n_features)
:param labels: Training data labels (n_features)
:return distance: Distance metric function (e.g. Euclidean)
:return leaf_size: Size of the ball tree leaves
"""
cdef class kNN:

    cdef BallTree _ball_tree
    cdef cnp.float64_t[:, :] _data
    cdef int _leaf_size
    cdef _distance, _labels, _pq

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=2] data, cnp.ndarray labels, distance, int leaf_size): # not None

        if len(data) == 0:
            raise ValueError("Data cannot be empty.")

        if len(labels) == 0:
            raise ValueError("Labels cannot be empty.")
        if len(data) != len(labels):
            raise ValueError("Labels and data must have the same length")

        if not isinstance(distance, collections.Callable):
            raise ValueError("Distance metric must be a callable method.")

        if leaf_size < 1:
            raise ValueError("Leaf size must be a positive integer.")

        self._data = data
        self._labels = labels
        self._distance = distance
        self._leaf_size = leaf_size
        self._ball_tree = BallTree(np.asarray(self._data), self._distance, self._leaf_size)

    @property 
    def data(self):
        return self._data

    @property 
    def labels(self):
        return self._labels

    @property 
    def distance(self):
        return self._distance

    @property 
    def leaf_size(self):
        return self._leaf_size

    @property 
    def ball_tree(self):
        return self._ball_tree

    @data.setter
    def data(self, cnp.ndarray[cnp.float64_t, ndim=2] data):
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if len(data) != len(self._labels):
            raise ValueError("Labels and data must have the same length")
        self._data = data
        self._ball_tree = BallTree(np.asarray(self._data), self._distance, self._leaf_size)

    @labels.setter
    def labels(self, cnp.ndarray labels):
        if len(labels) == 0:
            raise ValueError("Labels cannot be empty.")
        if len(self._data) != len(labels):
            raise ValueError("Labels and data must have the same length")
        self._labels = labels

    @distance.setter
    def distance(self, distance):
        if not isinstance(distance, collections.Callable):
            raise ValueError("Distance metric must be a callable method.")
        self._distance = distance
        self._ball_tree = BallTree(np.asarray(self._data), self._distance, self._leaf_size)

    @leaf_size.setter
    def leaf_size(self, int leaf_size):
        if leaf_size < 1:
            raise ValueError("Leaf size must be a positive integer.")
        self._leaf_size = leaf_size
        self._ball_tree = BallTree(np.asarray(self._data), self._distance, self._leaf_size)

    """ Search the ball tree for the k nearest neighbours
    :param self: Instance to which the method belongs
    :param k: Number of nearest neighbours to return
    :return neighbour_labels: The labels of the k neighbours of the given point
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def find_knn(self, int k, cnp.ndarray[cnp.float64_t, ndim=1] point):
        if k <= 0:
            raise ValueError("Number of neighbours must be a positive integer.")
        if k > self._data.shape[0]:
            raise ValueError("Number of neighbours must be less than the number of data points.")

        if point.shape[0] != self._data.shape[1]:
            raise ValueError("Dimension of query point must match dimension of the data.")

        self._pq = priority_queue(np.transpose(np.vstack((np.arange(k), np.inf))))
        self._search_kd_subtree(self._ball_tree.root, k, point)

        neighbour_labels = np.array([self._labels[self._pq.pop()[0]] for _ in range(k)])
        return neighbour_labels

    """ Search one ball subtree for the k nearest neighbours
    :param self: Instance to which the method belongs
    :param node: The node to be traversed
    :param k: Number of nearest neighbours to find
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _search_kd_subtree(self, BallNode node, int k, cnp.ndarray[cnp.float64_t, ndim=1] point):
        # Check whether the largest distance in the current priority queue is smaller than the distance to the node's ball
        if self._distance(point, node.pivot) - node.radius >= self._pq.first()['value']:
            pass
        elif node.is_leaf():
            for i in node.points:
                point_distance = self._distance(point, self._data[i])
                if point_distance < self._pq.first()['value']:
                    self._pq.push([i, point_distance])
                    if self._pq.size() > k:
                        self._pq.pop()
        else:
            if node.closest_child:
                self._search_kd_subtree(node.closest_child, k, point)
            if node.farthest_child:
                self._search_kd_subtree(node.farthest_child, k, point)


""" Ball Tree
:param data: Numpy training data (n_samples x n_features)
:return distance: Distance metric function (e.g. Euclidean)
:return leaf_size: Size of the ball tree leaves
"""
cdef class BallTree:

    cdef BallNode _root
    cdef _data, _distance, _leaf_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=2] data, distance, int leaf_size):

        if len(data) == 0:
            raise ValueError("Data cannot be empty.")

        if not isinstance(distance, collections.Callable):
            raise ValueError("Distance metric must be a callable method.")

        if leaf_size < 1:
            raise ValueError("Leaf size must be a positive integer.")

        self._data = data
        self._distance = distance
        self._leaf_size = leaf_size
        self._root = BallNode(data, np.arange(len(self._data)), distance, leaf_size)

    @property
    def data(self):
        return self._data

    @property
    def distance(self):
        return self._distance

    @property
    def leaf_size(self):
        return self._leaf_size

    @property
    def root(self):
        return self._root

    @data.setter
    def data(self, cnp.ndarray[cnp.float64_t, ndim=2] data):
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        self._data = data

    @distance.setter
    def distance(self, distance):
        if not isinstance(distance, collections.Callable):
            raise ValueError("Distance metric must be a callable method.")
        self._distance = distance

    @leaf_size.setter
    def leaf_size(self, int leaf_size):
        if leaf_size < 1:
            raise ValueError("Leaf size must be a positive integer.")
        self._leaf_size = leaf_size


""" Ball Tree Node
:param data: Numpy training data (n_samples x n_features)
:param distance: Distance metric function (e.g. Euclidean)
:param leaf_size: Size of the ball tree leaves
"""
cdef class BallNode:

    cdef _points, _pivot
    cdef float _radius
    cdef BallNode _farthest_child, _closest_child

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, cnp.ndarray[cnp.float64_t, ndim=2] data, cnp.ndarray[cnp.long_t, ndim=1] points, distance, int leaf_size):

        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
            
        if not isinstance(distance, collections.Callable):
            raise ValueError("Distance metric must be a callable method.")

        if leaf_size < 1:
            raise ValueError("Leaf size must be a positive integer.")

        self._points = points
        self._farthest_child = None
        self._closest_child = None
        self._pivot = np.mean([data[i] for i in points], axis=0)
        self._radius = 0

        cdef list farthest_child_points, closest_child_points 
        cdef cnp.ndarray[cnp.float64_t, ndim=1] farthest, second_farthest

        p_queue = priority_queue([])

        for i in points:
            p_queue.push([i, distance(self._pivot, data[i])])

        i_farthest, self._radius = p_queue.pop()
        farthest = data[i_farthest]
        
        if p_queue.size() > 0:
            second_farthest = data[p_queue.pop()[0]]
            farthest_child_points = []
            closest_child_points = []
            for i in points:
                if distance(data[i], farthest) <= distance(data[i], second_farthest):
                    farthest_child_points.append(i)
                else:
                    closest_child_points.append(i)

        if len(points) > leaf_size:
            self._farthest_child = BallNode(data, np.array(farthest_child_points), distance, leaf_size)
            self._closest_child = BallNode(data, np.array(closest_child_points), distance, leaf_size)

    @property
    def pivot(self):
        return self._pivot

    @property
    def radius(self):
        return self._radius

    @property
    def points(self):
        return self._points

    @property
    def farthest_child(self):
        return self._farthest_child

    @property
    def closest_child(self):
        return self._closest_child

    def is_leaf(self):
        return not self._farthest_child and not self._closest_child