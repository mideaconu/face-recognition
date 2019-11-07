import heapq
import collections
import numpy as np
cimport numpy as cnp
cimport cython


""" k-Nearest Neighbours (kNN)

:param data: Numpy training data (n_samples x n_features)
:param labels: Training data labels (n_features)
:return distance: Distance metric function (e.g. Euclidean)
:return leaf_size: Size of the ball tree leaves
"""
cdef class kNN:
    cdef BallTree _ball_tree
    cdef _data, _labels, _distance, _pq

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=2] data, labels, distance, int leaf_size):

        if type(labels) is not np.ndarray:
            raise ValueError("Labels must be in the form of aNumPy array.")

        if not len(data):
            raise ValueError("Data cannot be empty.")

        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length.")

        if not isinstance(distance, collections.Callable):
            raise ValueError("Distance metric must be a callable method.")

        if leaf_size < 1:
            raise ValueError("Leaf size must be a positive integer.")

        self._data = data
        self._labels = labels
        self._distance = distance
        self._ball_tree = BallTree(self._data, self._distance, leaf_size)

    @property 
    def ball_tree(self):
        return self._ball_tree

    """ Search the ball tree for the k nearest neighbours

    :param self: Instance to which the method belongs
    :param k: Number of nearest neighbours to return
    :return neighbour_labels: The labels of the k neighbours of the given point
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def find_knn(self, int k, cnp.ndarray[cnp.float64_t, ndim=1] point):
        if k <= 0:
            raise ValueError("Number of neighbours must be a positive integer.")

        if k > self._data.shape[0]:
            raise ValueError("Number of neighbours must be less than the number of data points.")

        if point.shape[0] != self._data.shape[1]:
            raise ValueError("Dimension of query point must match dimension of the data.")

        # Initialize the priority queue with k infinity falues (distances) and origin points (neighbours)
        self._pq = [(np.inf, np.zeros(self._data.shape[1]))] * k
        heapq.heapify(self._pq)
        self._search_kd_subtree(self._ball_tree.root, k, point)
        neighbour_labels = np.array([self._labels[self._data.tolist().index(i[1])] for i in self._pq])
        return neighbour_labels

    """ Search one ball subtree for the k nearest neighbours

    :param self: Instance to which the method belongs
    :param node: The node to be traversed
    :param k: Number of nearest neighbours to find
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef void _search_kd_subtree(self, BallNode node, int k, cnp.ndarray[cnp.float64_t, ndim=1] point):
        # Check whether the largest distance in the current priority queue is smaller than the distance to the node's ball
        if self._distance(point, node.pivot) - node.radius >= heapq.nlargest(1, self._pq)[0][0]:
            pass
        elif node.is_leaf():
            for node_point in node.points:
                if self._distance(point, node_point) < heapq.nlargest(1, self._pq)[0][0]:
                    heapq.heappush(self._pq, (self._distance(point, node_point), node_point.tolist()))
                    heapq._heapify_max(self._pq)
                    if len(self._pq) > k:
                        heapq._heappop_max(self._pq)
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=2] data, distance, int leaf_size):
        self._root = BallNode(data, distance, leaf_size)

    @property
    def root(self):
        return self._root


""" Ball Tree Node

:param data: Numpy training data (n_samples x n_features)
:return distance: Distance metric function (e.g. Euclidean)
:return leaf_size: Size of the ball tree leaves
"""
cdef class BallNode:
    cdef _data, _pivot
    cdef float _radius
    cdef BallNode _farthest_child, _closest_child

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=2] data, distance, int leaf_size):
        self._data = data
        self._farthest_child = None; self._closest_child = None
        self._pivot = np.mean(self._data, axis=0)

        distances_from_pivot = [distance(self._pivot, point) for point in self._data]
        farthest = heapq.nlargest(2, range(len(distances_from_pivot)), key=distances_from_pivot.__getitem__)

        i_farthest = farthest[0]
        self._radius = distance(self._pivot, self._data[i_farthest])
        if len(farthest) > 1:
            i_2nd_farthest = farthest[1]
            farthest_child_points = []; closest_child_points = []
            for point in self._data:
                if distance(point, self._data[i_farthest]) <= distance(point, self._data[i_2nd_farthest]):
                    farthest_child_points.append(point)
                else:
                    closest_child_points.append(point)

        if len(self._data) > leaf_size:
            self._farthest_child = BallNode(np.array(farthest_child_points), distance, leaf_size)
            self._closest_child = BallNode(np.array(closest_child_points), distance, leaf_size)

    @property
    def pivot(self):
        return self._pivot

    @property
    def radius(self):
        return self._radius

    @property
    def points(self):
        return self._data

    @property
    def farthest_child(self):
        return self._farthest_child

    @property
    def closest_child(self):
        return self._closest_child

    def is_leaf(self):
        return not self._farthest_child and not self._closest_child