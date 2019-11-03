import numpy as np
import numbers
import collections
import heapq

""" k-Nearest Neighbours (kNN)

:param data: Training data
:param labels: Labels of the training data
:return dist: Distance metric function
:return leaf_size: Size of the ball tree leaves
"""
class kNN:

    def __init__(self, data, labels, dist, leaf_size):
        self.__data = data
        self.__labels = labels
        self.__dist = dist
        self.__leaf_size = leaf_size

        if not isinstance(self.__data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if len(self.__data) == 0:
            raise ValueError("Data cannot be empty.")

        if not all(isinstance(i, numbers.Number) for i in np.array(self.__data).flatten()):
            raise ValueError("Data contains non-numeric values.")

        if not isinstance(self.__labels, (list, np.ndarray)):
            raise ValueError("Labels must be a list or NumPy array.")

        if len(self.__data) != len(self.__labels):
            raise ValueError("Data and labels must have the same length.")

        if not isinstance(self.__dist, collections.Callable):
            raise ValueError("Distance metric must be a callable method.")

        if self.__leaf_size < 1:
            raise ValueError("Leaf size must be a positive integer.")

    def construct_tree(self):
        self.__ball_tree = BallTree(self.__data, self.__dist, self.__leaf_size)

    """ Search the ball tree for the k nearest neighbours

    :param self: Instance to which the method belongs
    :param k: Number of nearest neighbours to return
    :return point: The point whose neighbours are requested
    """
    def find_knn(self, k, point):
        if k < 0:
            raise ValueError("Number of neighbours must be non-negative.")
        if k == 0:
            return np.array([])
        if k > self.__data.shape[0]:
            raise ValueError("Number of neighbours must be less than the number of data points.")
        if point.shape[0] != self.__data.shape[1]:
            raise ValueError("Dimension of query point must match dimension of data.")

        self.__pq = [(np.inf, np.zeros(self.__data.shape[1]))] * k
        heapq.heapify(self.__pq)
        self.__search_kd_subtree(self.__ball_tree.get_root(), k, point)
        return np.array([self.__labels[self.__data.tolist().index(i[1])] for i in self.__pq])

    """ Search one ball subtree for the k nearest neighbours

    :param self: Instance to which the method belongs
    :param node: The node to be traversed
    :param k: Number of nearest neighbours to return
    :return point: The point whose neighbours are requested
    """
    def __search_kd_subtree(self, node, k, point):
        if self.__dist(point, node.get_pivot()) >= heapq.nlargest(1, self.__pq)[0][0]:
            return 
        elif node.is_leaf():
            for node_point in node.get_points():
                if self.__dist(point, node_point) < heapq.nlargest(1, self.__pq)[0][0]:
                    heapq.heappush(self.__pq, (self.__dist(point, node_point), node_point.tolist()))
                    heapq._heapify_max(self.__pq)
                    if len(self.__pq) > k:
                        heapq._heappop_max(self.__pq)
        else:
            if node.get_closest_child():
                self.__search_kd_subtree(node.get_closest_child(), k, point)
            if node.get_furthest_child():
                self.__search_kd_subtree(node.get_furthest_child(), k, point)

""" Ball Tree

:param data: Training data
:return dist: Distance metric function
:return leaf_size: Size of the ball tree leaves
"""
class BallTree:
    def __init__(self, data, dist, leaf_size):
        self.__data = data
        self.__dist = dist
        self.__leaf_size = leaf_size
        self.__root = BallNode(self.__data, self.__dist, self.__leaf_size)

    def get_root(self):
        return self.__root

""" Ball Tree Node

:param data: Training data
:return dist: Distance metric function
:return leaf_size: Size of the ball tree leaves
"""
class BallNode:
    def __init__(self, data, dist, leaf_size):
        self.__data = data
        self.__dist = dist
        self.__leaf_size = leaf_size

        self.__child1 = None; self.__child2 = None

        self.__pivot = np.mean(self.__data, axis=0)
        distances_from_pivot = [self.__dist(self.__pivot, point) for point in self.__data]
        farthest = heapq.nlargest(2, range(len(distances_from_pivot)), key=distances_from_pivot.__getitem__)
        
        i_farthest = farthest[0]
        self.__radius = self.__dist(self.__pivot, self.__data[i_farthest])
        if len(farthest) > 1:
            i_2nd_farthest = farthest[1]
            child1_points = []; child2_points = []
            for point in self.__data:
                if self.__dist(point, self.__data[i_farthest]) <= self.__dist(point, self.__data[i_2nd_farthest]):
                    child1_points.append(point)
                else:
                    child2_points.append(point)

        if len(self.__data) > self.__leaf_size:
            self.__child1 = BallNode(child1_points, self.__dist, self.__leaf_size)
            self.__child2 = BallNode(child2_points, self.__dist, self.__leaf_size)

    def get_pivot(self):
        return self.__pivot

    def get_points(self):
        return self.__data

    def get_closest_child(self):
        return self.__child1

    def get_furthest_child(self):
        return self.__child2

    def is_leaf(self):
        return not self.__child1 and not self.__child2