import numbers
import numpy as np


""" Principal Component Analysis (PCA)

:param data: NumPy dataset to perform PCA on (n_samples x n_features)
:param n_components: Number of Principal Components to be selected
"""
class PCA:

    def __init__(self, data, n_components):
        self._data = data
        self._n_components = n_components

        if not isinstance(self._data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if self._data.size == 0:
            raise ValueError("Data cannot be empty.")

        if not all(isinstance(i, numbers.Number) for i in np.array(self._data).flatten()):
            raise ValueError("Data contains non-numeric values.")

        if not isinstance(self._n_components, int):
            raise ValueError("Number of components must be an integer.")

        if len(self._data.shape) < 2:
            raise ValueError("Data contains ambiguous dimensionality.")
        elif self._n_components < 1 or self._n_components > self._data.shape[1]:
            raise ValueError("Invalid number of components, must be between 1 and n_features.")

        self._s_matrix = np.cov(np.transpose(self._data))
        # Test that the scatter matrix is computer properly
        np.testing.assert_allclose(self._s_matrix, np.dot(np.transpose(self._data), self._data) / (self._data.shape[0]-1))
        # Compute eigenvalues and eigenvectors of covariance matrix
        self._eigval, self._eigvec, = np.linalg.eig(self._s_matrix)
        # Test the equality constraint of eigenvalues and aigenvectors:
        # _s_matrix @ _eigvec = _eigval @ _eigvec
        for i in range(self._data.shape[1]):
            np.testing.assert_allclose(np.dot(np.cov(np.transpose(self._data)), _eigvec[i]), np.dot(_eigval[i], _eigvec[i]))
        self._eigval, self._eigvec = _order_eig(self._eigval, self._eigvec)
        # Test whether eigenvalues have been sorted properly
        np.testing.assert_equal(_eigval, sorted(_eigval, reverse=True))
        self._components = self._eigvec[:self._n_components]
        # Verify the shape of the components
        assert self._components.shape == (self._n_components, self._data.shape[1])
        self._explained_variance = np.sum(self._eigval[:self._n_components]) / np.sum(self._eigval) * 100
        # Test that the explained variance is a statistic
        assert self._explained_variance >= 0 and self._explained_variance <= 100

    """ Order Eigenvalues/Eigenvectors
    Given a vector of eigenvectors and corresponding eigenvalues, it orders the 
    eigenvectors with respect to their eigenvalues.

    :param eigval: List of eigenvalues
    :param eigvec: List of eigenvectors 
    :return eigval: Sorted list of eigenvalues
    :return eigvec: Sorted list of eigenvectors
    """
    def _order_eig(eigval, eigvec):
        eigpairs = [(eigval[i], eigvec[:,i]) for i in range(len(eigval))]
        eigpairs.sort(reverse=True)
        eigval = np.array([elem.real for elem,_ in eigpairs])
        eigvec = np.array([elem.real for _,elem in eigpairs])
        return eigval, eigvec

    @property
    def scatter_matrix(self):
        return self._s_matrix

    @property
    def eigvectors(self):
        return self._eigvec

    @property
    def eigvalalues(self):
        return self._eigval

    @property
    def components(self):
        return self._components

    @property
    def explained_variance(self):
        return self._explained_variance


""" Independent Component Analysis (ICA)

:param data: NumPy dataset to perform ICA on (n_samples x n_features)
:param n_components: Number of Independent Components to be selected
"""
class ICA:

    def __init__(self, data, n_components):
        self._data = data
        self._n_components = n_components

        if not isinstance(self._data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if self._data.size == 0:
            raise ValueError("Data is an empty array.")

        if self._n_components < 1 or self._n_components > self._data.shape[1]:
            raise ValueError("Invalid number of components, must be between 1 and n_features.")

        self._whitened_data, self._whitening_matrix = _whiten(self._data)
        self._whitened_data = np.transpose(self._whitened_data)
        self._raw_components = []
        for i in range(self._n_components):
            component = _compute_unit(self._whitened_data, self._raw_components)
            self.__raw_components.append(component)
        self._raw_components = np.vstack(self._raw_components)
        self._components = np.dot(self._raw_components, self._whitening_matrix)


    """ Data Whitening
    Decorrelates the components of the data

    :param: Data to whiten
    :return X_w: Whitened data
    :return W: Whitening matrix
    """
    def _whiten(data):
        data = _center(data)
        eigval, eigvec, = np.linalg.eig(np.cov(np.transpose(data)))
        W = np.dot(eigvec, np.dot(np.diag(1 / np.sqrt(eigval+1e-9)), np.transpose(eigvec)))
        data_w = np.dot(data, np.transpose(W))
        return data_w, W

    """ Data Centering
    Center features by removing the mean

    :param: Data to center
    :return: Centered data
    """
    def _center(data):
        return data - np.mean(data, axis=0)

    """ Kurtosis Function
    kurt(x) = 4 * u^3
    """
    def _g(u):
        return 4 * u**3

    """ Derivated Kurtosis Function
    kurt'(x) = 12 * u^2
    """
    def _dg(u):
        return 12 * u**2

    """ Compute one Independent Component in ICA

    :param X: Whitened data
    :param W: Existing independent components
    :return: New indendent component
    """
    def _compute_unit(X, W):
        w = np.random.rand(X.shape[0])
        w /= np.linalg.norm(w)
        for iter in range(5000):
            w0 = w
            w = (1 / X.shape[1]-1) * np.dot(X, _g(np.dot(np.transpose(w), X))) - (1 / X.shape[1]-1) * np.dot(_dg(np.dot(np.transpose(w), X)), np.ones((X.shape[1], 1))) * w
            for w_ in W:
                w = w - np.dot(np.transpose(w), w_) * w_
            w /= np.linalg.norm(w) 
            # Check for convergence
            if (1 - np.abs(np.dot(np.transpose(w0), w)) < 1e-10):
                break
        return w

    @property
    def whitening_matrix(self):
        return self._whitening_matrix

    @property
    def whitened_data(self):
        return self._whitened_data

    @property
    def raw_components(self):
        return self._raw_components

    @property
    def components(self):
        return self._components


""" Linear Discriminant Analysis (LDA)

:param data: NumPy dataset to perform LDA on (n_samples x n_features)
:param labels: List of labels (classes) associated with each data point
:param n_dimensions: Number of dimensions to be selected
"""
class LDA:

    def __init__(self, data, labels, n_dimensions):
        self._data = data
        self._labels = labels
        self._n_dimensions = n_dimensions

        if not isinstance(self._data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if self._data.size == 0:
            raise ValueError("Data is an empty array.")

        if self._n_dimensions < 1 or self._n_dimensions > self._data.shape[1]:
            raise ValueError("Invalid number of dimensions, must be between 1 and n_features.")

        _c_means, _c_sizes = _class_means(self._data, self._labels)
        _t_mean = np.array(list(_c_means.values())).mean(axis=0)
        _S_b = _between_class(_c_means, _c_sizes, _t_mean)
        _S_w = _within_class(self._data, self._labels, _c_means)
        self._W = np.dot(np.linalg.inv(_S_w), _S_b)

        self._eigval, self._eigvec, = np.linalg.eig(np.cov(self._W))
        self._eigval, self._eigvec = _order_eig(self._eigval, self._eigvec)
        self._components = self._eigvec[:self._n_dimensions]

    """ Compute the mean for each class in the data

    :param data: Input data (n_samples x n_features)
    :param labels: Input data labels
    :return means: Dictionary of class:mean entries
    :return c_sizes: List of class sizes
    """
    def _class_means(data, labels):
        c_means = dict((class_, np.zeros(data.shape[1])) for class_ in set(labels))
        c_sizes = dict((class_, 0) for class_ in set(labels))
        for index in range(len(labels)):
            c_sizes[labels[index]] += 1
            c_means[labels[index]] += data[index]
        means = {i: c_means[i]/c_sizes[i] for i in c_means}
        return means, c_sizes

    """ Between-Class Matrix

    :param c_means: List of class means
    :param c_sizes: List of class sizes
    :param t_mean: Total mean
    :return: Between-class matrix
    """
    def _between_class(c_means, c_sizes, t_mean):
        S_b = np.zeros((len(t_mean), len(t_mean)))
        for class_ in c_means.keys():
            S_b += c_sizes[class_] * np.outer((c_means[class_] - t_mean), (c_means[class_] - t_mean))
        return S_b

    """ Within-Class Matrix

    :param data: Input data (n_samples x n_features)
    :param labels: Input data labels
    :param c_means: List of class means
    :return: Within-class matrix
    """
    def _within_class(data, labels, c_means):
        data = np.array([data[i] - c_means[labels[i]] for i in range(data.shape[0])])
        S_w = np.zeros((data.shape[1], data.shape[1]))
        for class_ in c_means.keys():
            S_w += np.dot(np.transpose(data[[i for i in range(len(labels)) if labels[i] == class_]]), data[[i for i in range(len(labels)) if labels[i] == class_]])
        return S_w

    """ Order Eigenvalues/Eigenvectors
    Given a vector of eigenvectors and corresponding eigenvalues, it orders the 
    eigenvectors with respect to their eigenvalues.

    :param eigval: List of eigenvalues
    :param eigvec: List of eigenvectors 
    :return eigval: Sorted list of eigenvalues
    :return eigvec: Sorted list of eigenvectors
    """
    def _order_eig(eigval, eigvec):
        eigpairs = [(eigval[i], eigvec[:,i]) for i in range(len(eigval))]
        print(eigpairs)
        eigpairs.sort(reverse=True)
        eigval = np.array([elem.real for elem,_ in eigpairs])
        eigvec = np.array([elem.real for _,elem in eigpairs])
        return eigval, eigvec

    @property
    def transformation_matrix(self):
        return self._W

    @property
    def eigvectors(self):
        return self._eigvec

    @property
    def eigvalues(self):
        return self._eigval

    @property
    def components(self):
        return self._components