import numpy as np


""" Principal Component Analysis (PCA)

:param data: NumPy dataset to perform PCA on (n_samples x n_features)
:param n_components: Number of principal components to be selected
"""
class PCA:

    def __init__(self, data, n_components):

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if len(data) == 0:
            raise ValueError("Data cannot be empty.")

        if not isinstance(n_components, int):
            raise ValueError("Number of components must be an integer.")

        if n_components < 1 or n_components > data.shape[1]:
            raise ValueError("Invalid number of components, must be between 1 and n_features.")

        _centered_data = data - np.mean(data, axis=0)
        # Compute eigenvalues and eigenvectors of covariance matrix
        _eigval, _eigvec, = np.linalg.eig(np.cov(np.transpose(_centered_data)))
        # Test the constraint of eigenvalues and eigenvectors:
        # _s_matrix @ _eigvec = _eigval @ _eigvec
        np.testing.assert_allclose(np.dot(np.cov(np.transpose(_centered_data)), _eigvec), np.dot(_eigvec, np.diag(_eigval)), atol=1e-10)

        # Order eigenvalues and eigenvectors
        _idx = np.argsort(_eigval)[::-1]
        _eigval = np.array(_eigval[_idx].real)
        _eigvec = np.array([_eigvec[:,i].real for i in _idx])

        self._components = _eigvec[:n_components]
        self._explained_variance = np.sum(_eigval[:n_components]) / np.sum(_eigval) * 100

    @property
    def components(self):
        return self._components

    @property
    def explained_variance(self):
        return self._explained_variance


""" Independent Component Analysis (ICA)

:param data: NumPy dataset to perform ICA on (n_samples x n_features)
:param n_components: Number of independent components to be selected
"""
class ICA:

    def __init__(self, data, n_components):

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if len(data) == 0:
            raise ValueError("Data cannot be empty.")

        if not isinstance(n_components, int):
            raise ValueError("Number of components must be an integer.")

        if n_components < 1 or n_components > data.shape[1]:
            raise ValueError("Invalid number of components, must be between 1 and n_features.")

        _centered_data = self._center(data)
        _whitened_data, _whitening_matrix = self._whiten(_centered_data)
        # Test the constraint of the whitened data:
        # cov(X @ W) = I
        np.testing.assert_allclose(np.cov(np.transpose(_whitened_data)), np.eye(data.shape[1]), atol=1e-10)
        # Test the constraint of the whitening matrix:
        # W^T @ W = cov(X)^-1
        np.testing.assert_allclose(np.linalg.inv(np.cov(np.transpose(data))), np.dot(np.transpose(_whitening_matrix), _whitening_matrix), atol=1e-10)

        _whitened_data = np.transpose(_whitened_data)
        self._components = []
        for i in range(n_components):
            component = self._compute_unit(_whitened_data, self._components)
            self._components.append(component)
        self._components = np.vstack(self._components)
        # Test for independence of the components:
        # S^T @ S = I
        np.testing.assert_allclose(np.dot(np.transpose(self._components), self._components), np.eye(data.shape[1]), atol=1e-10)

        self._components = np.dot(self._components, _whitening_matrix)


    """ Data whitening
    Decorrelates the components of the data

    :param: Data to whiten
    :return X_w: Whitened data
    :return W: Whitening matrix
    """
    def _whiten(self, data):
        eigval, eigvec, = np.linalg.eigh(np.cov(np.transpose(data)))
        W = np.dot(eigvec, np.dot(np.diag(1 / np.sqrt(eigval+1e-9)), np.transpose(eigvec)))
        data_w = np.dot(data, np.transpose(W))
        return data_w, W

    """ Data centering
    Center features by removing the mean

    :param: Data to center
    :return: Centered data
    """
    def _center(self, data):
        return data - np.mean(data, axis=0)

    """ Kurtosis function
    kurt(x) = 4 * u^3
    """
    def _g(self, u):
        return 4 * u**3

    """ Derivated kurtosis function
    kurt'(x) = 12 * u^2
    """
    def _dg(self, u):
        return 12 * u**2

    """ Compute one independent component

    :param X: Whitened data
    :param W: Existing independent components
    :return: New indendent component
    """
    def _compute_unit(self, X, W):
        w = np.random.rand(X.shape[0])
        w /= np.linalg.norm(w)
        for iter in range(5000):
            w0 = w
            w = (1 / X.shape[1]-1) * np.dot(X, self._g(np.dot(np.transpose(w), X))) - (1 / X.shape[1]-1) * np.dot(self._dg(np.dot(np.transpose(w), X)), np.ones((X.shape[1], 1))) * w
            for w_ in W:
                w = w - np.dot(np.transpose(w), w_) * w_
            w /= np.linalg.norm(w) 
            # Check for convergence
            if 1 - np.abs(np.dot(np.transpose(w0), w)) < 1e-10:
                break
        return w

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

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if len(data) == 0:
            raise ValueError("Data cannot be empty.")

        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length.")

        if not isinstance(n_dimensions, int):
            raise ValueError("Number of components must be an integer.")

        _c_means, _c_sizes = self._class_means(data, labels)
        _t_mean = np.array(list(_c_means.values())).mean(axis=0)
        _S_b = self._between_class(_c_means, _c_sizes, _t_mean)
        _S_w = self._within_class(data, labels, _c_means)
        self._W = np.dot(np.linalg.inv(_S_w), _S_b)

        _eigval, _eigvec, = np.linalg.eig(np.cov(self._W))
        # Test the constraint of eigenvalues and eigenvectors:
        # _s_matrix @ _eigvec = _eigval @ _eigvec
        np.testing.assert_allclose(np.dot(np.cov(self._W), _eigvec), np.dot(_eigvec, np.diag(_eigval)), atol=1e-10)

        # Order eigenvalues and eigenvectors
        _idx = np.argsort(_eigval)[::-1]
        _eigval = np.array(_eigval[_idx].real)
        _eigvec = np.array([_eigvec[:,i].real for i in _idx])

        self._components = _eigvec[:n_dimensions]

    """ Compute the mean for each class in the data

    :param data: Input data (n_samples x n_features)
    :param labels: Input data labels
    :return means: Dictionary of class:mean entries
    :return c_sizes: List of class sizes
    """
    def _class_means(self, data, labels):
        c_means = dict((class_, np.zeros(data.shape[1])) for class_ in set(labels))
        c_sizes = dict((class_, 0) for class_ in set(labels))
        for index in range(len(labels)):
            c_sizes[labels[index]] += 1
            c_means[labels[index]] += data[index]
        means = {i: c_means[i]/c_sizes[i] for i in c_means}
        return means, c_sizes

    """ Between-class Matrix

    :param c_means: List of class means
    :param c_sizes: List of class sizes
    :param t_mean: Total mean
    :return: Between-class matrix
    """
    def _between_class(self, c_means, c_sizes, t_mean):
        S_b = np.zeros((len(t_mean), len(t_mean)))
        for class_ in c_means.keys():
            S_b += c_sizes[class_] * np.outer((c_means[class_] - t_mean), (c_means[class_] - t_mean))
        return S_b

    """ Within-class Matrix

    :param data: Input data (n_samples x n_features)
    :param labels: Input data labels
    :param c_means: List of class means
    :return: Within-class matrix
    """
    def _within_class(self, data, labels, c_means):
        data = np.array([data[i] - c_means[labels[i]] for i in range(data.shape[0])])
        S_w = np.zeros((data.shape[1], data.shape[1]))
        for class_ in c_means.keys():
            S_w += np.dot(np.transpose(data[[i for i in range(len(labels)) if labels[i] == class_]]), data[[i for i in range(len(labels)) if labels[i] == class_]])
        return S_w

    @property
    def transformation_matrix(self):
        return self._W

    @property
    def components(self):
        return self._components