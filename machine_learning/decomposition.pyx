import numpy as np
import scipy as sp
cimport numpy as cnp
cimport cython


""" Principal Component Analysis (PCA)
:param data: NumPy dataset to perform PCA on (n_samples x n_features)
:param n_components: Number of principal components to be selected
"""
cdef class PCA:

    cdef int _n_components
    cdef str _solver
    cdef int _n_oversamples
    cdef int _n_iter
    cdef _components
    cdef float _explained_variance

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, int n_components, str solver="svd", int n_oversamples=10, int n_iter=4):
        self._n_components = n_components
        self._solver = solver
        self._n_oversamples = n_oversamples
        self._n_iter = n_iter

    """ Fit the model to the data
    Compute the principal components and total explained variance
    :param: Data
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray[cnp.float64_t, ndim=2] data):

        cdef cnp.ndarray[cnp.float64_t, ndim=2] _centered_data = self._center(np.transpose(data))
        cdef cnp.ndarray[cnp.float64_t, ndim=2] _U, _Uh, _S, _Vt, _Omega, _Q, _B
        cdef cnp.ndarray[cnp.float64_t, ndim=1] _s, _variance
        cdef cnp.ndarray[cnp.long_t, ndim=1] _idx
        cdef cnp.ndarray _eigval, _eigvec
        cdef Py_ssize_t _n_samples = _centered_data.shape[0], _n_features = _centered_data.shape[1]
        cdef Py_ssize_t _i
        cdef int _n_dimensions

        if self._solver == "eig":

            # Compute eigenvalues and eigenvectors of covariance matrix
            _eigval, _eigvec, = sp.linalg.eigh(np.cov(_centered_data))
            # sp.linalg.eig might return complex values, so convert them to float
            _eigval, _eigvec = _eigval.real, _eigvec.real
            # Order eigenvalues and eigenvectors
            _idx = np.argsort(_eigval)[::-1]
            _eigval = np.array(_eigval[_idx])
            _eigvec = np.array([_eigvec[:,_i] for _i in _idx])
            self._components = _eigvec[:self._n_components]
            self._explained_variance = np.sum(_eigval[:self._n_components]) / np.sum(_eigval) * 100

        elif self._solver == "svd":

            if max(_n_samples, _n_features) < 500 or self._n_components > .8 * min(_n_samples, _n_features):

                print("full")
                _U, _s, __ = sp.linalg.svd(_centered_data)
                _variance = (_s ** 2) / (_n_samples - 1)
                self._components = _U[:,:self._n_components]
                self._explained_variance = np.sum(_variance[:self._n_components]) / np.sum(_variance) * 100

            else:

                print("randomised")
                _n_dimensions = self._n_components + self._n_oversamples
                # Sample (k + p) i.i.d. vectors from a normal distribution
                _Omega = np.random.normal(size=(_n_features, _n_dimensions))
                # Perform QR decompotision on (A @ At)^q @ A @ Omega
                for __ in range(self._n_iter):
                    _Q, __ = np.linalg.qr(np.dot(_centered_data, _Q))
                    _Q, __ = np.linalg.qr(np.dot(np.transpose(_centered_data), _Q))
                _Q, __ = np.linalg.qr(np.dot(_centered_data, _Omega))

    """ Data centering
    Center features by removing the mean
    :param: Data to center
    :return: Centered data
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _center(self, cnp.ndarray[cnp.float64_t, ndim=2] data):    
        return data - np.mean(data, axis=0)

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
cdef class ICA:

    cdef int _n_components
    cdef _components

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, int n_components):
        self._n_components = n_components

    """ Fit the model to the data
    Compute the individual components
    :param: Data
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray[cnp.float64_t, ndim=2] data):  

        cdef cnp.ndarray[cnp.float64_t, ndim=2] _centered_data, _whitened_data, _whitening_matrix
        cdef cnp.ndarray[cnp.float64_t, ndim=1] _component
        cdef cnp.ndarray _eigval, _eigvec
        cdef Py_ssize_t _n_features = data.shape[1]
        cdef Py_ssize_t _i

        _centered_data = self._center(data)
        _eigval, _eigvec, = sp.linalg.eigh(np.cov(np.transpose(_centered_data)))
        _eigval, _eigvec = _eigval.real, _eigvec.real
        _whitening_matrix = np.dot(_eigvec, np.dot(np.diag(1 / np.sqrt(_eigval+1e-10)), np.transpose(_eigvec)))
        _whitened_data = np.dot(_centered_data, np.transpose(_whitening_matrix))
        # Test the constraint of the whitened data:
        # cov(X @ W) = I
        np.testing.assert_allclose(np.cov(np.transpose(_whitened_data)), np.eye(_n_features), atol=1e-7)
        # Test the constraint of the whitening matrix:
        # W^T @ W = cov(X)^-1
        np.testing.assert_allclose(np.linalg.inv(np.cov(np.transpose(data))), np.dot(np.transpose(_whitening_matrix), _whitening_matrix), atol=1e-7)

        _whitened_data = np.transpose(_whitened_data)
        _components = []
        for _i in range(self._n_components):
            _component = self._compute_unit(_whitened_data, _components)
            _components.append(_component)
        _components = np.vstack(_components)
        # Test for independence of the components:
        # S^T @ S = I
        np.testing.assert_allclose(np.dot(np.transpose(_components), _components), np.eye(_n_features), atol=1e-7)

        self._components = np.dot(_components, _whitening_matrix)

    """ Data centering
    Center features by removing the mean
    :param: Data to center
    :return: Centered data
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _center(self, cnp.ndarray[cnp.float64_t, ndim=2] data):
        return data - np.mean(data, axis=0)

    """ Kurtosis function
    kurt(x) = 4 * u^3
    """
    cdef _g(self, u):
        return 4 * u**3

    """ Derivated kurtosis function
    kurt'(x) = 12 * u^2
    """
    cdef _dg(self, u):
        return 12 * u**2

    """ Compute one independent component
    :param X: Whitened data
    :param W: Existing independent components
    :return: New indendent component
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _compute_unit(self, cnp.ndarray[cnp.float64_t, ndim=2] X, W):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] w = np.random.rand(X.shape[0])
        cdef cnp.ndarray[cnp.float64_t, ndim=1] _w, w0 
        cdef Py_ssize_t _iter

        w /= np.linalg.norm(w)
        for _iter in range(1000):
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
:param n_components: Number of dimensions to be selected
"""
cdef class LDA:

    cdef int _n_components
    cdef _components
    cdef _W

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, int n_components):
        self._n_components = n_components

    """ Fit the model to the data
    Compute the individual components
    :param: Data
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray[cnp.float64_t, ndim=2] data, labels):  

        cdef cnp.ndarray[cnp.float64_t, ndim=2] _S_b, _S_w
        cdef cnp.ndarray[cnp.float64_t, ndim=1] _t_mean
        cdef cnp.ndarray[cnp.long_t, ndim=1] _idx
        cdef cnp.ndarray _eigval, _eigvec
        cdef _c_means, _c_sizes

        _c_means, _c_sizes = self._class_means(data, labels)
        _t_mean = np.array(list(_c_means.values())).mean(axis=0)
        _S_b = self._between_class(_c_means, _c_sizes, _t_mean)
        _S_w = self._within_class(data, labels, _c_means)
        self._W = np.dot(np.linalg.inv(_S_w), _S_b)

        _eigval, _eigvec, = np.linalg.eigh(np.cov(self._W))
        # sp.linalg.eig might return complex values, so convert them to float
        _eigval, _eigvec = _eigval.real, _eigvec.real    
        # Order eigenvalues and eigenvectors
        _idx = np.argsort(_eigval)[::-1]
        _eigval = np.array(_eigval[_idx])
        _eigvec = np.array([_eigvec[:,_i] for _i in _idx])

        self._components = _eigvec[:self._n_components]

    """ Compute the mean for each class in the data
    :param data: Input data (n_samples x n_features)
    :param labels: Input data labels
    :return means: Dictionary of class:mean entries
    :return c_sizes: List of class sizes
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tuple _class_means(self, data, labels):

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
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _between_class(self, c_means, c_sizes, t_mean):

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
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _within_class(self, data, labels, c_means):

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