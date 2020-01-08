#
# Author: Mihai Ionut Deaconu
#

cimport cython

import numpy as np
import scipy as sp
cimport numpy as cnp


""" Principal Component Analysis (PCA)

:param n_components: 
    Number of principal components to be selected
:param method: 
    Method of PC computation, through eigendecomposition ("eig") or SVD ("svd") -- default is "svd"
:param n_oversamples: 
    Number of oversamples used in randomised SVD -- default is 10
:param n_iter: 
    Number of power iterations used in randomised SVD -- default is 2
"""
cdef class PCA:

    cdef int _n_components, _n_oversamples, _n_iter
    cdef str _method

    cdef cnp.float64_t[:, :] _components
    cdef float _explained_variance

    def __cinit__(self, int n_components, str method="svd", int n_oversamples=10, int n_iter=2):

        if n_components <= 0:
            raise ValueError("Number of components must be positive.")

        if method not in ["svd", "eig"]:
            raise ValueError("method must be either 'svd' or 'eig'.")

        if n_oversamples < 1:
            raise ValueError("Number of oversamples must be positive.")

        if n_iter < 1:
            raise ValueError("Number of iterations must be positive.")

        self._n_components = n_components
        self._method = method
        self._n_oversamples = n_oversamples
        self._n_iter = n_iter

    """ Fit the model
    Compute the principal components and total explained variance

    :param data: 
        Non-sparse dataset in NumPy format (n_samples x n_features)
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray[cnp.float64_t, ndim=2] data):

        if data.size == 0:
            raise ValueError("Data cannot be empty.")
        if self._n_components > data.shape[1]:
            raise ValueError("Number of components can't be greater than the number of data features.")

        cdef cnp.ndarray[cnp.float64_t, ndim=2] centered_data = self._center(np.transpose(data))
        cdef cnp.ndarray[cnp.float64_t, ndim=2] U, Uh, Q, B
        cdef cnp.ndarray[cnp.float64_t, ndim=1] s, variance
        cdef cnp.ndarray[cnp.long_t, ndim=1] idx
        cdef cnp.ndarray eigval, eigvec

        cdef Py_ssize_t n_samples = centered_data.shape[0], n_features = centered_data.shape[1]
        cdef Py_ssize_t i

        cdef int n_dimensions

        if self._method == "eig":

            # Compute eigenvalues and eigenvectors of covariance matrix
            eigval, eigvec, = sp.linalg.eigh(np.cov(centered_data))
            # sp.linalg.eig might return complex values, so convert them to float
            eigval, eigvec = eigval.real, eigvec.real

            # Order eigenvalues and eigenvectors
            idx = np.argsort(eigval)[::-1]
            eigval = np.array(eigval[idx])
            eigvec = np.array([eigvec[:,i] for i in idx])

            self._components = eigvec[:self._n_components]
            self._explained_variance = np.sum(eigval[:self._n_components]) / np.sum(eigval) * 100

        elif self._method == "svd":

            if max(n_samples, n_features) < 500 or self._n_components > .8 * min(n_samples, n_features): # full SVD

                U, s, _ = sp.linalg.svd(centered_data)
                variance = (s ** 2) / (n_samples - 1)

                self._components = U[:,:self._n_components]
                self._explained_variance = np.sum(variance[:self._n_components]) / np.sum(variance) * 100

            else: # randomised SVD

                n_dimensions = self._n_components + self._n_oversamples
                # Sample (k + p) i.i.d. vectors from a normal distribution
                Q = np.random.normal(size=(n_features, n_dimensions))
                # Perform QR decompotision on (A @ At)^q @ A @ Q
                for _ in range(self._n_iter):
                    Q, _ = np.linalg.qr(np.dot(centered_data, Q))
                    Q, _ = np.linalg.qr(np.dot(np.transpose(centered_data), Q))
                Q, _ = np.linalg.qr(np.dot(centered_data, Q))

                # Compute low-dimensional B
                B = np.dot(np.transpose(Q), centered_data)

                # Find principal components of B
                Uh, s, _ = sp.linalg.svd(B)
                variance = (s ** 2) / (n_samples - 1)
                U = np.dot(Q, Uh)

                self._components = U[:,:self._n_components]
                self._explained_variance = np.sum(variance[:self._n_components]) / np.sum(variance) * 100

    """ Data centering
    Center features by removing the mean

    :param data: 
        Dataset in NumPy format (n_samples x n_features)

    :return: 
        Data centered in the origin
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _center(self, cnp.ndarray[cnp.float64_t, ndim=2] data):    
        return data - np.mean(data, axis=0)

    @property
    def n_components(self):       
        return self._n_components

    @property
    def method(self):       
        return self._method

    @property
    def n_oversamples(self):       
        return self._n_oversamples

    @property
    def n_iter(self):       
        return self._n_iter

    @property
    def components(self):       
        return np.asarray(self._components)

    @property
    def explained_variance(self):
        return self._explained_variance

    @n_components.setter
    def n_components(self, int n_components):
        if n_components < 1:
            raise ValueError("Number of components must be positive.")
        self._n_components = n_components

    @method.setter
    def method(self, str method):
        if method not in ["svd", "eig"]:
            raise ValueError("method must be either 'svd' or 'eig'.")
        self._method = method

    @n_oversamples.setter
    def n_oversamples(self, int n_oversamples):
        if n_oversamples < 1:
            raise ValueError("Number of oversamples must be positive.")
        self._n_oversamples = n_oversamples

    @n_iter.setter
    def n_iter(self, int n_iter):
        if n_iter < 1:
            raise ValueError("Number of iterations must be positive.")
        self._n_iter = n_iter


""" Independent Component Analysis (ICA)

:param n_components:  
    Number of independent components to be selected
:param method: 
    Method of IC computation, through "symmetric" or "deflationary" algorithms -- default is "symmetric"
"""
cdef class ICA:

    cdef int _n_components
    cdef str _method

    cdef cnp.float64_t[:, :] _components

    def __cinit__(self, int n_components, str method="symmetric"):

        if n_components < 1:
            raise ValueError("Number of components must be positive.")

        if method not in ["symmetric", "deflationary"]:
            raise ValueError("method must be either 'symmetric' or 'deflationary'.")

        self._n_components = n_components
        self._method = method

    """ Fit the model to the data
    Compute the individual components

    :param data: 
        Non-sparse dataset in NumPy format (n_samples x n_features)
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray[cnp.float64_t, ndim=2] data):

        if data.size == 0:
            raise ValueError("Data cannot be empty.")
        if self._n_components > data.shape[1]:
            raise ValueError("Number of components can't be greater than the number of features in the data.")

        cdef cnp.ndarray[cnp.float64_t, ndim=2] centered_data = self._center(data)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] whitened_data, whitening_matrix, components
        cdef cnp.ndarray[cnp.float64_t, ndim=1] component
        cdef cnp.ndarray eigval, eigvec
        cdef list components_ = []

        cdef Py_ssize_t n_features = data.shape[1]

        # Data whitening
        eigval, eigvec, = sp.linalg.eigh(np.cov(np.transpose(centered_data)))
        eigval, eigvec = eigval.real, eigvec.real
        whitening_matrix = np.dot(eigvec, np.dot(np.diag(1 / np.sqrt(eigval+1e-10)), np.transpose(eigvec)))
        whitened_data = np.dot(centered_data, np.transpose(whitening_matrix))
        # Test the constraint of the whitened data:
        # cov(X @ W) = I
        np.testing.assert_allclose(np.cov(np.transpose(whitened_data)), np.eye(n_features), atol=1e-7)
        # Test the constraint of the whitening matrix:
        # W^T @ W = cov(X)^-1
        np.testing.assert_allclose(np.linalg.inv(np.cov(np.transpose(data))), np.dot(np.transpose(whitening_matrix), whitening_matrix), atol=1e-7)
        whitened_data = np.transpose(whitened_data)

        if self._method == "deflationary":

            for _ in range(self._n_components):
                component = self._compute_unit(whitened_data, components_)
                components_.append(component)
            components = np.vstack(components_)

        elif self._method == "symmetric":
            components = self._compute_matrix(whitened_data)
            
        # Test for independence of the components:
        # S^T @ S = I
        np.testing.assert_allclose(np.dot(np.transpose(components), components), np.eye(n_features), atol=1e-7)

        self._components = np.dot(components, whitening_matrix)

    """ Data centering
    Center features by removing the mean

    :param data: 
        Dataset in NumPy format (n_samples x n_features)

    :return: 
        Data centered in the origin
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

    :param X: 
        Whitened data
    :param W: 
        Existing independent components

    :return:
        New indendent component
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _compute_unit(self, cnp.ndarray[cnp.float64_t, ndim=2] X, list W):

        cdef cnp.ndarray[cnp.float64_t, ndim=1] w = np.random.rand(X.shape[0])
        cdef cnp.ndarray[cnp.float64_t, ndim=1] _w, w0

        w /= np.linalg.norm(w)
        for _ in range(1000):
            w0 = w
            w = (1 / X.shape[1]-1) * np.dot(X, self._g(np.dot(np.transpose(w), X))) - (1 / X.shape[1]-1) * np.dot(self._dg(np.dot(np.transpose(w), X)), np.ones((X.shape[1], 1))) * w
            for w_ in W:
                w = w - np.dot(np.transpose(w), w_) * w_
            w /= np.linalg.norm(w) 

            # Check for convergence
            if 1 - np.abs(np.dot(np.transpose(w0), w)) < 1e-10:
                break
        return w

    """ Compute all independent component in parallel

    :param X: 
        Whitened data

    :return: 
        New indendent component matrix
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _compute_matrix(self, cnp.ndarray[cnp.float64_t, ndim=2] X):

        cdef cnp.ndarray[cnp.float64_t, ndim=2] W = np.random.rand(X.shape[0], self._n_components)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] W0
        cdef Py_ssize_t i

        # Normalise component vectors in W
        W = W / np.linalg.norm(W, axis=-1)[:, np.newaxis]
        # Symmetric orthogonalisation
        # W = (W @ W.T) ^ -1/2 @ W
        eigval, eigvec = sp.linalg.eigh(np.dot(W, np.transpose(W)))
        W = np.dot(np.dot(eigvec * (1 / np.sqrt(eigval)), np.transpose(eigvec)), W)

        for _ in range(1000):
            W0 = W
            for i in range(self._n_components):
                W[:, i] = (1 / X.shape[1]-1) * np.dot(X, self._g(np.dot(np.transpose(W[:, i]), X))) - (1 / X.shape[1]-1) * np.dot(self._dg(np.dot(np.transpose(W[:, i]), X)), np.ones((X.shape[1], 1))) * W[:, i]
            # Symmetric orthogonalisation
            # W = (W @ W.T) ^ -1/2 @ W
            eigval, eigvec = sp.linalg.eigh(np.dot(W, np.transpose(W)))
            W = np.dot(np.dot(eigvec * (1 / np.sqrt(eigval)), np.transpose(eigvec)), W)

            # Check for convergence
            if 1 - max(np.abs(np.diag(np.dot(W0, W)))) < 1e-10:
                break
        return W

    @property
    def n_components(self):       
        return self._n_components

    @property
    def method(self):       
        return self._method

    @property
    def components(self):
        return np.asarray(self._components)

    @n_components.setter
    def n_components(self, int n_components):
        if n_components <= 0:
            raise ValueError("Number of components must be positive.")
        self._n_components = n_components

    @method.setter
    def method(self, str method):
        if method not in ["symmetric", "deflationary"]:
            raise ValueError("method must be either 'symmetric' or 'deflationary'.")
        self._method = method


""" Linear Discriminant Analysis (LDA)

:param n_components: 
    Number of dimensions to be selected
"""
cdef class LDA:

    cdef int _n_components

    cdef cnp.float64_t[:, :] _components, _W

    def __cinit__(self, int n_components):

        if n_components < 1:
            raise ValueError("Number of components must be positive.")

        self._n_components = n_components

    """ Fit the model
    Compute the linear discriminant

    :param data: 
        Dataset in NumPy format (n_samples x n_features)
    :param labels: 
        Array of labels associated to each data point (n_samples x 1)
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray[cnp.float64_t, ndim=2] data, cnp.ndarray labels):

        if data.size == 0:
            raise ValueError("Data cannot be empty.")
        if self._n_components > data.shape[1]:
            raise ValueError("Number of components can't be greater than the number of features in the data.")

        if labels.size == 0:
            raise ValueError("Labels cannot be empty.")
        if data.shape[0] != labels.size:
            raise ValueError("Labels and data must have the same length")

        cdef cnp.ndarray[cnp.float64_t, ndim=2] S_b, S_w, c_means
        cdef cnp.ndarray[cnp.float64_t, ndim=1] t_mean, 
        cdef cnp.ndarray[cnp.long_t, ndim=1] c_sizes, c_indices, idx
        cdef cnp.ndarray eigval, eigvec, c_names

        cdef Py_ssize_t i

        c_names, c_indices = np.unique(labels, return_inverse=True)
        c_sizes = np.bincount(c_indices)

        c_means = self._class_means(data, labels, c_sizes, c_indices)
        t_mean = c_means.mean(axis=0)

        S_b = self._between_class(c_means, c_sizes, t_mean)
        S_w = self._within_class(data, labels, c_means, c_sizes, c_indices)

        self._W = np.dot(np.linalg.inv(S_w), S_b)

        eigval, eigvec, = np.linalg.eigh(np.cov(self._W))
        # sp.linalg.eig might return complex values, so convert them to float
        eigval, eigvec = eigval.real, eigvec.real    
        # Order eigenvalues and eigenvectors
        idx = np.argsort(eigval)[::-1]
        eigval = np.array(eigval[idx])
        eigvec = np.array([eigvec[:,i] for i in idx])

        self._components = eigvec[:self._n_components]

    """ Compute the mean for each class

    :param data: 
        Dataset in NumPy format (n_samples x n_features)
    :param labels: 
        Array of labels associated to each data point (n_samples x 1)
    :param c_sizes: 
        Array of class sizes
    :param c_indices: 
        Array of class indices

    :return c_means: 
        Array of class means
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _class_means(self, cnp.ndarray[cnp.float64_t, ndim=2] data, 
                                                               cnp.ndarray labels, 
                                                               cnp.ndarray[cnp.long_t, ndim=1] c_sizes, 
                                                               cnp.ndarray[cnp.long_t, ndim=1] c_indices):

        cdef cnp.ndarray[cnp.float64_t, ndim=2] c_means = np.zeros((len(c_sizes), data.shape[1]))
        cdef Py_ssize_t i, j

        # Create a cumulative matrix with each class as a row
        for i in range(labels.size):
            c_means[c_indices[i]] += data[i]

        # Create class mean matrix
        for i in range(c_means.shape[0]):
            for j in range(c_means.shape[1]):
                c_means[i, j] = c_means[i, j] / c_sizes[i]

        return c_means

    """ Between-class Matrix

    :param c_means: 
        Array of class means
    :param c_sizes: 
        Array of class sizes
    :param t_mean: 
        Total mean

    :return: 
        Between-class matrix
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _between_class(self, cnp.ndarray[cnp.float64_t, ndim=2] c_means, 
                                                                 cnp.ndarray[cnp.long_t, ndim=1] c_sizes, 
                                                                 cnp.ndarray[cnp.float64_t, ndim=1] t_mean):

        cdef cnp.ndarray[cnp.float64_t, ndim=2] S_b
        cdef Py_ssize_t i

        S_b = np.zeros((len(c_means[1]), len(c_means[1])))
        for i in range(c_sizes.size):
            S_b += c_sizes[i] * np.outer((c_means[i] - t_mean), (c_means[i] - t_mean))

        return S_b

    """ Within-class Matrix

    :param data: 
        Dataset in NumPy format (n_samples x n_features)
    :param labels: 
        Array of labels associated to each data point (n_samples x 1)
    :param c_means: 
        Array of class means
    :param c_sizes: 
        Array of class sizes
    :param c_indices: 
        Array of class indices

    :return: 
        Within-class matrix
    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] _within_class(self, cnp.ndarray[cnp.float64_t, ndim=2] data, 
                                                                cnp.ndarray labels, 
                                                                cnp.ndarray[cnp.float64_t, ndim=2] c_means, 
                                                                cnp.ndarray[cnp.long_t, ndim=1] c_sizes,
                                                                cnp.ndarray[cnp.long_t, ndim=1] c_indices): 

        cdef cnp.ndarray[cnp.float64_t, ndim=2] c_centered_data = np.copy(data)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] S_w
        cdef Py_ssize_t i, j

        # Remove the mean from each data entry point
        for i in range(c_centered_data.shape[0]):
            for j in range(c_centered_data.shape[1]):
                c_centered_data[i, j] = data[i, j] - c_means[c_indices[i], j]

        S_w = np.zeros((len(c_means[1]), len(c_means[1])))
        for i in range(c_sizes.size):
            S_w += np.dot(np.transpose(c_centered_data[[j for j in range(labels.size) if c_indices[j] == i]]), c_centered_data[[j for j in range(labels.size) if c_indices[j] == i]])

        return S_w

    @property
    def n_components(self):       
        return self._n_components

    @property
    def transformation_matrix(self):
        return self._W

    @property
    def components(self):
        return np.asarray(self._components)

    @n_components.setter
    def n_components(self, int n_components):
        if n_components <= 0:
            raise ValueError("Number of components must be positive.")
        self._n_components = n_components