import numpy as np

""" Principal Component Analysis (PCA)

:param data: Dataset to perform PCA on (n_samples x n_features)
:param n_components: Number of Principal Components to be selected
"""
class PCA:

    def __init__(self, data, n_components):
        self.__data = data
        self.__n_components = n_components

        if not isinstance(self.__data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if self.__data.size == 0:
            raise ValueError("Data is an empty array.")

        if self.__n_components < 1 or self.__n_components > self.__data.shape[1]:
            raise ValueError("Invalid number of components, must be between 1 and n_features.")

        self.__s_matrix = np.cov(self.__data.T)
        # Compute eigenvalues and eigenvectors of covariance matrix
        self.__eigval, self.__eigvec, = np.linalg.eig(self.__s_matrix)
        self.__eigval, self.__eigvec = self.__order_eig(self.__eigval, self.__eigvec)

        self.__components = self.__eigvec[:self.__n_components]
        self.__explained_variance = np.sum(self.__eigval[:self.__n_components]) / np.sum(self.__eigval) * 100

    """ Order Eigenvalues/Eigenvectors
    Given a vector of eigenvectors and corresponding eigenvalues, it orders the 
    eigenvectors with respect to their eigenvalues.

    :param eigval: List of eigenvalues
    :param eigvec: List of eigenvectors 
    :return eigval: Sorted list of eigenvalues
    :return eigvec: Sorted list of eigenvectors
    """
    def __order_eig(self, eigval, eigvec):
        eigpairs = [(eigval[i], eigvec[:,i]) for i in range(len(eigval))]
        eigpairs.sort(reverse=True)
        eigval = np.array([elem.real for elem,_ in eigpairs])
        eigvec = np.array([elem.real for _,elem in eigpairs])
        return eigval, eigvec

    def get_scatter_matrix(self):
        return self.__s_matrix

    def get_eigvec(self):
        return self.__eigvec

    def get_eigval(self):
        return self.__eigval

    def get_components(self):
        return self.__components

    def get_explained_variance(self):
        return self.__explained_variance

""" Independent Component Analysis (ICA)

:param data: Dataset to perform ICA on (n_samples x n_features)
:param n_components: Number of Independent Components to be selected
"""
class ICA:

    def __init__(self, data, n_components):
        self.__data = data
        self.__n_components = n_components

        if not isinstance(self.__data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if self.__data.size == 0:
            raise ValueError("Data is an empty array.")

        if self.__n_components < 1 or self.__n_components > self.__data.shape[1]:
            raise ValueError("Invalid number of components, must be between 1 and n_features.")

        self.__whitened_data, self.__whitening_matrix = self.__whiten(self.__data)
        self.__whitened_data = np.transpose(self.__whitened_data)
        self.__raw_components = []
        for i in range(self.__n_components):
            component = self.__compute_unit(self.__whitened_data, self.__raw_components)
            self.__raw_components.append(component)
        self.__raw_components = np.vstack(self.__raw_components)
        self.__components = np.dot(self.__raw_components, self.__whitening_matrix)


    """ Data Whitening
    Decorrelates the components of the data

    :param: Data to whiten
    :return X_w: Whitened data
    :return K: Whitening matrix
    """
    def __whiten(self, data):
        data = self.__center(data)
        eigval, eigvec, = np.linalg.eig(np.cov(np.transpose(data)))
        W = np.dot(eigvec, np.dot(np.diag(1 / np.sqrt(eigval+1e-9)), eigvec.T))
        data_w = np.dot(data, W.T)
        return data_w, W

    """ Data Centering
    Center features by removing the mean

    :param: Data to center
    :return: Centered data
    """
    def __center(self, data):
        return data - np.mean(data, axis=0)

    """ Kurtosis Function
    kurt(x) = 4 * u^3
    """
    def __g(self, u):
        return 4 * u**3

    """ Derivated Kurtosis Function
    kurt'(x) = 12 * u^2
    """
    def __dg(self, u):
        return 12 * u**2

    """ Compute one Independent Component in ICA

    :param X: Whitened data
    :param W: Existing independent components
    :return: New indendent component
    """
    def __compute_unit(self, X, W):
        w = np.random.rand(X.shape[0])
        w /= np.linalg.norm(w)
        for iter in range(5000):
            w0 = w
            w = (1 / X.shape[1]-1) * np.dot(X, self.__g(np.dot(np.transpose(w), X))) - (1 / X.shape[1]-1) * np.dot(self.__dg(np.dot(np.transpose(w), X)), np.ones((X.shape[1], 1))) * w
            for w_ in W:
                w = w - np.dot(np.transpose(w), w_) * w_
            w /= np.linalg.norm(w) 
            # Check for convergence
            if (1 - np.abs(np.dot(np.transpose(w0), w)) < 1e-10):
                break
        return w

    def get_whitening_matrix(self):
        return self.__whitening_matrix

    def get_whitened_data(self):
        return self.__whitened_data

    def get_raw_components(self):
        return self.__raw_components

    def get_components(self):
        return self.__components

""" Linear Discriminant Analysis (LDA)

:param data: Dataset to perform LDA on (n_samples x n_features)
:param labels: List of labels (classes) associated with each data point
:param n_dimensions: Number of dimensions to be selected
"""
class LDA:
    
    def __init__(self, data, labels, n_dimensions):
        self.__data = data
        self.__labels = labels
        self.__n_dimensions = n_dimensions

        if not isinstance(self.__data, np.ndarray):
            raise ValueError("Data must be a NumPy array.")

        if self.__data.size == 0:
            raise ValueError("Data is an empty array.")

        if self.__n_dimensions < 1 or self.__n_dimensions > self.__data.shape[1]:
            raise ValueError("Invalid number of dimensions, must be between 1 and n_features.")

        __c_means, __c_sizes = self.__class_means(self.__data, self.__labels)
        __t_mean = np.array(list(__c_means.values())).mean(axis=0)
        __S_b = self.__between_class(__c_means, __c_sizes, __t_mean)
        __S_w = self.__within_class(self.__data, self.__labels, __c_means)
        self.__W = np.dot(np.linalg.inv(__S_w), __S_b)

        self.__eigval, self.__eigvec, = np.linalg.eig(np.cov(self.__W))
        self.__eigval, self.__eigvec = self.__order_eig(self.__eigval, self.__eigvec)
        self.__components = self.__eigvec[:self.__n_dimensions]

    """ Compute the mean for each class in the data

    :param X: Data
    :param y: Labels
    :return means: Dictionary of class:mean entries
    :return c_sizes: List of class sizes
    """
    def __class_means(self, data, labels):
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
    def __between_class(self, c_means, c_sizes, t_mean):
        S_b = np.zeros((len(t_mean), len(t_mean)))
        for class_ in c_means.keys():
            S_b += c_sizes[class_] * np.outer((c_means[class_] - t_mean), (c_means[class_] - t_mean))
        return S_b

    """ Within-Class Matrix

    :param X: Data
    :param y: Labels
    :param c_means: List of class means
    :return: Within-class matrix
    """
    def __within_class(self, data, labels, c_means):
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
    def __order_eig(self, eigval, eigvec):
        eigpairs = [(eigval[i], eigvec[:,i]) for i in range(len(eigval))]
        print(eigpairs)
        eigpairs.sort(reverse=True)
        eigval = np.array([elem.real for elem,_ in eigpairs])
        eigvec = np.array([elem.real for _,elem in eigpairs])
        return eigval, eigvec

    def get_transformation_matrix(self):
        return self.__W

    def get_eigvec(self):
        return self.__eigvec

    def get_eigval(self):
        return self.__eigval

    def get_components(self):
        return self.__components