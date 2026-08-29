"""
Graph construction utilities: similarity weights, k-NN graphs, and the
(normalised) graph Laplacian.
"""

import numpy as np
from scipy.spatial.distance import cdist


def gaussian_weights(X: np.ndarray, sigma: float) -> np.ndarray:
    """Fully-connected Gaussian similarity matrix: w_ij = exp(-||xi-xj||^2 / sigma^2)."""
    sq_dists = cdist(X, X, metric="sqeuclidean")
    return np.exp(-sq_dists / (sigma ** 2))


def knn_graph(X: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """
    Builds a symmetric k-nearest-neighbour graph with Gaussian similarity
    weights. Each vertex is connected to its k nearest neighbours; the
    result is symmetrised so W is a valid (undirected) weight matrix.
    """
    n = X.shape[0]
    sq_dists = cdist(X, X, metric="sqeuclidean")
    np.fill_diagonal(sq_dists, np.inf)

    W = np.zeros((n, n))
    for i in range(n):
        nn_idx = np.argsort(sq_dists[i])[:k]
        W[i, nn_idx] = np.exp(-sq_dists[i, nn_idx] / (sigma ** 2))

    # symmetrise: connect i-j if either i is a kNN of j, or vice versa
    W = np.maximum(W, W.T)
    return W


def graph_laplacian(W: np.ndarray, normalised: bool = True) -> np.ndarray:
    """
    Computes the graph Laplacian L = D - W (unnormalised) or the symmetric
    normalised Laplacian L_s = D^{-1/2} L D^{-1/2}.
    """
    degrees = W.sum(axis=1)
    D = np.diag(degrees)
    L = D - W

    if not normalised:
        return L

    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
    return d_inv_sqrt @ L @ d_inv_sqrt


def leading_eigenvectors(L: np.ndarray, m: int):
    """
    Returns the m eigenvectors of L corresponding to the m smallest
    eigenvalues (used as the low-dimensional spectral basis for the
    diffuse-interface energy minimisation).
    """
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[:m]
    return eigvals[idx], eigvecs[:, idx]
