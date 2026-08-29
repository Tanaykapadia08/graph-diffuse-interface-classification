"""
Graph-based semi-supervised classification via a diffuse-interface
(Ginzburg-Landau) energy model.

This follows the general approach of Bertozzi & Flenner, "Diffuse Interface
Models on Graphs for Classification of High Dimensional Data" (SIAM Review,
2016): the classification problem is relaxed from a discrete graph-cut into
a continuous Ginzburg-Landau energy on the graph, minimised via a
convex-splitting scheme in a low-dimensional spectral basis given by the
leading eigenvectors of the graph Laplacian.

This is an original, from-scratch implementation of the published method,
written as a portfolio piece alongside my Master's thesis on the same
topic. It is not the thesis's internal codebase, just a clean
demonstration of the underlying algorithm on a public toy dataset.
"""

import numpy as np

from graph_utils import knn_graph, graph_laplacian, leading_eigenvectors


def double_well_derivative(u: np.ndarray) -> np.ndarray:
    """W'(u) for the double-well potential W(u) = (1/4)(u^2 - 1)^2."""
    return u ** 3 - u


def convex_splitting_step(a: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray,
                           u0: np.ndarray, fidelity_idx: np.ndarray, fidelity_val: np.ndarray,
                           epsilon: float, dt: float, C: float, mu: float):
    """
    One convex-splitting update step in the spectral basis:

        a^{n+1} = (I + dt * epsilon * Lambda + dt * C * I)^{-1} *
                  (a^n + dt * V^T * (C * u^n - W'(u^n)/epsilon - mu * fidelity_term))

    where Lambda = diag(eigvals), V = eigvecs.
    """
    u_n = eigvecs @ a

    fidelity_term = np.zeros_like(u_n)
    fidelity_term[fidelity_idx] = mu * (u_n[fidelity_idx] - fidelity_val)

    rhs = C * u_n - double_well_derivative(u_n) / epsilon - fidelity_term
    rhs_spectral = eigvecs.T @ rhs

    denom = 1 + dt * epsilon * eigvals + dt * C
    a_next = (a + dt * rhs_spectral) / denom

    return a_next


def classify(X: np.ndarray, labels_known: dict, k: int = 10, sigma: float = 1.0,
             m: int = 20, epsilon: float = 1.0, dt: float = 0.01, C: float = 1.0,
             mu: float = 50.0, n_iter: int = 500, seed: int = 0):
    """
    Semi-supervised binary classification via graph diffuse-interface
    energy minimisation.

    Parameters
    ----------
    X : (n_samples, n_features) feature matrix
    labels_known : dict mapping vertex index -> label in {-1, +1}
                   (the small set of "known" labelled points)
    k, sigma      : k-NN graph construction parameters
    m             : number of leading Laplacian eigenvectors used as basis
    epsilon, dt, C, mu : Ginzburg-Landau / convex-splitting parameters
    n_iter        : number of convex-splitting iterations

    Returns
    -------
    u : (n_samples,) continuous class indicator, thresholded at 0 for the
        final +1 / -1 classification
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    W = knn_graph(X, k=k, sigma=sigma)
    L = graph_laplacian(W, normalised=True)
    eigvals, eigvecs = leading_eigenvectors(L, m)

    fidelity_idx = np.array(list(labels_known.keys()))
    fidelity_val = np.array(list(labels_known.values()), dtype=float)

    # random small initial perturbation around 0, per Bertozzi-Flenner
    u0 = 0.01 * (2 * rng.random(n) - 1)
    a = eigvecs.T @ u0

    for _ in range(n_iter):
        a = convex_splitting_step(a, eigvals, eigvecs, u0, fidelity_idx,
                                   fidelity_val, epsilon, dt, C, mu)

    u_final = eigvecs @ a
    return u_final, np.sign(u_final)
