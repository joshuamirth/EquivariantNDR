"""Methods for doing MDS with the sine metric on projective space."""
import autograd.numpy as np
import pymanopt

def sine_distance_matrix(X):
    D = np.sqrt(1 - (X.T@X)**2)
    return D

def rp_mds(X, D):
    """MDS via gradient descent with sine metric.

    Parameters
    ----------
    X : ndarray (n, k)
        Initial value for gradient descent. `n` points in dimension `k`.
    D : ndarray (n, n)
        Goal distance matrix.
    """
    n = X.shape[0]
    k = X.shape[1]
    manifold = pymanopt.manifolds.Oblique(k, n)
    solver = pymanopt.solvers.ConjugateGradient()
    cost = setup_cost(D)
    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    X_out = solver.solve(problem, x=X)
    X_out = X_out.T
    return X_out

def setup_cost(D):
    """Cost when using sine distance."""
    # TODO: add analytic gradient to this method.
    def cost(X):
        A = np.ones(D.shape)
        C = A - D
        F = np.linalg.norm((X.T@X)*(X.T@X) - C)**2
        return F
    return cost
