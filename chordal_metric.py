"""Methods for doing MDS with the sine metric on projective space.

Throughout, we model `k` points on RP^n as `k` column vectors with unit norm,
or collectively an (n+1)*k matrix with unit norm columns. This corresponds to
the "Oblique" manifold defined in pymanopt.

"""

import autograd.numpy as np
import pymanopt

def sine_distance_matrix(X):
    D = np.sqrt(1 - (X.T@X)**2)
    return D

def rp_mds(D, dim=3, X=None):
    """MDS via gradient descent with sine metric.

    Parameters
    ----------
    D : ndarray (n, n)
        Goal distance matrix.
    dim : int, optional
        Goal dimension (of ambient Euclidean space). Default is `dim = 3`.
    X : ndarray (dim, n), optional
        Initial value for gradient descent. `n` points in dimension `dim`. If
        both a dimension and an initial condition are specified, the initial
        condition overrides the dimension.

    """

    n = D.shape[0]
    max_d = np.max(D)
    if max_d > 1:
        print('WARNING: maximum value in distance matrix exceeds diameter of '\
            'projective space. Max distance = $2.4f.' %max_d)
    manifold = pymanopt.manifolds.Oblique(dim, n)
    solver = pymanopt.solvers.ConjugateGradient()
    cost = setup_cost(D)
    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    if X is None:
        X_out = solver.solve(problem)
    else:
        if X.shape[0] != dim:
            print('WARNING: initial condition does not match specified goal '\
                'dimension. Finding optimum in dimension %d' %X.shape[0])
        X_out = solver.solve(problem, x=X)
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
