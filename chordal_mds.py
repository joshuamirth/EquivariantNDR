"""Methods for doing MDS with the chordal metric on projective space.

Throughout, we model `k` points on RP^n as `k` column vectors with unit norm,
or collectively an (n+1)*k matrix with unit norm columns. This corresponds to
the "Oblique" manifold defined in pymanopt.

"""

import autograd.numpy as np
import pymanopt
from pymanopt.manifolds import Oblique
from pymanopt.solvers import ConjugateGradient

def rp_mds(D, dim=2, X=None):
    """Wrapper function."""
    X_out = main_mds(D, dim=dim+1, X=X, space='real')
    return X_out

def main_mds(D, dim=3, X=None, space='real'):
    """MDS via gradient descent with the chordal metric.

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
    field : str
        Choice of real or complex version. Options 'real', 'complex'. If
        'complex' dim must be even.

    """

    n = D.shape[0]
    max_d = np.max(D)
    if max_d > 1:
        print('WARNING: maximum value in distance matrix exceeds diameter of '\
            'projective space. Max distance = $2.4f.' %max_d)
    manifold = Oblique(dim, n)
    solver = ConjugateGradient()
    if space == 'real':
        cost = setup_cost(D)
    elif space == 'complex':
        cost = setup_CPn_cost(D, int(dim/2))
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
    A = np.ones(D.shape)
    C = A - D
    def cost(X):
        F = np.linalg.norm((X.T@X)*(X.T@X) - C)**2
        return F
    return cost

################################################################################
# Complex Projective Version #
################################################################################

def setup_CPn_cost(D, n):
    """Cost using chordal metric on CPn."""
    i_mtx = np.block([
        [np.zeros((n, n)), -np.eye(n)],
        [np.eye(n), np.zeros((n, n))]
        ])
    A = np.ones(D.shape)
    C = A - D
    def cost(X):
        F = np.linalg.norm((X.T @ X)**2 + (X.T @ (i_mtx@X))**2 - C)**2
        return F
    return cost

def cp_mds(D, dim=4, X=None):
    X_out = main_mds(D, dim=dim, X=X, space='complex')
    return X_out
