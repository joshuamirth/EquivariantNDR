"""Methods for doing MDS with the chordal metric on projective space.

Throughout, we model `k` points on RP^n as `k` column vectors with unit norm,
or collectively an (n+1)*k matrix with unit norm columns. This corresponds to
the "Oblique" manifold defined in pymanopt.

"""

import autograd.numpy as np
import pymanopt
from pymanopt.manifolds import Oblique
from pymanopt.solvers import ConjugateGradient
from geometry import acos_validate, distance_to_weights

###############################################################################
# MDS Algorithms
################################################################################

def rp_mds(D, dim=2, X=None):
    """Wrapper function."""
    X_out = main_mds(D, dim=dim+1, X=X, space='real')
    return X_out

# def cp_mds(D, dim=1, X=None):
    # X_out = main_mds(D, dim=2*dim+2, X=X, space='complex')
    # return X_out

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
    if max_d > np.pi/2:
        print('WARNING: maximum value in distance matrix exceeds diameter of '\
            'projective space. Max value in distance matrix = %2.4f.' %max_d)
    manifold = pymanopt.manifolds.Oblique(dim, n)
    solver = pymanopt.solvers.ConjugateGradient(maxiter=5000)
    if space == 'real':
        cost, egrad = setup_RPn_cost(D)
    elif space == 'complex':
        cost = setup_CPn_cost(D, int(dim/2))
    problem = pymanopt.Problem(manifold=manifold, cost=cost, egrad=egrad)
    if X is None:
        X_out = solver.solve(problem)
    else:
        if X.shape[0] != dim:
            print('WARNING: initial condition does not match specified goal '\
                'dimension. Finding optimum in dimension %d' %X.shape[0])
        X_out = solver.solve(problem, x=X)
    return X_out

################################################################################
# Cost Functions
################################################################################

def setup_RPn_cost(D):
    """Create the cost functions for pymanopt.

    The cost function is given by
        F(X) = (1/2)*||W * |X^T X| - cos(D))||^2
    Where `W` is the weight matrix accounting for removing the arccos term.
    Currently only returns the cost. For better performance, could add gradient
    as a return value.

    Parameters
    ----------
    D : ndarray (n, n)
        Matrix of target distances.

    Returns
    -------
    cost : function
        Weighted Frobenius norm cost function.
    egrad : function
        Gradient (in Euclidean space) of cost function.

    """

    def cost(Y):
        """Weighted Frobenius norm cost function."""
        ip = acos_validate(Y.T@Y)
        return 0.5*np.linalg.norm(D - np.arccos(np.abs(ip)))**2
    def egrad(Y):
        """Derivative of the cost function."""
        # tmp = -1*(np.ones(D.shape) - (Y.T@Y)**2 + np.eye(D.shape[0]))**(-0.5)
        zero_tol = 1e-12
        tmp = np.ones(D.shape) - (Y.T@Y)**2
        idx = np.where(np.abs(tmp) < zero_tol)   # Avoid division by zero.
        tmp[idx] = 1
        tmp = -1*tmp**(-0.5)
        fill_val = np.min(tmp)  # All entries are negative.
        tmp[idx] = fill_val     # Make non-diagonal zeros large.
        np.fill_diagonal(tmp, 0)    # Ignore known zeros on diagonal.
        ip = acos_validate(Y.T@Y)
        return 2*Y@((np.arccos(np.abs(ip)) - D) * tmp * np.sign(Y.T@Y))
    return cost, egrad

# def setup_CPn_cost(D, n):
#     """Cost using geodesic metric on CPn."""
#     W = distance_to_weights(D)
#     C = np.cos(D)**2
#     i_mtx = np.vstack(
#         (np.hstack((np.zeros((n, n)), -np.eye(n))),
#         np.hstack((np.eye(n), np.zeros((n, n))))))
#     def cost(Y):
#         return 0.5*np.linalg.norm(W * ((Y.T @ Y)**2 + (Y.T @ (i_mtx@Y))**2 - C))**2
#     return cost
