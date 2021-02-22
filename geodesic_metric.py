"""Methods for doing MDS with the chordal metric on projective space.

Throughout, we model `k` points on RP^n as `k` column vectors with unit norm,
or collectively an (n+1)*k matrix with unit norm columns. This corresponds to
the "Oblique" manifold defined in pymanopt.

"""

import autograd.numpy as np
import pymanopt

###############################################################################
# MDS Algorithms
################################################################################

def rp_mds(D, dim=3, X=None):
    """Wrapper function."""
    X_out = main_mds(D, dim=dim, X=X, space='real')
    return X_out

def cp_mds(D, dim=4, X=None):
    X_out = main_mds(D, dim=dim, X=X, space='complex')
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
    if max_d > np.pi/2:
        print('WARNING: maximum value in distance matrix exceeds diameter of '\
            'projective space. Max value in distance matrix = %2.4f.' %max_d)
    manifold = pymanopt.manifolds.Oblique(dim, n)
    solver = pymanopt.solvers.ConjugateGradient()
    if space == 'real':
        cost = setup_RPn_cost(D)
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

################################################################################
# Cost Functions
################################################################################

def setup_RPn_cost(D):
    """Create the cost functions for pymanopt.

    The cost function is given by
        F(X) = ||W * ((X^T X)^2 - cos^2(D))||^2
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

    """

    W = distance_to_weights(D)
    C = np.cos(D)**2
    def cost(Y):
        """Weighted Frobenius norm cost function."""
        return 0.5*np.linalg.norm(W*(C - (Y.T@Y)**2))**2
    # def grad(Y):
        # """Derivative of weighted Frobenius norm cost."""
        # return 2*Y@(W**2*(Y.T@Y-C))
    # def hess(Y,H):
        # """Second derivative (Hessian) of weighted Frobenius norm cost."""
        # return 2*((W**2*(Y.T@Y-C))@H + (W**2*(Y@H.T + H@Y.T))@Y)
    return cost

def setup_CPn_cost(D, n):
    """Cost using geodesic metric on CPn."""
    W = distance_to_sq_weights(D)
    C = np.cos(D)**2
    cj_mtx = np.block([
        [np.eye(n), np.zeros((n, n))],
        [np.zeros((n, n)), -np.eye(n)]])
    def cost(Y):
        return 0.5*np.linalg.norm(((cj_mtx@Y).T @ Y) * (Y.T @ (cj_mtx@Y)) - C)**2
    return cost

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D."""
    W = np.sqrt((1 - np.cos(D)**2 + np.eye(D.shape[0]))**-1)
    return W

###############################################################################
# Projective space geometry tools
###############################################################################

def RPn_validate(Y):
    """Check that Y is a valid element of RPn."""
    valid = np.isrealobj(Y)
    if Y.ndim > 1:
        valid *= np.allclose(LA.norm(Y, axis=0), np.ones(Y.shape[1]))
    else:
        valid *= np.allclose(LA.norm(Y), np.ones(Y.shape))
    return bool(valid)

def CPn_validate(Y):
    """Check that Y is a valid element of CPn in the real representation."""
    valid = ( np.isrealobj(Y) * (np.mod(Y.shape[0], 2) == 0))
    if Y.ndim > 1:
        valid *= np.allclose(LA.norm(Y, axis=0), np.ones(Y.shape[1]))
    else:
        valid *= np.allclose(LA.norm(Y), np.ones(Y.shape))
    return valid

def RPn_distance_matrix(Y):
    """Construct the (exact) distance matrix of data Y on RP^n."""
    M = np.abs(Y.T@Y)
    acos_validate(M)
    D = np.arccos(M)
    np.fill_diagonal(D, 0)  # To avoid ripser glitch, need exact zeros.
    return D

def CPn_distance_matrix(Y):
    """Construct the (exact) distance matrix of data Y on CP^n."""
    n = int(Y.shape[0]/2)
    i_mtx = np.vstack(
        (np.hstack((np.zeros((n, n)), -np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n)))))
    )
    M = (Y.T@Y)**2 + (Y.T@(i_mtx@Y))**2
    M = np.sqrt(M)
    acos_validate(M)
    D = np.arccos(M)
    np.fill_diagonal(D, 0)
    return D

def acos_validate(M,tol=1e-6):
    """Replace values outside of domain of acos with +/- 1.

    Parameters
    ----------
    M : ndarray (m,n)
        Input matrix.
    tol : float
        Raises a warning if the values of `M` lie outside of
        [-1-tol,1+tol]. Default is `1e-6`.

    Returns
    -------
    M : ndarray (m,n)
        Matrix with values > 1 replaced by 1.0 and values < -1 replaced
        by -1.0. Modifies the input matrix in-place.

    Examples
    --------

    """

    if  np.max(M) > 1 + tol or np.min(M) < -1 - tol:
        print('Warning: matrix contained a value of %2.4f. Input may be '\
            'outside of [-1,1] by more than floating point error.' %np.max(M))
    big_vals = M >= 1.0
    M[big_vals] = 1.0
    small_vals = M <= -1.0
    M[small_vals] = -1.0
    return M
