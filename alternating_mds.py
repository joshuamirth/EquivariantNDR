""" Dim reduction on RPn using an MDS-type method. """
import autograd.numpy as np
import autograd.numpy.linalg as LA
#import numpy as np
#import numpy.linalg as LA
import pymanopt
from pymanopt.manifolds import Oblique, Product
from pymanopt.solvers import ConjugateGradient
from geometry import norm_rotations, times_i

###############################################################################
# Real Projective Space Algoirthms
###############################################################################

def rp_mds(D, X, max_iter=20, verbosity=1):
    """Projective multi-dimensional scaling algorithm.

    Detailed description in career grant, pages 6-7 (method 1).

    Parameters
    ----------
    X : ndarray
        Initial guess of points in RP^k. Result will lie on RP^k for
        same k as the initial guess.
    D : ndarray
        Square distance matrix determining cost.
    max_iter : int, optional
        Number of times to iterate the loop. Will eventually be updated
        to a better convergence criterion. Default is 20.
    verbosity : int, optional
        If positive, print output relating to convergence conditions at each
        iteration.
    solve_prog : string, optional
        Choice of algorithm for low-rank correlation matrix reduction.
        Options are "pymanopt" or "matlab", default is "pymanopt".

    Returns
    -------
    X : ndarray
        Optimal configuration of points in RP^k.
    C : list
        List of costs at each iteration.

    """

    num_points = X.shape[0]
    start_cost_list = []
    end_cost_list = []
    loop_cost_diff = np.inf
    percent_cost_diff = 100
    rank = LA.matrix_rank(X)
    vprint('Finding projection onto RP^%i.' %(rank-1), 1, verbosity)
    W = distance_to_weights(D)
    S = np.sign(X@X.T)
    C = S*np.cos(D)
    if np.sum(S == 0) > 0:
        print('Warning: Some initial guess vectors are orthogonal, this may ' +
            'cause issues with convergence.')
    manifold = Oblique(rank, num_points) # Short, wide matrices.
    solver = ConjugateGradient()
    for i in range(0, max_iter):
        # cost, egrad, ehess = setup_RPn_cost(D, S)
        cost = setup_square_cost(D)
        start_cost_list.append(cost(X.T))
        # problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess,
            # verbosity=verbosity)
        problem = pymanopt.Problem(manifold, cost, verbosity=verbosity)
        X_new = solver.solve(problem, x=X.T)
        X_new = X_new.T     # X should be tall-skinny
        end_cost_list.append(cost(X_new.T))
        S_new = np.sign(X_new@X_new.T)
        C_new = S_new*np.cos(D)
        S_diff = ((LA.norm(S_new - S))**2)/4
        percent_S_diff = 100*S_diff/S_new.size
        iteration_cost_diff = start_cost_list[i] - end_cost_list[i]
        if i > 0:
            loop_cost_diff = end_cost_list[i-1] - end_cost_list[i]
            percent_cost_diff = 100*loop_cost_diff/end_cost_list[i-1]
        vprint('Through %i iterations:' %(i+1), 1, verbosity)
        vprint('\tCost at start: %2.4f' %start_cost_list[i], 1, verbosity)
        vprint('\tCost at end: %2.4f' %end_cost_list[i], 1, verbosity)
        vprint('\tCost reduction from optimization: %2.4f' %iteration_cost_diff, 1, verbosity)
        vprint('\tCost reduction over previous loop: %2.4f' %loop_cost_diff, 1, verbosity)
        vprint('\tPercent cost difference: % 2.4f' %percent_cost_diff, 1, verbosity)
        vprint('\tPercent Difference in S: % 2.2f' %percent_S_diff, 1, verbosity)
        vprint('\tDifference in cost matrix: %2.2f' %(LA.norm(C-C_new)), 1, verbosity)
        if S_diff < 1:
            vprint('No change in S matrix. Stopping iterations', 0, verbosity)
            break
        if percent_cost_diff < .0001:
            vprint('No significant cost improvement. Stopping iterations.', 0,
                verbosity)
            break
        if i == max_iter:
            vprint('Maximum iterations reached.', 0, verbosity)
        # Update variables:
        X = X_new
        C = C_new
        S = S_new
    return X

###############################################################################
# Complex Projective Space Algoirthms
###############################################################################

def cp_mds(D, X, max_iter=20, v=1):
    """Projective multi-dimensional scaling algorithm.

    Detailed description in career grant, pages 6-7 (method 1).

    Parameters
    ----------
    X : ndarray (2n+2, k)
        Initial guess of `k` points in CP^n. Result will lie on CP^n for
        same `n` as the initial guess. (Each column is a data point.)
    D : ndarray (k, k)
        Square distance matrix determining cost.
    max_iter : int, optional
        Number of times to iterate the loop. Will eventually be updated
        to a better convergence criterion. Default is 20.
    v : int, optional
        Verbosity. If positive, print output relating to convergence
        conditions at each iteration.

    Returns
    -------
    X : ndarray (2n+2, k)
        Optimal configuration of points in CP^n.
    C : list
        List of costs at each iteration.

    """

    dim = X.shape[0]
    num_points = X.shape[1]
    start_cost_list = []
    end_cost_list = []
    loop_diff = np.inf
    percent_cost_diff = 100
    # rank = LA.matrix_rank(X)
    vprint('Finding optimal configuration in CP^%i.'
        %((dim-2)//2), 1, v)
    W = distance_to_weights(D)
    Sreal, Simag = norm_rotations(X)
    manifold = Oblique(dim, num_points)
    # Oblique manifold is dim*num_points matrices with unit-norm columns.
    solver = ConjugateGradient()
    for i in range(0, max_iter):
        # AUTOGRAD VERSION
        cost = setup_CPn_autograd_cost(D, Sreal, Simag, int(dim/2))
        # ANALYTIC VERSION:
        #cost, egrad, ehess = setup_CPn_cost(D, Sreal, Simag)
        start_cost_list.append(cost(X))
        # AUTOGRAD VERSION:
        problem = pymanopt.Problem(manifold, cost, verbosity=v)
        # ANALYTIC VERSION:
        #problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess,
        #   verbosity=v)
        X_new = solver.solve(problem, x=X)
        end_cost_list.append(cost(X_new))
        Sreal_new, Simag_new = norm_rotations(X_new)
        S_diff = LA.norm(Sreal_new - Sreal)**2 + LA.norm(Simag_new - Simag)**2
        iter_diff = start_cost_list[i] - end_cost_list[i]
        if i > 0:
            loop_diff = end_cost_list[i-1] - end_cost_list[i]
            percent_cost_diff = 100*loop_diff/end_cost_list[i-1]
        vprint('Through %i iterations:' %(i+1), 1, v)
        vprint('\tCost at start: %2.4f' %start_cost_list[i], 1, v)
        vprint('\tCost at end: %2.4f' %end_cost_list[i], 1, v)
        vprint('\tCost reduction from optimization: %2.4f' %iter_diff, 1, v)
        vprint('\tCost reduction over previous loop: %2.4f' %loop_diff, 1, v)
        vprint('\tPercent cost difference: % 2.4f' %percent_cost_diff, 1, v)
        vprint('\tDifference in S: % 2.2f' %S_diff, 1, v)
        if S_diff < .0001:
            vprint('No change in S matrix. Stopping iterations', 0, v)
            break
        if percent_cost_diff < .0001:
            vprint('No significant cost improvement. Stopping iterations.', 0,
                v)
            break
        if i == max_iter:
            vprint('Maximum iterations reached.', 0, v)
        # Update variables:
        X = X_new
        Sreal = Sreal_new
        Simag = Simag_new
    return X

def cp_mds_reg(X, D, lam=1.0, v=1, maxiter=1000):
    """Version of MDS in which "signs" are also an optimization parameter.

    Rather than performing a full optimization and then resetting the
    sign matrix, here we treat the signs as a parameter `A = [a_ij]` and
    minimize the cost function
        F(X,A) = ||W*(X^H(A*X) - cos(D))||^2 + lambda*||A - X^HX/|X^HX| ||^2
    Lambda is a regularization parameter we can experiment with. The
    collection of data, `X`, is treated as a point on the `Oblique`
    manifold, consisting of `k*n` matrices with unit-norm columns. Since
    we are working on a sphere in complex space we require `k` to be
    even. The first `k/2` entries of each column are the real components
    and the last `k/2` entries are the imaginary parts.

    Parameters
    ----------
    X : ndarray (k, n)
        Initial guess for data.
    D : ndarray (k, k)
        Goal distance matrix.
    lam : float, optional
        Weight to give regularization term.
    v : int, optional
        Verbosity

    Returns
    -------
    X_opt : ndarray (k, n)
        Collection of points optimizing cost.

    """

    dim = X.shape[0]
    num_points = X.shape[1]
    W = distance_to_weights(D)
    Sreal, Simag = norm_rotations(X)
    A = np.vstack((np.reshape(Sreal, (1, num_points**2)),
        np.reshape(Simag, num_points**2)))
    cp_manifold = Oblique(dim, num_points)
    a_manifold = Oblique(2, num_points**2)
    manifold = Product((cp_manifold, a_manifold))
    solver = ConjugateGradient(maxiter=maxiter, maxtime=float('inf'))
    cost = setup_reg_autograd_cost(D, int(dim/2), num_points, lam=lam)
    problem = pymanopt.Problem(cost=cost, manifold=manifold)
    Xopt, Aopt = solver.solve(problem, x=(X, A))
    Areal = np.reshape(Aopt[0,:], (num_points, num_points))
    Aimag = np.reshape(Aopt[1,:], (num_points, num_points))
    return Xopt, Areal, Aimag

###############################################################################
# Cost function and associated methods
###############################################################################

def setup_RPn_cost(D,S,return_derivatives=False):
    """Create the cost functions for pymanopt, using explicit derivatives.

    Pymanopt performs optimization routines on manifolds, which require
    knowing the gradient and possibly hessian of the objective function
    (on the appropriate Riemannian manifold). For the weighted Frobenius
    norm objective function, there are explicit formulas defined here.
    The weighted Frobenius norm is given by
        F(X) = ||W*S*cos(D) - W*X.T@X||^2
    where W is a weight matrix. Note that here X is short and wide, so
    each column is a data point (a vector with norm one). The gradient
    and hessian of F are computed in Grubisic and Pietersz.

    Parameters
    ----------
    D : ndarray (n, n)
        Matrix of target distances.
    S : ndarray (n, n)
        Matrix of signs.

    Returns
    -------
    cost : function
        Weighted Frobenius norm cost function.
    grad : function
        Gradient of cost function.
    hess : function
        Hessian of cost function.

    """

    W = distance_to_weights(D)
    C = S*np.cos(D)
#   @pymanopt.function.Autograd
    def cost(X):
        """Weighted Frobenius norm cost function."""
        return 0.5*np.linalg.norm(W*(C-X.T@X))**2
    def grad(X):
        """Derivative of weighted Frobenius norm cost."""
        return 2*X@(W**2*(X.T@X-C))
    def hess(X,H):
        """Second derivative (Hessian) of weighted Frobenius norm cost."""
        return 2*((W**2*(X.T@X-C))@H + (W**2*(X@H.T + H@X.T))@X)
    return cost, grad, hess

def setup_RPn_square_cost(D,return_derivatives=False):
    """Create the cost functions for pymanopt, using explicit derivatives.

    Pymanopt performs optimization routines on manifolds, which require
    knowing the gradient and possibly hessian of the objective function
    (on the appropriate Riemannian manifold). For the weighted Frobenius
    norm objective function, there are explicit formulas defined here.
    The weighted Frobenius norm is given by
        F(X) = ||W*S*cos(D) - W*X.T@X||^2
    where W is a weight matrix. Note that here X is short and wide, so
    each column is a data point (a vector with norm one). The gradient
    and hessian of F are computed in Grubisic and Pietersz.

    Parameters
    ----------
    D : ndarray (n, n)
        Matrix of target distances.
    S : ndarray (n, n)
        Matrix of signs.

    Returns
    -------
    cost : function
        Weighted Frobenius norm cost function.
    grad : function
        Gradient of cost function.
    hess : function
        Hessian of cost function.

    """

    W = distance_to_weights(D)
    C = np.cos(D)**2
#   @pymanopt.function.Autograd
    def cost(X):
        """Weighted Frobenius norm cost function."""
        return 0.5*np.linalg.norm(W*(C - (X.T@X)**2))**2
    return cost
    # def grad(X):
        # """Derivative of weighted Frobenius norm cost."""
        # return 2*X@(W**2*(X.T@X-C))
    # def hess(X,H):
        # """Second derivative (Hessian) of weighted Frobenius norm cost."""
        # return 2*((W**2*(X.T@X-C))@H + (W**2*(X@H.T + H@X.T))@X)
    # return cost, grad, hess

def setup_CPn_autograd_cost(D, Sreal, Simag, n):
    i_mtx = np.vstack(
        (np.hstack((np.zeros((n, n)), -np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n)))))
    )
    W = distance_to_weights(D)
    C = np.cos(D)**2
    # Creal = Sreal*np.cos(D)
    # Cimag = Simag*np.cos(D)
    def cost(X):
        """Weighted Frobenius norm cost function."""
        F = 0.5*np.linalg.norm((X.T @ X)**2 + (X.T @ (i_mtx@X))**2 - C)**2
        return F
    return cost

def setup_reg_autograd_cost(D, k, n, lam=1):
    i_mtx = np.vstack(
        (np.hstack((np.zeros((k, k)), -np.eye(k))),
        np.hstack((np.eye(k), np.zeros((k, k)))))
    )
    W = distance_to_weights(D)
    C = np.reshape(np.cos(D), (1, n**2))
    def cost(pair):
        """Weighted Frobenius norm cost function."""
        X = pair[0]
        A = pair[1]
        XX = (X.T@X)**2 + (X.T@(i_mtx@X))**2 + 1/6  # linear ~ of sqrt()
        Re = np.linalg.norm(W*(np.reshape(C*A[0,:], (n, n)) - X.T@X))
        Im = np.linalg.norm(W*(np.reshape(C*A[1,:], (n, n)) - X.T@(i_mtx@X)))
        reg_Re = np.linalg.norm(XX*np.reshape(A[0,:], (n, n)) - X.T@X)
        reg_Im = np.linalg.norm(XX*np.reshape(A[1,:], (n, n)) - X.T@(i_mtx@X))
        return 0.5*(Re**2 + Im**2) + lam*(reg_Re**2 + reg_Im**2)
    return cost

def setup_CPn_cost(D, Sreal, Simag):
    """Create the cost functions for pymanopt, using explicit derivatives.

    Pymanopt performs optimization routines on manifolds, which require
    knowing the gradient and possibly hessian of the objective function
    (on the appropriate Riemannian manifold). For the weighted Frobenius
    norm objective function, there are explicit formulas defined here.
    The weighted Frobenius norm is given by
        F(X) = ||W*S*cos(D) - W*X.T@X||^2
    where W is a weight matrix. Note that here X is short and wide, so
    each column is a data point (a vector with norm one). The gradient
    and hessian of F are computed in Grubisic and Pietersz.

    Parameters
    ----------
    D : ndarray (n, n)
        Matrix of target distances.
    Sreal : ndarray (n, n)
        Real parts of the "signs."
    Simag : ndarray (n, n)
        Imaginary parts of the "signs."

    Returns
    -------
    cost : function
        Weighted Frobenius norm cost function.
    grad : function
        Gradient of cost function.
    hess : function
        Hessian of cost function.

    """

    W = distance_to_weights(D)
    Creal = Sreal*np.cos(D)
    Cimag = Simag*np.cos(D)
    def cost(X):
        """Weighted Frobenius norm cost function."""
        return 0.5*(LA.norm(W*(Creal - X.T@X))**2 + LA.norm(W*(Cimag - X.T@times_i(X)))**2)
    def grad(X):
        """Derivative of weighted Frobenius norm cost."""
        return 2*X@(W**2*(X.T@X-C))
    def hess(X,H):
        """Second derivative (Hessian) of weighted Frobenius norm cost."""
        return 2*((W**2*(X.T@X-C))@H + (W**2*(X@H.T + H@X.T))@X)
    return cost, grad, hess
