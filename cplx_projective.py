""" Dim reduction on RPn using an MDS-type method. """
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import pymanopt
from pymanopt.manifolds import Oblique 
from pymanopt.solvers import ConjugateGradient

###############################################################################
# Algorithm Components
###############################################################################

def cp_mds(Y, D, max_iter=20, verbosity=1):
    """Projective multi-dimensional scaling algorithm.

    Detailed description in career grant, pages 6-7 (method 1).

    Parameters
    ----------
    Y : ndarray
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
    Y : ndarray
        Optimal configuration of points in RP^k.
    C : list
        List of costs at each iteration.

    """

    num_points = Y.shape[0]
    start_cost_list = []
    end_cost_list = []
    loop_cost_diff = np.inf
    percent_cost_diff = 100
    rank = LA.matrix_rank(Y)
    vprint('Finding projection onto RP^%i.' %(rank-1), 1, verbosity)
    W = distance_to_weights(D)
    S = np.sign(Y@Y.T)
    C = S*np.cos(D)
    if np.sum(S == 0) > 0:
        print('Warning: Some initial guess vectors are orthogonal, this may ' +
            'cause issues with convergence.')
    manifold = Oblique(rank, num_points) # Short, wide matrices.
    solver = ConjugateGradient()
    for i in range(0, max_iter):
        cost, egrad, ehess = setup_cost(D, S)
        start_cost_list.append(cost(Y.T))
        problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess,
            verbosity=verbosity)
        Y_new = solver.solve(problem, x=Y.T)
        Y_new = Y_new.T     # Y should be tall-skinny
        end_cost_list.append(cost(Y_new.T))
        S_new = np.sign(Y_new@Y_new.T)
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
        Y = Y_new
        C = C_new
        S = S_new
    return Y

###############################################################################
# Complex projective space geometry tools
###############################################################################

# Elements of complex projective space can be thought of as points on the
# 2n-sphere modulo an equivalence relation. I will think of the first
# n coordinates as the real part and the last n coordinates as the complex
# part. All functions will work with this real representation of the vectors.
# There is one conversion method in case of natural data with complex
# representation. Additionally, all data points are thought of as column
# vectors.

def CPn_validate(Y):
    """Check that Y is a valid element of CPn in the real representation."""
    valid = ( np.isrealobj(Y) * (np.mod(Y.shape[0], 2) == 0))
    if Y.ndim > 1:
        valid *= np.allclose(LA.norm(Y, axis=0), np.ones(Y.shape[1]))
    else:
        valid *= np.allclose(LA.norm(Y), np.ones(Y.shape))
    return valid

def realify(Y):
    """Convert data in n-dimensional complex space into 2n-dimensional real
    space.
    """
    Yreal = np.vstack((np.real(Y), np.imag(Y)))
    return Yreal

def complexify(Y):
    """Convert real 2n-dimensional points into n-dimensional complex vectors.
    """

    if np.mod(Y.shape[0], 2) != 0:
        raise ValueError('Cannot convert odd-dimensional vector to complex.')
    n = int(Y.shape[0]/2)
    Ycplx = Y[0:n] + 1j*Y[n:2*n]
    return Ycplx
    
def times_i(Y):
    """Multiply the real representation of a complex vector by i."""
    n = int(Y.shape[0]/2)
    i_mtx = np.vstack(
        (np.hstack((np.zeros((n, n)), -np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n)))))
    )
    iY = i_mtx@Y
    return iY

def CPn_distance_matrix(Y):
    """Construct the (exact) distance matrix of data Y on CP^n."""
    M = (Y.T@Y)**2 + (Y.T@times_i(Y))**2
    M = np.sqrt(M)
    acos_validate(M)
    D = np.arccos(M)
    return D

def norm_rotations(Y):
    """Compute a matrix S of complex numbers such that |<y_i, y_j>| is
    given by <y_i, s_ij y_j>."""
    sreal = Y.T @ Y
    simag = Y.T @ times_i(Y)
    norms = np.sqrt(sreal**2 + simag**2)
    sreal = sreal / norms
    simag = simag / norms
    return sreal, simag

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


###############################################################################
# Cost function and associated methods
###############################################################################

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D."""
    W_inv = (1 - np.cos(D)**2)     
    W = np.sqrt((W_inv+np.eye(D.shape[0],D.shape[1]))**-1
        - np.eye(D.shape[0],D.shape[1]))
    return W

def setup_cost(D, Sreal, Simag):
    """Create the cost functions for pymanopt, using explicit derivatives.

    Pymanopt performs optimization routines on manifolds, which require
    knowing the gradient and possibly hessian of the objective function
    (on the appropriate Riemannian manifold). For the weighted Frobenius
    norm objective function, there are explicit formulas defined here.
    The weighted Frobenius norm is given by
        F(Y) = ||W*S*cos(D) - W*Y.T@Y||^2
    where W is a weight matrix. Note that here Y is short and wide, so
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
#   @pymanopt.function.Autograd
    def cost(Y):
        """Weighted Frobenius norm cost function."""
        return 0.5*(LA.norm(W*(Creal - Y.T@Y))**2 + LA.norm(W*(Cimag - Y.T@times_i(Y)))**2)
    def grad(Y):
        """Derivative of weighted Frobenius norm cost."""
        return 2*Y@(W**2*(Y.T@Y-C))
    def hess(Y,H):
        """Second derivative (Hessian) of weighted Frobenius norm cost."""
        return 2*((W**2*(Y.T@Y-C))@H + (W**2*(Y@H.T + H@Y.T))@Y)
    return cost, grad, hess

###############################################################################
# Miscellanea
###############################################################################

def vprint(msg, level, verbosity):
    """Conventions: verbosity -1 is no printing. 0 is major positional
    info. 1 is all convergence-type info. Nested loops operate one level
    down (so 2 gives all top level information and convergence info on
    subloops, etc.
    """
    if verbosity >= level:
        print(msg)

def hopf(Y):
    """
    Map from CP^1 in C^2 = R^4 to the standard representation of S^2
    in R^3 using the Hopf fibration. This is useful for visualization
    purposes.

    Parameters
    ----------
    Y : ndarray (4, k)
        Array of `k` points in CP^1 < R^4 = C^2.

    Returns
    -------
    S : ndarray (3, k)
        Array of `k` points in S^2 < R^3.
 
    """
   
    if Y.shape[0] != 4:
        raise ValueError('Points must be in R^4 to apply Hopf map!.')
    S = np.vstack((
        [2*Y[0,:]*Y[1,:] + 2*Y[2,:]*Y[3,:]],
        [-2*Y[0,:]*Y[3,:] + 2*Y[1,:]*Y[2,:]],
        [Y[0,:]**2 + Y[2,:]**2 - Y[1,:]**2 - Y[3,:]**2]))
    return S

