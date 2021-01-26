""" Dim reduction on RPn using an MDS-type method. """
import autograd.numpy as np
import autograd.numpy.linalg as LA
#import numpy as np
#import numpy.linalg as LA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import pymanopt
from pymanopt.manifolds import Oblique, Product
from pymanopt.solvers import ConjugateGradient

###############################################################################
# Algorithm Components
###############################################################################

def cp_mds(Y, D, max_iter=20, v=1):
    """Projective multi-dimensional scaling algorithm.

    Detailed description in career grant, pages 6-7 (method 1).

    Parameters
    ----------
    Y : ndarray (2n+2, k)
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
    Y : ndarray (2n+2, k)
        Optimal configuration of points in CP^n.
    C : list
        List of costs at each iteration.

    """

    dim = Y.shape[0]
    num_points = Y.shape[1]
    start_cost_list = []
    end_cost_list = []
    loop_diff = np.inf
    percent_cost_diff = 100
    # rank = LA.matrix_rank(Y)
    vprint('Finding optimal configuration in CP^%i.'
        %((dim-2)//2), 1, v)
    W = distance_to_weights(D)
    Sreal, Simag = norm_rotations(Y)
    manifold = Oblique(dim, num_points)
    # Oblique manifold is dim*num_points matrices with unit-norm columns.
    solver = ConjugateGradient()
    for i in range(0, max_iter):
        # AUTOGRAD VERSION
        cost = setup_autograd_cost(D, Sreal, Simag, int(dim/2))
        # ANALYTIC VERSION:
        #cost, egrad, ehess = setup_cost(D, Sreal, Simag)
        start_cost_list.append(cost(Y))
        # AUTOGRAD VERSION:
        problem = pymanopt.Problem(manifold, cost, verbosity=v)
        # ANALYTIC VERSION:
        #problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess,
        #   verbosity=v)
        Y_new = solver.solve(problem, x=Y)
        end_cost_list.append(cost(Y_new))
        Sreal_new, Simag_new = norm_rotations(Y_new)
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
        Y = Y_new
        Sreal = Sreal_new
        Simag = Simag_new
    return Y

def cp_mds_reg(Y, D, lam=1.0, v=1):
    """Version of MDS in which "signs" are also an optimization parameter.

    Rather than performing a full optimization and then resetting the
    sign matrix, here we treat the signs as a parameter `A = [a_ij]` and
    minimize the cost function
        F(Y,A) = ||W*(Y^H(A*Y) - cos(D))||^2 + lambda*||A - Y^HY/|Y^HY| ||^2
    Lambda is a regularization parameter we can experiment with. The
    collection of data, `Y`, is treated as a point on the `Oblique`
    manifold, consisting of `k*n` matrices with unit-norm columns. Since
    we are working on a sphere in complex space we require `k` to be
    even. The first `k/2` entries of each column are the real components
    and the last `k/2` entries are the imaginary parts. 

    Parameters
    ----------
    Y : ndarray (k, n)
        Initial guess for data.
    D : ndarray (k, k)
        Goal distance matrix.
    lam : float, optional
        Weight to give regularization term.
    v : int, optional
        Verbosity

    Returns
    -------
    Y_opt : ndarray (k, n)
        Collection of points optimizing cost.

    """

    dim = Y.shape[0]
    num_points = Y.shape[1]
    W = distance_to_weights(D)
    Sreal, Simag = norm_rotations(Y)
    A = np.vstack((np.reshape(Sreal, (1, num_points**2)),
        np.reshape(Simag, num_points**2)))
    cp_manifold = Oblique(dim, num_points)
    a_manifold = Oblique(2, num_points**2)
    manifold = Product((cp_manifold, a_manifold))
    solver = ConjugateGradient()
    cost = setup_reg_autograd_cost(D, int(dim/2), num_points)
    problem = pymanopt.Problem(cost=cost, manifold=manifold)
    Yopt, Aopt = solver.solve(problem)
    return Yopt, Aopt


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
    # Note that the formula for weights is undefined on the diagonal of
    # D, that is, on any element of D equal to zero. Here we set the
    # diagonal to ones, but in the real projective version we use zero.
    W = np.sqrt((1 - np.cos(D)**2 + np.eye(D.shape[0]))**-1)
    return W

def distance_to_sq_weights(D):
    """Weights to use with cos^2(D) error form."""
    W = 1/np.sin(2*(D+np.eye(D.shape[0]))) + np.eye(D.shape[0])
    return W

def setup_autograd_sq_cost(D, n):
    W = distance_to_sq_weights(D)
    C = np.cos(D)**2
    i_mtx = np.vstack(
        (np.hstack((np.zeros((n, n)), -np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n)))))
    )
    cj_mtx = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]])
    def cost(Y):
        return 0.5*np.linalg.norm(((cj_mtx@Y).T @ Y) * (Y.T @ (cj_mtx@Y)) - C)**2
    return cost

def setup_autograd_cost(D, Sreal, Simag, n):
    i_mtx = np.vstack(
        (np.hstack((np.zeros((n, n)), -np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n)))))
    )
    W = distance_to_weights(D)
    Creal = Sreal*np.cos(D)
    Cimag = Simag*np.cos(D)
    def cost(Y):
        """Weighted Frobenius norm cost function."""
        return 0.5*(np.linalg.norm(W*(Creal - Y.T@Y))**2 + np.linalg.norm(W*(Cimag - Y.T@(i_mtx@Y)))**2)
    return cost

def setup_reg_autograd_cost(D, k, n):
    i_mtx = np.vstack(
        (np.hstack((np.zeros((k, k)), -np.eye(k))),
        np.hstack((np.eye(k), np.zeros((k, k)))))
    )
    W = distance_to_weights(D)
    C = np.reshape(np.cos(D), (1, n**2))
    def cost(pair):
        """Weighted Frobenius norm cost function."""
        Y = pair[0]
        A = pair[1]
        Re = np.linalg.norm(W*(np.reshape(C*A[0,:], (n, n)) - Y.T@Y))
        Im = np.linalg.norm(W*(np.reshape(C*A[1,:], (n, n)) - Y.T@(i_mtx@Y)))
        return 0.5*(Re**2 + Im**2)
    return cost



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

