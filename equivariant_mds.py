"""Dimensionality reduction on quotient spaces using an MDS method"""

import autograd.numpy as np
import autograd.numpy.linalg as LA
from autograd.numpy.linalg import matrix_power as mp

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

import pymanopt
from pymanopt.solvers import *
from pymanopt.manifolds import Oblique

# dreimac does not install properly on my system.
try:
    from dreimac.projectivecoords import ppca
except:
    from ppca import ppca
    print("""Loading personal version of PPCA. This may not be consistent with
        the published version""")

###############################################################################
# Main Algorithms
###############################################################################

def pmds(
    D,
    X=None,
    dim=3,
    max_iter=20,
    convergence_tol=1e-4
    verbosity=1,
    autograd=False
    pmo_solve='cg'
):
    """MDS on projective space.

    Detailed description in career grant, pages 6-7 (method 1).

    Parameters
    ----------
    D : ndarray
        Square distance matrix determining cost.
    X : ndarray, optional
        Initial guess of points in RP^k. Result will lie on RP^k for
        same k as the initial guess. If no initial guess is provided an
        initial guess is automatically generated. Providing an initial
        guess generally produces better results.
    dim : int, optional
        Dimension (of ambient Euclidean space) to reduce into. This is
        one greater than the dimension of projective space, so `dim==3`
        will give the reduction onto :math:`\mathbb{R}P^2`. Default is
        3. Overridden by rank of `X` if `X` is supplied.
    max_iter : int, optional
        Maximum number of times to iterate the optimization loop.
        Default is 20.
    convergence_tol : float, optional
        Minimum decrease in strain for which to continue with iterations.
        Default is 1e-4.
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

    num_points = D.shape[0]
    if X is not None:
        dim = LA.matrix_rank(X)
    else:
        # TODO: implement a simple way of getting points on sphere.
    if verbosity > 0:
        print('Finding projection onto RP^%i.' %(dim-1))
    W = distance_to_weights(D)
    S = np.sign(X@X.T)
    C = S*np.cos(D)
    if np.sum(S == 0) > 0:
        print('Warning: Some initial guess vectors are orthogonal, this may ' +
            'cause issues with convergence.')
    cost = setup_cost(D,S)
    cost_list = [cost(X.T)]
    true_cost = setup_cost(projective_distance_matrix(X),S)
    true_cost_list = [true_cost(X.T)]
    manifold = Oblique(dim,num_points) # Short, wide matrices.
    if pmo_solve == 'nm':
        solver = NelderMead()
    if pmo_solve == 'ps':
        solver = ParticleSwarm()
    if pmo_solve == 'tr':
        solver = TrustRegions()
    if pmo_solve == 'sd':
        solver = SteepestDescent()
    else:
        solver = ConjugateGradient()
    for i in range(0,max_iter):
        if autograd:
            cost = setup_cost(D,S)
            problem = pymanopt.Problem(manifold, cost, verbosity=verbosity)
        else:
            cost, egrad, ehess = setup_cost(D,S,return_derivatives=True)
            problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess, verbosity=verbosity)
        if pmo_solve == 'cg' or pmo_solve == 'sd' or pmo_solve == 'tr':
            # Use initial condition with gradient-based solvers.
            X_new = solver.solve(problem,x=X.T)
        else:
            X_new =  solver.solve(problem)
        X_new = X_new.T     # X should be tall-skinny
        cost_oldS = cost(X_new.T)
        cost_list.append(cost_oldS)
        S_new = np.sign(X_new@X_new.T)
        C_new = S_new*np.cos(D)
        cost_new = setup_cost(D,S_new)
        cost_newS = cost_new(X_new.T)
        S_diff = ((LA.norm(S_new - S))**2)/4
        percent_S_diff = 100*S_diff/S_new.size
        percent_cost_diff = 100*(cost_list[i] - cost_list[i+1])/cost_list[i]
        true_cost = setup_cost(projective_distance_matrix(X),S)
        true_cost_list.append(true_cost(X_new.T))
        if verbosity > 0:
            print('Through %i iterations:' %(i+1))
            print('\tTrue cost: %2.2f' %true_cost(X_new.T))
            print('\tComputed cost: %2.2f' %cost_list[i+1])
            print('\tPercent cost difference: % 2.2f' %percent_cost_diff)
            print('\tPercent Difference in S: % 2.2f' %percent_S_diff)
            print('\tComputed cost with new S: %2.2f' %cost_newS)
            print('\tDifference in cost matrix: %2.2f' %(LA.norm(C-C_new)))
        if S_diff < 1:
            print('No change in S matrix. Stopping iterations')
            break
        if percent_cost_diff < .0001:
            print('No significant cost improvement. Stopping iterations.')
            break
        if i == max_iter:
            print('Maximum iterations reached.')
        # Update variables:
        X = X_new
        C = C_new
        S = S_new
    return X, cost_list, true_cost_list

def lmds(
    D,
    X=None
    max_iter=20,
    verbosity=1,
    autograd=False
    pmo_solve='cg'

):
    """MDS on lens spaces."""

def gmds(
    D,
    X=None
    max_iter=20,
    verbosity=1,
    autograd=False
    pmo_solve='cg'
):
    """MDS on Grassmannians."""

###############################################################################
# Utility methods.
###############################################################################

def projective_cost():
def lens_cost():
def grassmannian_cost():

# General:
def acos_validate():
def distance_to_weights():

# For projective:
# TODO: determine if this is actually needed when using pymanopt.
def cholesky_rep():

# For lens:
def g_action_matrix():
def optimal_rotation():
def get_masks():
# TODO: could probably merge optimal_rotation get_masks into single,
# more efficient, function.

# For grassmannian:

###############################################################################
# Initialization methods.
###############################################################################

# TODO: determine if these can just be imported from elsewhere.
def ppca():
def lpca():
def gpca():

def geodesic_distance_matrix():

def initial_guess():



# TODO: don't write a general algorithm - there's too many differences.
def emds(
    D,
    X,
    q=1,
    max_iter=20,
    verbosity=1,
    autograd=False,
    pmo_solve='cg'
):
    """General equivariant multi-dimensional scaling algorithm.

    Takes a collection of points `X` on a sphere modulo a group G, and
    returns representation `Y` on the sphere which has pairwise
    distances optimally aligned to the distance matrix D. Currently G
    is permitted to be any finite cyclic group G = Z/qZ for some
    positive integer q.

    Parameters
    ----------
    X : ndarray (d*n)
        Initial data points in low-dimensional space, taken to be some
        quotient of a sphere S^(d-1). Each column of X represents a data
        point. Thus each column must have unit norm. The matrix rank of
        X determines the dimension to which the data will be reduced. 
    D : ndarray (n*n)
        Distance matrix determining cost. 
    q : int, optional
        Integer determining which cyclic group to use for quotient.
        Default is 1, meaning the trivial quotient, and the reduction
        happens on the sphere. Choosing q=2 gives projective space, and
        q>=3 is the lens space L^d_q.
    max_iter : int, optional
        Number of times to iterate the loop. Default is 20. Rarely are
        more required.
    verbosity : int, optional
        If positive, print output relating to convergence conditions at
        each iteration.
    autograd : bool, optional
        If true, use autograd to automatically compute derivatives.
        Unnecessary on lens spaces since the analytic gradient is known.
        Retained for testing purposes.
    pmo_solve : {'cg','sd','tr','nm','ps'}, optional
        Minimization tool to use in pymanopt. Default is a Riemannian
        manifold version of conjugate gradient, which generally performs
        well.

    Returns
    -------
    Y : ndarray (d*n)
        Output data points on S^(d-1) modulo G. Each column has unit
        norm.
    C : float list
        List of cost at each iteration.

    Notes
    -----
    The sphere S^n modulo Z/qZ is a lens space. The special case of q=2
    is real projective space. The objective function optimized is the
    Hadamard semi-norm
    .. math:: F(Y) = \|W\odot(\cos(D)-Y^TY)\|^2
    where the weights W are determined by the distance matrix D.

    Examples
    --------

    >>> import data_examples
    >>> X = data_examples.circleRPn()
    >>> D = geo_distance_matrix(X,k=5)
    >>> X0 = epca(X,2)
    >>> Y = emds(X,D,p=2)
    
    """






###############################################################################
# Miscellaneous
###############################################################################

def acos_validate(M):
    """Replace values in M outside of domain of acos with +/- 1.

    Parameters
    ----------
    M : ndarray, mutable
        Matrix of values that are approximately in [-1,1].

    Returns
    -------
    M : ndarray
        The original matrix with values > 1 replaced with 1 and values <
        -1 replaced by -1.

    """

    big_vals = M >= 1.0
    M[big_vals] = 1.0
    small_vals = M <= -1.0
    M[small_vals] = -1.0
    return M

def distance_to_weights(D,tol=10.0**-14):
    """Compute the weight matrix W from the distance matrix D.

    Parameters
    ----------
    D : ndarray (m*n)
        Matrix representing a metric or dissimilarity.
    tol : float, optional
        Tolerance around zero. Computing `W` involves taking the
        pointwise reciprocal of entries in `D`. To avoid division by
        zero errors, values less than `tol` are not inverted.

    Returns
    -------
    W : ndarray (m*n)
        Weights corresponding to D

    Notes
    -----
    In order to remove the arccos from the objective function
        ||arccos(X.T@X) - D||,
    cosine is taken and the norm reweighted by
        W[ij] = (1-cos^2(D[ij])^(-1/2).
    (This is justified by a mean-value theorem argument.) However, `W`
    undefined if D[ij] = 0. For a distance matrix it must hold that
    D[ii] = 0, so zeros must be handled. We choose to set the weight
    corresponding to any 0 in `D` to 0 since the structure of the
    problem guarantees the correct values will appear on the diagonal
    regardless of the weight placed there. This also permits the metric
    `D` to be represented by an upper- or lower-triangular matrix. If
    `D` is not a true distance matrix or contains very small distances
    zeroing these values may have unintended consequences.

    """

    W_inv = (1 - np.cos(D)**2)     
    bad_vals = np.abs(D) < tol
    W_inv[bad_vals] = 1
    W = np.sqrt(W_inv**-1)
    W[bad_vals] = 0
    return W

###############################################################################
# Output and Plotting
###############################################################################

def plot_RP2(X,axes=None,pullback=True,compare=False,Z=[]):
    """Plot data reduced onto RP2"""

    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111,projection='3d')
    axes.scatter(X[:,0],X[:,1],X[:,2])
    if pullback:
        Y = -X
        axes.scatter(Y[:,0],Y[:,1],Y[:,2])
    if compare:
        axes.scatter(Z[:,0],Z[:,1],Z[:,2])
    plt.suptitle('Plot on RP^2')
    return axes


