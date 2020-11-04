"""Dimensionality reduction on quotient spaces using an MDS method"""

import autograd.numpy as np
import autograd.numpy.linalg as LA
from autograd.numpy.linalg import matrix_power as mp

import pymanopt
from pymanopt.solvers import *
from pymanopt.manifolds import Oblique

import matplotlib.pyplot as plt
import random

# dreimac does not install properly on my system.
try:
    from dreimac.projectivecoords import ppca
except:
    from ppca import ppca
    print("""Loading personal version of PPCA. This may not be consistent with
        the published version""")

###############################################################################
# Main Algorithm
###############################################################################

def emds(
    X,
    D,
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


