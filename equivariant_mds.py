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
    q=2,
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
        Default is 2, so that the quotient is real projective space.
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
        F(Y) = ||W*(cos(D)-Y.T@Y)||^2
    where the weights W are determined by the distance matrix D.

    """






###############################################################################
# Miscellaneous
###############################################################################

def acos_validate(M):
    """Replace values in M outside of domain of acos with +/- 1."""
    big_vals = M >= 1.0
    M[big_vals] = 1.0
    small_vals = M <= -1.0
    M[small_vals] = -1.0
    return M

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D."""
    W_inv = (1 - np.cos(D)**2)     
    W = np.sqrt((W_inv+np.eye(D.size))**-1 - np.eye(D.size))
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



###############################################################################
# Toy data generation methods
###############################################################################

def circleRPn(
    dim=4,
    segment_points=50,
    num_segments=4,
    noise=False,
    v=0.2,
    randomize=True
):
    """Construct points on a "kinked" circle in RP^d.

    Constructs a curve of evenly-spaced points along the great circle
    from e_i to e_{i+1} in R^{d+1}, starting at e_0 and finishing at
    e_i with i = num_segments, then returns to -e_0.

    It is recommended that dim==num_segments, otherwise the resulting
    data matrix will not be full rank, which can cause issues later.
    Similarly, the output data is randomly permuted so that the first n
    points are not on the same linear subspace, generically.

    Parameters
    ----------
    dim : int, optional
        Dimension of RP^d to work on (ambient euclidean space is dim+1).
    segment_points : int, optional
        Number of points along each segment of curve.
    num_segments : int, optional
        Number of turns to make before returning to start point.

    Returns
    -------
    X : ndarray
        Array of coordinate values in R^{d+1}.
    """
    if int(num_segments) != num_segments:
        raise ValueError("""Number of segments must be a positive integer.
            Supplied value was %2.2f.""" %num_segments)
    if num_segments < 1 or dim < 1:
        raise ValueError("""Number of segments and dimension must be positive
            integers. Supplied values were num_segments = %2.2f and dimension
            = %2.2f""" %(num_segments,dim))    
    if dim < num_segments:
        raise ValueError("""Value of dimension must be larger than number of
            segments. Supplied dimension was %i and number of segments was
            %i""" %(dim,num_segments))
    rng = np.random.default_rng(57)
    num_points = segment_points*(num_segments+1)
    theta = np.linspace(0,np.pi/2,segment_points,endpoint=False)
    X = np.zeros((num_points,dim+1))
    segment_curve = np.array([np.cos(theta),np.sin(theta)]).T
    for i in range(0,num_segments):
        X[i*segment_points:(i+1)*segment_points,i:i+2] = segment_curve
    X[num_segments*segment_points:num_points,0] = -np.sin(theta)
    X[num_segments*segment_points:num_points,num_segments] = np.cos(theta)
    if randomize:
        X = rng.permutation(X)
    if noise:
        N = v*rng.random((dim+1,num_points))
        Xt = (X.T + N)/LA.norm(X.T+N,axis=0)
        X = Xt.T
    return X

def bezier_RPn(ctrl_points,N=100,noise=0):
    """Define a weird curve for testing purposes.
    
    Parameters
    ----------
    ctrl_points : ndarray
        n*d array where each row is a control point of a Bezier curve
        in R^d. The first row is the start point of the curve, and the
        last row is the end point.
    N : int, optional
        Number of points to put on curve. Default is 1000.
    
    Returns
    -------
    B : ndarray
        Array (N*d) with each row a point on the curve. Normalized to
        lie on the sphere.

    """

    t = np.reshape(np.linspace(0,1,N),(N,1))
    deg = ctrl_points.shape[0]-1
    dim = ctrl_points.shape[1]
    P = np.reshape(ctrl_points[0,:],(1,dim))
    B = ((1-t)**deg)@P
    for i in range(1,deg):
        P = np.reshape(ctrl_points[i,:],(1,dim))
        B = B + comb(deg,i)*((t**i)*((1-t)**(deg-i)))@P
    P = np.reshape(ctrl_points[deg,:],(1,dim))
    B = B + (t**deg)@P
    if noise > 0 :
        ns = noise*(np.random.rand(N,dim)-.5)
        B = B+ns
    B = (B.T/LA.norm(B,axis=1)).T
    return B   


