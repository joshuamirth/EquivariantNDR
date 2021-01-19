""" Dim reduction on RPn using an MDS-type method. """
# import matlab.engine    # for LRCM MIN.
#import autograd.numpy as np
#import autograd.numpy.linalg as LA
import numpy as np
import numpy.linalg as LA
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.special import comb  # n Choose k
import matplotlib.pyplot as plt
# Setup for pymanopt.
import pymanopt
from pymanopt.manifolds import Oblique 
from pymanopt.solvers import *
import os
import random 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: decide where best to place these utility functions.
from pipeline import acos_validate

# dreimac does not install properly on my system.
try:
    from dreimac.projectivecoords import ppca
except:
    from ppca import ppca
    print('Loading personal version of PPCA. This may not be consistent with '\
        'the published version.')

#def circleDM(n=50,diam=1.57):
#    """Distance matrix for 2n evenly spaced circle points."""
#    theta = np.hstack((
#        np.linspace(0,diam,n,endpoint=False),
#        np.linspace(diam,0,n,endpoint=False)
#    ))
#    D = np.array([np.roll(theta,i) for i in range(0,2*n)])
#    return D
    
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
    num_points : int
        Number of points in data set.
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
    return X, num_points

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

###############################################################################
# Algorithm Components
###############################################################################

def initial_guess(data,dim,guess_method='ppca'):
    """Use PPCA and Cholesky factor to get an initial input for lrcm_min

    Parameters
    ----------
    data : ndarray
        Input data as n*d matrix
    dim : int
        Dimension (of projective space) on which to place an initial
        guess. Must be less than the dimension of the original data.
    guess_method : ('ppca','random','iterative_ppca')
        Method of obtaining initial guess. Default is to use PPCA.

    Returns
    -------
    X : ndarray
        Data on the Cholesky manifold.
    
    Notes
    -----
    The Cholesky manifold is defined in Grubisic and Pietersz.
    
    """
    
    if guess_method == 'ppca':
        V = ppca(data, dim)
        X = V['X']
    elif guess_method == 'random':
        X = 2*np.random.random_sample((dim+1,data.shape[0]))-1
        X = X/LA.norm(X,axis=0)
        X = X.T
    elif guess_method == 'iterative_ppca':
        N = data.shape[1]
        X = data
        for i in range(N-2,dim-1,-1):
            V = ppca(X, i)
            X = V['X']
    X = cholesky_rep(X)
    return X

def cholesky_rep(X):
    """Find a rotation of X which is on the Cholesky manifold.

    Parameters
    ----------
    X : ndarray
        Array of data as an n*d matrix.

    Returns
    -------
    C : ndarray
        Rotation of the data which lives on the Cholesky manifold.

    Notes
    -----
    The Cholesky manifold is represented by n*d orthonormal matrices
    with n > d (i.e. the norm of each row is 1) and where the top d*d
    submatrix is lower-triangular. Moreover, the first row is always
    chosen to be [1,0,...,0]. This function finds a rotation matrix Q
    such that right-multiplication by Q turns the upper square of the
    input matrix into the correct form. It does not verify the
    orthonormality condition.
    
    """

    d = X.shape[1]
    Xd = X[0:d,0:d]
    if LA.matrix_rank(Xd) < d:
        raise LA.LinAlgError('Input data not generic. If computing using ' +
            'circleRPn, check that dim==num_segments.')
    # Get an appropriate cholesky matrix to start lrcm_min.
    L = LA.cholesky(Xd@Xd.T)
    Q = LA.solve(Xd,L)
    C = np.tril(X@Q)    # The matrix is lower-triangular, but apply tril
    return C            # to handle floating-point errors.

def geo_distance_matrix(D,epsilon=0.4,k=-1,normalize=True):
    """Approximate a geodesic distance matrix.

    Given a distance matrix uses either an epsilon neighborhood or a
    k-NN algorithm to find nearby points, then builds a distance matrix
    such that nearby points have their ambient distance as defined by
    the original distance matrix, while far away points are given the
    shortest path distance in the graph.
   
    Parameters
    ----------
    data : ndarray
        Data as an n*2 matrix, assumed to lie on RP^n (i.e. S^n).
    epsilon : float, optional
        Radius of neighborhood when constructing graph. Default is ~pi/8.
    k : int, optional
        Number of nearest neighbors in k-NN graph. Default is -1 (i.e.
        use epsilon neighborhoods).

    Returns
    -------
    Dhat : ndarray
        Square distance matrix matrix of the graph. Distances are
        normalized to correspond to RP^n, i.e. Dhat is scaled so the
        maximum distance is no larger than pi/2.

    Raises
    ------
    ValueError
        If the provided value of epsilon or k is too small, the graph
        may not be connected, giving infinite values in the distance
        matrix. A value error is raised if this occurs, as the later
        algorithms do not handle infinite values smoothly.

    """

    # Use kNN. Sort twice to get nearest neighbour list.
    if k > 0:
        D_sort = np.argsort(np.argsort(D))
        A = D_sort <= k
        A = (A + A.T)/2
    # Use epsilon neighborhoods.
    else:
        A = D<epsilon
    G = csr_matrix(D*A)                   # Matrix representation of graph
    Dg = floyd_warshall(G,directed=False)     # Path-length distance matrix
    if np.isinf(np.max(Dg)):
        raise ValueError('The distance matrix contains infinite values, ' +
            'indicating that the graph is not connected. Try a larger value ' +
            'of epsilon or k.')
    Dhat = (np.max(D)/np.max(Dg))*Dg    # Normalize distances.
    return Dhat


def graph_distance_matrix(data,epsilon=0.4,k=-1):
    """Construct a geodesic distance matrix from data in RP^n.

    Given a point cloud of data in RP^n, uses either an epsilon
    neighborhood or a k-NN algorithm to find nearby points, then builds
    a distance matrix such that nearby points have their ambient
    distance, while far away points are given the shortest path distance
    in the graph.
    
    Parameters
    ----------
    D : ndarray (n*n)
        Distance matrix to convert to an approximation of the geodesic
        distance matrix.
    epsilon : float, optional
        Radius of neighborhood when constructing graph. Default is ~pi/8.
    k : int, optional
        Number of nearest neighbors in k-NN graph. Default is -1 (i.e.
        use epsilon neighborhoods).
    normalize : bool, optional 
        Normalize the output distance matrix `Dhat` so that the maximum
        distance is the same as in the original. Default is True.

    Returns
    -------
    Dhat : ndarray
        Square distance matrix matrix of the graph. Distances are
        normalized to correspond to RP^n, i.e. Dhat is scaled so the
        maximum distance is no larger than pi/2.

    Raises
    ------
    ValueError
        If the provided value of epsilon or k is too small, the graph
        may not be connected, giving infinite values in the distance
        matrix. A value error is raised if this occurs, as the later
        algorithms do not handle infinite values smoothly.

    """

    # Use kNN. Sort twice to get nearest neighbour list.
    if k > 0:
        D_sort = np.argsort(np.argsort(D))
        A = D_sort <= k
        A = (A + A.T)/2
    # Use epsilon neighborhoods.
    else:
        A = D<epsilon
    G = csr_matrix(D*A)                   # Matrix representation of graph
    Dg = floyd_warshall(G,directed=False)     # Path-length distance matrix
    if np.isinf(np.max(Dg)):
        raise ValueError('The distance matrix contains infinite values, ' +
            'indicating that the graph is not connected. Try a larger value ' +
            'of epsilon or k.')
    Dhat = (np.max(D)/np.max(Dg))*Dg    # Normalize distances.
    return Dhat

def pmds(Y,D,max_iter=20,verbosity=1,autograd=False,pmo_solve='cg'):
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
    rank = LA.matrix_rank(Y)
    if verbosity > 0:
        print('Finding projection onto RP^%i.' %(rank-1))
    W = distance_to_weights(D)
    S = np.sign(Y@Y.T)
    C = S*np.cos(D)
    if np.sum(S == 0) > 0:
        print('Warning: Some initial guess vectors are orthogonal, this may ' +
            'cause issues with convergence.')
    cost = setup_cost(D,S)
    cost_list = [cost(Y.T)]
    true_cost = setup_cost(projective_distance_matrix(Y),S)
    true_cost_list = [true_cost(Y.T)]
    manifold = Oblique(rank,num_points) # Short, wide matrices.
    solver = ConjugateGradient()
# TODO: play with alternate solve methods and manifolds.
#   if pmo_solve == 'nm':
#       solver = NelderMead()
#   if pmo_solve == 'ps':
#       solver = ParticleSwarm()
#   if pmo_solve == 'tr':
#       solver = TrustRegions()
#   if pmo_solve == 'sd':
#       solver = SteepestDescent()
#   else:
#       solver = ConjugateGradient()
#       if solve_prog == 'matlab':
# TODO: this may generate errors based on changes to other methods.
#           cost, egrad, ehess = setup_cost(C,W)
#           workspace = lrcm_wrapper(C,W,Y)
#           Y_new = workspace['optimal_matrix']
    for i in range(0,max_iter):
        if autograd:
            cost = setup_cost(D,S)
            problem = pymanopt.Problem(manifold, cost, verbosity=verbosity)
        else:
            cost, egrad, ehess = setup_cost(D,S,return_derivatives=True)
            problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess, verbosity=verbosity)
        if pmo_solve == 'cg' or pmo_solve == 'sd' or pmo_solve == 'tr':
            # Use initial condition with gradient-based solvers.
            Y_new = solver.solve(problem,x=Y.T)
        else:
            Y_new =  solver.solve(problem)
        Y_new = Y_new.T     # Y should be tall-skinny
        cost_oldS = cost(Y_new.T)
        cost_list.append(cost_oldS)
        S_new = np.sign(Y_new@Y_new.T)
        C_new = S_new*np.cos(D)
        cost_new = setup_cost(D,S_new)
        cost_newS = cost_new(Y_new.T)
        S_diff = ((LA.norm(S_new - S))**2)/4
        percent_S_diff = 100*S_diff/S_new.size
        percent_cost_diff = 100*(cost_list[i] - cost_list[i+1])/cost_list[i]
        true_cost = setup_cost(projective_distance_matrix(Y),S)
        true_cost_list.append(true_cost(Y_new.T))
        # Do an SVD to get the correlation matrix on the sphere.
        # Y,s,vh = LA.svd(out_matrix,full_matrices=False)
        if verbosity > 0:
            print('Through %i iterations:' %(i+1))
            print('\tTrue cost: %2.2f' %true_cost(Y_new.T))
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
        Y = Y_new
        C = C_new
        S = S_new
    return Y, cost_list, true_cost_list

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
# Miscellaneous
###############################################################################

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D."""
    W_inv = (1 - np.cos(D)**2)     
    W = np.sqrt((W_inv+np.eye(D.shape[0],D.shape[1]))**-1 - np.eye(D.shape[0],D.shape[1]))
    return W

def projective_distance_matrix(Y):
    """Construct the (exact) distance matrix of data Y on RP^d."""
    S = np.sign(Y@Y.T)
    M = S*(Y@Y.T)
    acos_validate(M)
    D = np.arccos(M)    # Initial distance matrix
    return D

###############################################################################
# Tools for external library interfaces
###############################################################################

def lrcm_wrapper(C,W,Y0):
    """Output to Grubisic and Pietersz LRCM matlab code."""
    io.savemat('ml_tmp.mat', dict(C=C,W=W,Y0=Y0))
    eng = matlab.engine.start_matlab()
    t = eng.lrcm_wrapper()
    workspace = io.loadmat('py_tmp.mat')
    return workspace

def setup_cost(D,S,return_derivatives=False):
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
    D : ndarray
        Cost matrix
    S : ndarray
        Sign matrix

    Returns
    -------
    cost : function
        Weighted Frobenius norm cost function.
    grad : function, optional
        Gradient of cost function.
    hess : function, optional
        Hessian of cost function.

    """

    W = distance_to_weights(D)
    C = S*np.cos(D)
    def F(Y):
        """Weighted Frobenius norm cost function."""
        return 0.5*np.linalg.norm(W*(C-Y.T@Y))**2
    def dF(Y):
        """Derivative of weighted Frobenius norm cost."""
        return 2*Y@(W**2*(Y.T@Y-C))
    def ddF(Y,H):
        """Second derivative (Hessian) of weighted Frobenius norm cost."""
        return 2*((W**2*(Y.T@Y-C))@H + (W**2*(Y@H.T + H@Y.T))@Y)
    return F, dF, ddF

###############################################################################
# Old Stuff
###############################################################################

#def setup_ag_cost(S,D):
#    """Setup the cost function for autograd and pymanopt.
#
#    Pymanopt can compute derivatives automatically using autograd. This
#    function provides different approximations of the objective function
#        F(Y) = 1/2*||D - acos(abs(Y.T@Y))||^2
#    for use with autograd.
#    * 'none' supplies the exact formula for F, which performs badly
#        because abs is non-differentiably and acos is tricky.
#    * 'taylor' uses a cubic-order Taylor series approximation for acos.
#    * 'rational' uses a quartic rational approximation for acos.
#    * 'frobenius' uses a mean-value theorem argument to rewrite F as a
#        weighted frobenius norm and supplies that function.
#
#    Parameters
#    ----------
#    S : ndarray (square, 1 or -1)
#        Signs of entries in input matrix Y.T@Y. Used to avoid absolute
#        values.
#    D : ndarray (square, symmetric, positive)
#        Distance matrix for objective function.
#    appx: string, {'none','taylor','rational','frobenius'}
#        Type of approximation to use. See details above. (Deprecated.)
#
#    Returns
#    -------
#    F(Y) : function
#        Autograd compatible cost function.
#
#    """
#
#    if appx == 'none':
#        """Direct cost function with no approximations."""
#        def F(Y):
#            YY = acos_validate(Y.T@Y)
#            return 0.5*np.linalg.norm(np.arccos(S*(YY)) - D)**2
#    elif appx == 'taylor':
#        # Taylor-series bases polynomial cost.
#        def F(Y):
#            return 0.5*np.linalg.norm(S*(np.pi/2 - D) - Y.T@Y - .1667*(Y.T@Y)**3)**2
#    elif appx == 'rational':
#        # Rational function approximation.
#        a = -0.939115566365855
#        b =  0.9217841528914573
#        c = -1.2845906244690837
#        d =  0.295624144969963174
#        def F(Y):
#            return 0.5*np.linalg.norm(S*D - S*np.pi/2 - (a*(Y.T@Y) + b*(Y.T@Y)**3)/(1 + c*(Y.T@Y)**2 + d*(Y.T@Y)**4))**2
#    elif appx == 'frobenius':
#   W = distance_to_weights(D)
#   def F(Y):
#       return 0.5*np.linalg.norm(W*(S*np.cos(D)-Y.T@Y))**2
#   return F

