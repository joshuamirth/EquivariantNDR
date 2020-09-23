""" Dim reduction on RPn using an MDS-type method. """
import matlab.engine    # for LRCM MIN.
import numpy as np
import numpy.linalg as LA
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# dreimac does not install properly on my system.
try:
    from dreimac.projectivecoords import ppca
except:
    from ppca import ppca
    print("""Loading personal version of PPCA. This may not be consistent with
        the published version""")

def circleDM(n=50,diam=1.57):
    """Distance matrix for 2n evenly spaced circle points."""
    theta = np.hstack((
        np.linspace(0,diam,n,endpoint=False),
        np.linspace(diam,0,n,endpoint=False)
    ))
    D = np.array([np.roll(theta,i) for i in range(0,2*n)])
    return D
    

def circleRPn(
    dimn=4,
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

    It is recommended that dimn==num_segments, otherwise the resulting
    data matrix will not be full rank, which can cause issues later.
    Similarly, the output data is randomly permuted so that the first n
    points are not on the same linear subspace, generically.

    Parameters
    ----------
    dimn : int, optional
        Dimension of RP^d to work on (ambient euclidean space is dimn+1).
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
    if num_segments < 1 or dimn < 1:
        raise ValueError("""Number of segments and dimension must be positive
            integers. Supplied values were num_segments = %2.2f and dimension
            = %2.2f""" %(num_segments,dimn))    
    if dimn < num_segments:
        raise ValueError("""Value of dimension must be larger than number of
            segments. Supplied dimension was %i and number of segments was
            %i""" %(dimn,num_segments))
    num_points = segment_points*(num_segments+1)
    theta = np.linspace(0,np.pi/2,segment_points,endpoint=False)
    X = np.zeros((num_points,dimn+1))
    segment_curve = np.array([np.cos(theta),np.sin(theta)]).transpose()
    for i in range(0,num_segments):
        X[i*segment_points:(i+1)*segment_points,i:i+2] = segment_curve
    X[num_segments*segment_points:num_points,0] = -np.sin(theta)
    X[num_segments*segment_points:num_points,num_segments] = np.cos(theta)
    if randomize:
        X = np.random.permutation(X)
    if noise:
        N = v*np.random.rand(dimn+1,num_points)
        Xt = (X.transpose() + N)/LA.norm(X.transpose()+N,axis=0)
        X = Xt.transpose()
    return X, num_points

def initial_guess(data,dim):
    """Use PPCA and Cholesky factor to get an initial input for lrcm_min

    Parameters
    ----------
    data : ndarray
        Input data as n*d matrix
    dim : int
        Dimension (of projective space) on which to place an initial
        guess. Must be less than the dimension of the original data.

    Returns
    -------
    X : ndarray
        Data on the Cholesky manifold.
    
    Notes
    -----
    The Cholesky manifold is defined in Grubisic and Pietersz.
    
    """
    
    # TODO: consider other ways of getting an initial low-rank guess,
    # e.g. iteratively doing ppca, or another method.
    V = ppca(data, dim)
    X = V['X']
    # Testing out a completely random initial condition.
#   X = 2*np.random.random_sample((dim+1,data.shape[0]))-1
#   X = X/LA.norm(X,axis=0)
#   X = X.transpose()
    X = cholesky_rep(X)
    return X

# This is probably junk. Goal is some way of getting initial condition
# w/o original data being in RP^N.
def initial_guess_DM(D,dim):
    U,S,V = LA.svd(D)
    Uu = U[:,0:dim+1]
    nrm = LA.norm(Uu,axis=1)
    Uu = (Uu.transpose()/nrm).transpose()
    sgn = np.sign(np.random.rand(Uu.shape[0]-.5))
    sgn = sgn.reshape(Uu.shape[0],1)
    Uu = Uu*sgn
    return Uu

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
        raise LA.LinAlgError("""Input data not generic. If computing
            using circleRPn, check that dimn==num_segments.""")
    # Get an appropriate cholesky matrix to start lrcm_min.
    L = LA.cholesky(Xd@Xd.transpose())
    Q = LA.solve(Xd,L)
    C = np.tril(X@Q)    # The matrix is lower-triangular, but apply tril
    return C            # to handle floating-point errors.

def graph_distance_matrix(data,epsilon=0.4,k=-1):
    """Construct a geodesic distance matrix from data in RP^n.

    Given a point cloud of data in RP^n, uses either an epsilon
    neighborhood or a k-NN algorithm to find nearby points, then builds
    a distance matrix such that nearby points have their ambient
    distance, while far away points are given the shortest path distance
    in the graph.
    
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

    M = np.abs(data@data.transpose())
    # Due to rounding errors, M may contain values 1+epsilon. Remove these.
    bad_vals = M >= 1.0
    M[bad_vals] = 1.0
    D = np.arccos(M)    # Initial distance matrix
    # Use kNN. Sort twice to get nearest neighbour list.
    if k > 0:
        D_sort = np.argsort(np.argsort(D))
        A = D_sort <= k
    # Use epsilon neighborhoods.
    else:
        A = D<epsilon
    G = csr_matrix(D*A)                   # Matrix representation of graph
    Dg = floyd_warshall(G,directed=False)     # Path-length distance matrix
    if np.isinf(np.max(Dg)):
        raise ValueError("""The distance contains infinite values, indicating
            that the graph is not connected. Try a larger value of epsilon or
            k.""")
    Dhat = (np.max(D)/np.max(Dg))*Dg    # Normalize distances.
    return Dhat

def pmds(Y,D,max_iter=20,verbose=True):
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
    verbose: bool, optional
        If true, print output relating to convergence conditions at each
        iteration.

    Returns
    -------

    """

    # Put zeros on the diagonal of W w/o dividing by zero.
    num_points = Y.shape[0]
    W_inv = (1 - np.cos(D)**2)     
    W = np.sqrt((W_inv+np.eye(num_points))**-1 - np.eye(num_points))
    Y_prev = Y
    S = np.sign(Y@Y.transpose())
    C = S*np.cos(D)
    cost = []
    for i in range(0,max_iter):
        # Actual algorithmic loop:
        workspace = lrcm_wrapper(C,W,Y)
        # Updated information:
        Y_new = workspace['optimal_matrix']
        S_new = np.sign(Y_new@Y_new.transpose())
        Fopt = workspace['Fopt']
        cost.append(Fopt)
        C_new = S_new*np.cos(D)
        cost_newS = 0.5*LA.norm(W*(Y_new@Y_new.transpose()) - W*C_new)**2 
        S_diff = ((LA.norm(S_new - S))**2)/4
        percent_S_diff = 100*S_diff/S_new.size
        if i > 0:
            percent_cost_diff = 100*(cost[i-1] - cost[i])/cost[i-1]
        else:
            percent_cost_diff = 100
        # (This is not actually necessary - not doing so gives us "mean
        # centered" data.)
        # Do an SVD to get the correlation matrix on the sphere.
        # Y,s,vh = LA.svd(out_matrix,full_matrices=False)
        if verbose:
            print('Through %i iterations:' %(i+1))
            print('\tComputed cost: %2.2f' %cost[i])
            print('\tPercent cost difference: % 2.2f' %percent_cost_diff)
            print('\tPercent Difference in S: % 2.2f' %percent_S_diff)
            print('\tComputed cost with new S: %2.2f' %cost_newS)
            print('\tDifference in cost matrix: %2.2f' %(LA.norm(C-C_new)))
#           print('\tPercent of orthogonal elements: %2.2f' %ortho_ips)
#           print('\tPercent of negligible cost terms: %2.2f' %neg_cost)
        if S_diff < 1:
            print('No change in S matrix. Stopping iterations.')
            break
        if percent_cost_diff < .0001:
            print('No significant cost improvement. Stopping iterations.')
            break
        # Update variables:
        Y_prev = Y
        Y = Y_new
        C = C_new
        S = S_new
    return Y, cost, Y_prev

def lrcm_wrapper(C,W,Y0):
#   print('Starting MATLAB ==================================================')
    io.savemat('ml_tmp.mat', dict(C=C,W=W,Y0=Y0))
    eng = matlab.engine.start_matlab()
    t = eng.lrcm_wrapper()
    workspace = io.loadmat('py_tmp.mat')
#   out_matrix = workspace['optimal_matrix']
#   cost = workspace['Fopt']
#   print('Cost at this iteration ' + str(Fopt[0][0]))
#   print('MATLAB complete ==================================================')
    return workspace
    
def plot_RP2(X,pullback=True):
    """Plot data reduced onto RP2"""

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2])
    if pullback:
        Y = -X
        ax.scatter(Y[:,0],Y[:,1],Y[:,2])
    plt.show()
