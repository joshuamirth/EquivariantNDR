# Hacked together script to do dim reduction on RPn
import matlab.engine    # for LRCM MIN.
import numpy as np
import numpy.linalg as LA
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import matplotlib.pyplot as plt

# dreimac does not install properly on my system.
try:
    from dreimac.projectivecoords import ppca
except:
    import ppca
    print("""Loading personal version of PPCA. This may not be consistent with
        the published version""")

def circleRPn(dimn=4,segment_points=50,num_segments=2):
    """Construct points on a "kinked" circle in RP^d.

    Constructs a curve of evenly-spaced points along the great circle
    from e_i to e_{i+1} in R^{d+1}, starting at e_0 and finishing at
    e_i with i = num_segments, then returns to -e_0.

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
    """
    
    if dimn < num_segments:
        raise ValueError('Dimension ' + str(dimn) + ' does not have enough ' +
            'dimensions for ' + str(num_segments) + ' segments.')
    num_points = segment_points*(num_segments+1)
    theta = np.linspace(0,np.pi/2,segment_points,endpoint=False)
    X = np.zeros((num_points,dimn+1))
    segment_curve = np.array([np.cos(theta),np.sin(theta)]).transpose()
    for i in range(0,num_segments):
        X[i*segment_points:(i+1)*segment_points,i:i+2] = segment_curve
    X[num_segments*segment_points:num_points,0] = -np.sin(theta)
    X[num_segments*segment_points:num_points,num_segments] = np.cos(theta)
    return X

def initial_guess(data,dim):
    """Use PPCA and Cholesky factor to get an initial input for lrcm_min"""
    # Get projective coordinates in reduced dimension.
    V = ppca.ppca(data, dim)
    X = V['X']
#   X = np.nan_to_num(X)    # Apparently possible to have some values slightly out of range.
    # Get an appropriate cholesky matrix to start lrcm_min.
    XX = X@X.transpose()
    XXd = XX[0:dim+1,0:dim+1]
    R = LA.cholesky(XXd) # Only compute cholesky of the upper corner.
    Q = LA.solve(XXd,R)
    X = X@Q
    return X

def pmds(data,goal_dim,epsilon=1.0,num_iter=20,verbose=True):
    """Projective multi-dimensional scaling algorithm.

    Detailed description in career grant, pages 6-7 (method 1).

    Parameters
    ----------
    data : ndarray
        Data as an n*2 matrix, assumed to lie on RP^n (i.e. S^n).
    goal_dim : int
        Output data will lie on RP^d with d = goal_dim.
    epsilon : float, optional
        Radius of neighborhood when constructing graph.
    num_iter : int, optional
        Number of times to iterate the loop. Will eventually be updated
        to a better convergence criterion.

    Returns
    -------

    """

    num_points = data.shape[0]
    M = np.abs(data@data.transpose())
    # Due to rounding errors, M may contain values 1+epsilon. Remove these.
    bad_vals = M > 0.999999
    M[bad_vals] = 0.999999
    D = np.arccos(M)    # Initial distance matrix
    A = D<epsilon
    G = csr_matrix(D*A)                         # Matrix representation of graph
    Dg = floyd_warshall(G,directed=False)       # Path-length distance matrix
    # Check that the graph is actually connected (otherwise there are infinite
    # values in the distance matrix).
    if np.isinf(np.max(Dg)):
        print("""Try a bigger value of epsilon - the distance matrix is not
            connected.""")
    Dhat = (np.max(D)/np.max(Dg))*Dg    # Normalize so distances are reasonable.
    W_inv = (1 - np.cos(Dhat)**2)       # Pointwise inverse of weight matrix.
    W = (W_inv+np.eye(num_points))**-1 - np.eye(num_points) # Put zeros on the diagonal w/o dividing by zero.
    # TODO: make this a loop, and not just a one-time iteration.
    Y = initial_guess(data,goal_dim)
    cost_list = []
    old_cost = np.inf 
    for i in range(0,num_iter):
        S = np.sign(Y@Y.transpose())
        C = S*np.cos(Dhat)  # Cost matrix, in terminology of G&P.
        workspace = lrcm_wrapper(C,W,Y)
        #print('Cost at this iteration: ' + str(cost))
        out_matrix = workspace['optimal_matrix']
        Fopt = workspace['Fopt']
        # Do an SVD to get the correlation matrix on the sphere.
        Y,s,vh = LA.svd(out_matrix,full_matrices=False)
        Sn = np.sign(Y@Y.transpose())
        Sdiff = 25*((LA.norm(Sn - S))**2)/(num_points**2)
        cost = LA.norm(np.sqrt(W)*(Y@Y.transpose()) - np.sqrt(W)*C)**2
        new_s_cost = LA.norm(np.sqrt(W)*(Y@Y.transpose()) -
            np.sqrt(W)*Sn*np.cos(Dhat))**2 
        percent_cost_diff = 100*(cost - old_cost)/cost
        if verbose:
            print('Through %i iterations:' %(i+1))
            print('\tComputed cost: %i' %(int(cost)))
            print('\tComputed cost with new S: %i' %(int(new_s_cost)))
            print('\tPercent cost difference: % 2.2f' %percent_cost_diff)
            print('\tPercent Difference in S: % 2.2f' %Sdiff)
            print('\tFopt (I don\'t know what this means): %i' %(int(Fopt)))
        if Sdiff < 1:
            print('No change in S matrix. Stopping iterations.')
            break
        if percent_cost_diff > -.0001:
            print('No significant cost improvement. Stopping iterations.')
            break
        old_cost = cost
    return Y

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
