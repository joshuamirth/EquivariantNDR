# Hacked together script to do dim reduction on RPn
import matlab.engine    # for LRCM MIN.
import numpy as np
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
# This will only work in the folder containing dreimac.
# from dreimac.projectivecoords import ppca

def circle_RPn(n=4,num_points=100,kinks=2)
    """Construct points on a "kinked" circle in RP^n.

    """

    theta_size = np.floor(num_points/kinks)
    if theta_size*kinks != num_points:
        print('Warning: data will have ' + str(theta_size*kinks) +
            ' points, not' + str(num_points))
    theta = np.linspace(0,np.pi/2,num_points/kinks)
    X = zeros((num_points,n+1))
    for i in range(0:kinks):
        X[i:theta_size+i,i:i+1] = [np.cos(theta),np.sin(theta)]
    return X
    

def initial_guess(data,dim):
    """Compute an initial guess for lrcm_min using projective pca."""
    # Get projective coordinates in reduced dimension.
    proj = ProjectiveCoords(
        data,
        n_landmarks = 100
    )
    res = proj.get_coordinates(proj_dim = dim, perc=0.9)
    Y = res['X']
#   Y = np.nan_to_num(Y)    # Apparently possible to have some values slightly out of range.
    # Get an appropriate cholesky matrix to start lrcm_min.
    YY = Y@Y.transpose()
    YYd = YY[0:dim,0:dim]
    print(np.linalg.matrix_rank(YYd))
    R = np.linalg.cholesky(YYd) # Only compute cholesky of the upper corner.
    Q = np.linalg.solve(YYd,Rd)
    Y = Y@Q
    return Y

def sphere_toy(num_points,dim,embedding_dim):
    """Get some points on the sphere of dimension dim and map up to
    sphere of dimension embedding_dim."""
    X = np.random.rand(num_points,dim)
    S = (X.transpose()/np.linalg.norm(X,axis=1)).transpose()
    S_high = np.zeros((num_points,embedding_dim))
    S_high[:,0:dim] = S     # About as trivial an embedding as possible.
    return S_high

# dim = 8
num_points = 10
goal_dim = 2
# X = get_line_patches(dim,10,10,0.25) # sample 100 points from line patches data.
X = sphere_toy(num_points,goal_dim,goal_dim)
Y0 = sphere_toy(num_points,goal_dim,goal_dim)
#Y0 = initial_guess(X,2) # Get an initial guess in RP^2.
#proj = ProjectiveCoords(X,n_landmarks=100)
#res = proj.get_coordinates(proj_dim = dim**2-1, perc=0.9) # Coordinates in RP^n w/o reduction.
#X = res['X']
M = np.abs(X@X.transpose())
# Due to rounding errors, M may contain values 1+epsilon. Remove these.
bad_vals = M > 0.999999
M[bad_vals] = 0.999999
D = np.arccos(M)    # Initial distance matrix
# Select neighborhoods using a radius (alt use k-NN)
epsilon = 1.0
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

S = np.sign(Y0@Y0.transpose())
C = S*np.cos(Dhat)  # Cost matrix, in terminology of G&P.

# TODO: make a loop, updating cost function with appropriate signs.
# This might be super slow for now because of the repeated calls to matlab.

# Switch over to matlab to call lrcm_min.
print('Starting MATLAB ====================================================')
io.savemat('ml_tmp.mat', dict(C=C,W=W,Y0=Y0))
eng = matlab.engine.start_matlab()
t = eng.lrcm_wrapper()
workspace = io.loadmat('py_tmp.mat')
out_matrix = workspace['optimal_matrix']
print('MATLAB complete ====================================================')

C = out_matrix@out_matrix.transpose()
print(C)
print(M)
