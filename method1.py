# Hacked together script to do dim reduction on RPn
import matlab.engine    # for LRCM MIN.
import numpy as np
import scipy.io as io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
# This will only work in the folder containing dreimac.
# from dreimac.projectivecoords import ppca

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
