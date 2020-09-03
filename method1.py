# Hacked together script to do dim reduction on RPn
import matlab.engine    # for LRCM MIN.
import numpy as np
# import DREiMac
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

# Make some random data. Ultimately replace with something that imports
# interesting data for testing.
X = np.random.rand(3,20)
data = X/np.linalg.norm(X,axis=0)
M = np.abs(data.transpose()@data)
# Due to rounding errors, M may contain values 1+epsilon. Remove these.
bad_vals = M > 0.999999
M[bad_vals] = 0.999999
D = np.arccos(M)    # Initial distance matrix
# Select neighborhoods using a radius (alt use k-NN)
epsilon = 0.5
A = D<epsilon
G = csr_matrix(D*A)                         # Matrix representation of graph
Dg = floyd_warshall(G,directed=False)       # Path-length distance matrix
# Implement a check here that the graph is actually connected (otherwise there 
# are infinite values in the distance matrix).
if np.isinf(np.max(Dg)):
    print("""Try a bigger value of epsilon - the distance matrix is not
        connected.""")
Dhat = (np.max(D)/np.max(Dg))*Dg    # Normalize so distances are reasonable.
W_inv = (1 - np.cos(Dhat)**2)       # Pointwise inverse of weight matrix.
W = (W_inv+np.eye(20))**-1 - np.eye(20) # Put zeros on the diagonal w/o dividing by zero.
output_rank = 2                    # Desired output dimension.
#S = # initial weights
# Call the matlab routine for optimization:
# TODO: make this a loop, updating cost function with appropriate signs.
# This might be super slow for now because of the repeated calls to matlab.
#eng = matlab.engine.start_matlab()
#out = eng.lrcm_wrapper(W,cost,d)


