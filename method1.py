# Hacked together script to do dim reduction on RPn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

# Make some random data. Ultimately replace with something that imports
# interesting data for testing.
X = np.random.rand(3,20)
X = X/np.linalg.norm(X,axis=0)
D = np.arccos(np.abs(X.transpose()@X))
# Select neighborhoods using a radius (alt use k-NN)
epsilon = 0.5
A = D<epsilon
G = csr_matrix(D*A)   # Matrix representation of graph
Dg = floyd_warshall(G,directed=False)
# Implement a check here that the graph is actually connected (otherwise there 
# are infinite values in the distance matrix).
if np.isinf(np.max(Dg)):
    print("""Try a bigger value of epsilon - the distance matrix is not
        connected.""")
Dhat = (np.max(D)/np.max(Dg))*Dg
W = (1 - np.cos(Dhat)**2)**-1
W = np.nan_to_num(W,posinf=0)   # W has inf on diagonal.


