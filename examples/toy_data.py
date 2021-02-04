# Simple script for generating toy data. This is not meant to illustrate
# anything, just to have something I can run through the algorithm to
# make sure the different components work correctly.
import numpy as np
import cplx_projective
from ppca import ppca

# Create some random points on the 7-sphere in R^8.
X = np.random.rand(8, 5) - 0.5
X = X/np.linalg.norm(X, axis=0)

# Construct the distance matrix using these points to represent
# equivalence classes in CP^3. The goal will be to recover these
# distances with the points in CP^1.
D = cplx_projective.CPn_distance_matrix(X)

# Build an initial guess of a low-dimensional version
X_ppca = ppca(X.T, 3)['X'].T
D_ppca = cplx_projective.CPn_distance_matrix(X_ppca)
np.savez('examples/toy_data.npz', X=X, D=D, X_ppca=X_ppca, D_ppca=D_ppca)
