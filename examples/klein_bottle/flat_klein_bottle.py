# Script for constructing data on the flat model of the Klein bottle.
# Modified from code provided by Joe Melby.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import real_projective
import cplx_projective
from examples import pipeline   # Note: to make this work, install
            # package in editable mode, `pip install -e .` in root directory.
from scipy.spatial.distance import pdist    # For some reason I have to
                                            # import this, instead of just
                                            # running it?
from numpy.random import default_rng
from ppca import ppca

#-----------------------------------------------------------------------------#
# Functions copied from Joe which give the geometry of the flat Klein
# bottle.
# TODO: are these wrong? Why do I not get the homology of the Klein bottle?

def cls(x,y):
    '''
    Outputs the equivalence class of the points represented by (x,y) in
    the fundamental domain of K.
    '''
#   arr = np.array([[x,y],[x-1,y],[x+1,y],[x,y-1],[x,y+1],[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1]])
    arr = np.array([[x,y],[x-1,2-y],[x,y+1],[x+1,2-y],[x-1,1-y],[x+1,1-y],[x-1,-y],[x,y-1],[x+1,-y]])
    return arr

def minDist(X,Y):
    '''
    Returns the geodesic distance on the fundamental domain between X
    and Y.
    '''
    md = np.min(sp.spatial.distance_matrix(cls(X[0],X[1]),cls(Y[0],Y[1])))
    return md

#-----------------------------------------------------------------------------#

# Construct a uniformly random grid of points in the unit square.
rng = default_rng(57)
numx = 20
numy = 20
N = numx*numy   # Total number of points
filename = 'flat_klein_bottle_N%d.npz' %N
print('Generating %d points on the flat klein bottle.' %N)
n_landmarks = 100

x, y = np.meshgrid(rng.random(numx),rng.random(numy))
xy = np.column_stack((x.ravel(),y.ravel()))

# Construct the distance matrix of the points.
print('Computing distance matrix and landmark subset.')
D = pdist(xy, minDist)
D = sp.spatial.distance.squareform(D)
# Subsample the distance matrix with max/min.
sub_ind = pipeline.maxmin_subsample_distance_matrix(D,
    n_landmarks)['indices']
D_sub = D[sub_ind, :][:, sub_ind]
xy_sub = xy[sub_ind,:]

print('Computing persistence of the landmarks.')
PH_sub = ripser(D_sub, coeff=2, do_cocycles=True, maxdim=2,
    distance_matrix=True)
#plot_diagrams(PH_sub['dgms'])
#plt.show()

# Get a prominent cocycle in dimension one.
print('Computing projective coordinates in dimension %d.' %len(sub_ind))
cocycles = PH_sub['cocycles'][1]
diagram = PH_sub['dgms'][1]
eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram,
    threshold_at_death=False)

# Get a partition of unity.
part_func = pipeline.partition_unity(D, .115, sub_ind)
proj_coords = pipeline.proj_coordinates(part_func, eta)

print(real_projective.RPn_validate(proj_coords))

D_pc = real_projective.projective_distance_matrix(proj_coords.T)
D_geo = real_projective.geo_distance_matrix(D_pc, k=8)
# Compute PH of landmarks of high-dimensional data.
PH_pc = ripser(D_geo[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_pc['dgms'])
plt.show()
# Apply PPCA to reduce to dimension 2.
X_ppca = ppca(proj_coords.T, 2)['X']
# Compute persistence of PCA output.
D_ppca = real_projective.projective_distance_matrix(X_ppca)
PH_ppca = ripser(D_ppca[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2)
plot_diagrams(PH_ppca['dgms'])
plt.show()
# Apply MDS to PCA output.
X_mds = real_projective.pmds(X_ppca, D_geo)
# Compute persistence of MDS output.
D_mds = real_projective.projective_distance_matrix(X_mds)
PH_mds = ripser(D_mds, distance_matrix=True, maxdim=2)
# Plot MDS and PCA outputs.


# Save the data.
#np.savez(filename, xy=xy, xy_sub = xy_sub, D=D, D_sub=D_sub, PH_sub=PH_sub,
#    proj_coords=proj_coords)

