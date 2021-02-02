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

#-----------------------------------------------------------------------------#
# Functions copied from Joe which give the geometry of the flat Klein
# bottle.

def cls(x,y):
    '''
    Outputs the equivalence class of the points represented by (x,y) in
    the fundamental domain of K.
    '''
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

# Construct a uniform grid of points in the unit square.

numx = 20
numy = 20
N = numx*numy   # Total number of points
filename = 'flat_klein_bottle_N%d.npz' %N
print('Generating %d points on the flat klein bottle and saving the '\
    'results to ' %N + filename + '.' )
n_landmarks = 100

x, y = np.meshgrid(np.linspace(0,1,numx),np.linspace(0,1,numy))
xy = np.column_stack((x.ravel(),y.ravel()))

# Construct the distance matrix of the points.

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
cocycles = PH_sub['cocycles'][1]
diagram = PH_sub['dgms'][1]
eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram,
    threshold_at_death=False)

# Get a partition of unity.
part_func = pipeline.partition_unity(D, (death-birth)/2, sub_ind)

# Map the data into RP^k with k=n_landmarks.
# TODO: this isn't quite right, because lens_coordinates is setup for complex
# space, and realifying that is double the proper dimension.
proj_coords = pipeline.lens_coordinates(part_func, eta, 2)
proj_coords = cplx_projective.realify(proj_coords)

print(real_projective.RPn_validate(proj_coords))

# TODO:
D_pc = real_projective.projective_distance_matrix(proj_coords.T)
D_geo = real_projective.geo_distance_matrix(D_pc, k=5)
# Compute PH of landmarks of high-dimensional data.
PH_pc = ripser(D_geo[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2)
# Apply PPCA to reduce to dimension 2.
X_ppca = pipeline.ppca(proj_coords, 2)
# Compute persistence of PCA output.
D_ppca = real_projective.projective_distance_matrix(X_ppca)
PH_ppca = ripser(D_ppca[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2)
# Apply MDS to PCA output.
X_mds = real_projective.pmds(X_ppca, D_geo)
# Compute persistence of MDS output.
D_mds = real_projective.projective_distance_matrix(X_mds)
PH_mds = ripser(D_mds, distance_matrix=True, maxdim=2)
# Plot MDS and PCA outputs.


# Save the data.
np.savez(filename, xy=xy, xy_sub = xy_sub, D=D, D_sub=D_sub, PH_sub=PH_sub,
    proj_coords=proj_coords)

print(proj_coords.shape)
