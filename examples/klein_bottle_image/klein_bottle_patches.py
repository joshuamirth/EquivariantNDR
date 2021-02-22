# Script for constructing the data set of image patches realizing Klein bottle.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import geodesic_metric
from examples import pipeline   # Note: to make this work, install
            # package in editable mode, `pip install -e .` in root directory.
from scipy.spatial.distance import pdist    # For some reason I have to
                                            # import this, instead of just
                                            # running it?
from numpy.random import default_rng
from ppca import ppca

#-----------------------------------------------------------------------------#
# Functions copied from Joe which create the image patches.

# %% codecell
def makeDCT():
    '''
    Constructs the DCT basis for the Klein bottle image patch model
    '''
    m1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    m2 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    m3 = np.array([[1,-2,1],[1,-2,1],[1,-2,1]])
    m4 = np.array([[1,1,1],[-2,-2,-2],[1,1,1]])
    m5 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
    m6 = np.array([[1,0,-1],[-2,0,2],[1,0,-1]])
    m7 = np.array([[1,-2,1],[0,0,0],[-1,2,-1]])
    m8 = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])
    D = np.array([[2,-1,0,-1,0,0,0,0,0],
                  [-1,3,-1,0,-1,0,0,0,0],
                  [0,-1,2,0,0,-1,0,0,0],
                  [-1,0,0,3,-1,0,-1,0,0],
                  [0,-1,0,-1,4,-1,0,-1,0],
                  [0,0,-1,0,-1,3,0,0,-1],
                  [0,0,0,-1,0,0,2,-1,0],
                  [0,0,0,0,-1,0,-1,3,-1],
                  [0,0,0,0,0,-1,0,-1,2]])
    v1 = m1.flatten()
    v1 = v1-np.mean(v1)
    v1 = v1/np.sqrt((v1.dot(D).dot(v1.T)))
    v2 = m2.flatten()
    v2 = v2-np.mean(v2)
    v2 = v2-np.sqrt((v2.dot(D).dot(v1.T)))*v1
    v2 = v2/np.sqrt((v2.dot(D).dot(v2.T)))
    v3 = m3.flatten()
    v3 = v3-np.mean(v3)
    v3 = v3-np.sqrt((v3.dot(D).dot(v2.T)))*v2-np.sqrt((v3.dot(D).dot(v1.T)))*v1
    v3 = v3/np.sqrt((v3.dot(D).dot(v3.T)))
    v4 = m4.flatten()
    v4 = v4-np.mean(v4)
    v4 = v4-np.sqrt((v4.dot(D).dot(v3.T)))*v3-np.sqrt((v4.dot(D).dot(v2.T)))*v2-np.sqrt((v4.dot(D).dot(v1.T)))*v1
    v4 = v4/np.sqrt((v4.dot(D).dot(v4.T)))
    return v1,v2,v3,v4

# %% codecell
def Klein(numa,numt):
    """
    Builds the Klein bottle image patch model with `numa` directional angles and
    `numt` bar angles. Ideal when numt = 2*numa.

    See Figure 6 in https://fds.duke.edu/db/attachment/2638

    Note: the data set is actually of size (numa+1)*(numt+1) for convenience,
    probably better ways to implement this.
    """
    K = []
    alphas = np.linspace(np.pi/4,5*np.pi/4,numa+1)[:numa+1]
    thetas = np.linspace(-np.pi/2,3*np.pi/2,numt+1)[:numt+1]
    for t in thetas:
        for a in alphas:
            vec = (np.cos(t)*np.cos(a)*v1 - np.cos(t)*np.sin(a)*v2
                + np.sin(t)*abs(np.cos(2*a)*v1 + np.sin(2*a)*(-v2)))
            K.append(vec)
    return np.round_(np.array(K),2), np.round_(alphas,2), np.round_(thetas,2)

#-----------------------------------------------------------------------------#

# %% codecell
# Get DCT basis vectors.
v1,v2,v3,v4 = makeDCT()
# Set up parameters.
numalphas = 50
numthetas = 2*numalphas
n_landmarks = 300
L, alphas,thetas = Klein(numalphas,numthetas)
L = np.squeeze(L)
D = sp.spatial.distance.pdist(L,'euclidean')
D = sp.spatial.distance.squareform(D)
print(D.shape)

# %% codecell
# Downsample the dataset to remove the points that are on top of other points.
big_sub_ind = pipeline.maxmin_subsample_distance_matrix(D, 5*n_landmarks)['indices']
D =  D[big_sub_ind, :][:, big_sub_ind]
L = L[big_sub_ind,:]
# Choose a landmark subset.
sub_ind = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks)['indices']
D_sub =  D[sub_ind, :][:, sub_ind]
L_sub = L[sub_ind,:]

# %% codecell
print('Computing persistence of the landmarks.')
PH_sub = ripser(D_sub, coeff=2, do_cocycles=True, maxdim=1,
    distance_matrix=True)
plot_diagrams(PH_sub['dgms'])
plt.title('Persistence of Data')
plt.show()
# Note that this should show two prominent cocycles in H^1 with F2 coefficients
# and only one with F3.

# %% codecell
# Get a prominent cocycle in dimension one.
print('Computing projective coordinates in dimension %d.' %len(sub_ind))
cocycles = PH_sub['cocycles'][1]
diagram = PH_sub['dgms'][1]
eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram,
    threshold_at_death=False, order=1)
eta2, birth2, death2 = pipeline.prominent_cocycle(cocycles, diagram,
    threshold_at_death=False, order=2)

# %% codecell
# Get a partition of unity.
part_func = pipeline.partition_unity(D, .45, sub_ind, bump_type='quadratic')
proj_coords = pipeline.proj_coordinates(part_func, eta)
D_pc = real_projective.projective_distance_matrix(proj_coords.T)
D_geo = real_projective.geo_distance_matrix(D_pc, k=8)

# %% codecell
# Compute PH of landmarks of high-dimensional data.
PH_pc = ripser(D_geo, distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_pc['dgms'])
plt.title('Persistence of Projective Coordinates')
plt.show()
# There are still two large H^1 cocyles, though they are hard to distinguish.

# %% codecell
# Apply PPCA to reduce to dimension 2.
X_ppca = ppca(proj_coords.T, 2)['X']
# Compute persistence of PCA output.
D_ppca = real_projective.projective_distance_matrix(X_ppca)
PH_ppca = ripser(D_ppca[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=1, coeff=3)
plot_diagrams(PH_ppca['dgms'])
plt.title('Persistence of PPCA Output')
plt.show()
# These are goofy. The major H^1 class seems to become slightly less persistent
# when coefficients are in F3. But it doesn't disappear...

# %% codecell
# Apply MDS to PCA output.
X_mds = geodesic_metric.rp_mds(D_geo, X=X_ppca.T)
#X_mds = real_projective.pmds(X_ppca, D_geo, max_iter=100)

# %% codecell
# Compute persistence of MDS output.
D_mds = geodesic_metric.RPn_distance_matrix(X_mds)
PH_mds = ripser(D_mds, distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_mds['dgms'])
plt.title('Peristence of MDS Output')
plt.show()

# Save the data.
#np.savez(filename, xy=xy, xy_sub = xy_sub, D=D, D_sub=D_sub, PH_sub=PH_sub,
#    proj_coords=proj_coords)
