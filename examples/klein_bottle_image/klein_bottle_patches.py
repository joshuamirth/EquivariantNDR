# Script for constructing the data set of image patches realizing Klein bottle.

# %% codecell
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import geodesic_mds
from scipy.spatial.distance import pdist    # For some reason I have to
                                            # import this, instead of just
                                            # running it?
from numpy.random import default_rng
from ppca import ppca
import pipeline
import geometry

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
pathname = 'examples/klein_bottle_image/'

# %% codecell
# Get DCT basis vectors.
v1,v2,v3,v4 = makeDCT()
# Set up parameters.
numalphas = 50
numthetas = 2*numalphas
n_landmarks = 300
L, alphas,thetas = Klein(numalphas,numthetas)
L = np.squeeze(L)
print(L.shape)
D = sp.spatial.distance.pdist(L,'euclidean')
D = sp.spatial.distance.squareform(D)
print(D.shape)

# %% codecell
# Downsample the dataset to remove the points that are on top of other points.
big_sub_ind = pipeline.maxmin_subsample_distance_matrix(D, 5*n_landmarks)['indices']
D =  D[big_sub_ind, :][:, big_sub_ind]
L = L[big_sub_ind,:]
print(D.shape)
print(L.shape)
# Choose a landmark subset.
sub_ind = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks)['indices']
D_sub =  D[sub_ind, :][:, sub_ind]
L_sub = L[sub_ind,:]
print(D_sub.shape)
print(L_sub.shape)

# %% codecell
np.savez('examples/klein_bottle_image/klein_bottle_patch.npz', D=D, L=L, D_sub=D_sub, L_sub=L_sub)

# %% codecell
print('Computing persistence of the landmarks.')
PH_sub = ripser(D_sub, coeff=2, do_cocycles=True, maxdim=1,
    distance_matrix=True)
plot_diagrams(PH_sub['dgms'])
plt.title('Klein Bottle Patches ($\mathbb{F}_2$)')
# plt.show()
plt.savefig(pathname+'klein_bottle_patches_F2.png', dpi=300)
# Note that this should show two prominent cocycles in H^1 with F2 coefficients
# and only one with F3.

# %% codecell
# Get a prominent cocycle in dimension one.
print('Computing projective coordinates in dimension %d.' %len(sub_ind))
cocycles = PH_sub['cocycles'][1]
diagram = PH_sub['dgms'][1]
eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram, order=1)
eta2, birth2, death2 = pipeline.prominent_cocycle(cocycles, diagram, order=2)
eta = pipeline.threshold_cocycle(eta, D_sub, birth+.01)
print(birth, death)

# %% codecell
# Get a partition of unity.
part_func = pipeline.partition_unity(D, death-.01, sub_ind, bump_type='quadratic')
proj_coords = pipeline.proj_coordinates(part_func, eta)
D_pc = geometry.RPn_geo_distance_matrix(proj_coords)
D_geo = pipeline.geo_distance_matrix(D_pc, k=8)

# %% codecell
# Compute PH of landmarks of high-dimensional data.
PH_pc = ripser(D_geo[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_pc['dgms'])
plt.title('Klein Bottle Patch Projective Coordinates ($\mathbb{F}_2$)')
plt.savefig(pathname+'klein_bottle_proj_coords_F2.png', dpi=300)
plt.show()
# There are still two large H^1 cocyles, though they are hard to distinguish.

# %% codecell
# Apply PPCA to reduce to dimension 2.
X_ppca = ppca(proj_coords.T, 2)['X'].T
# Compute persistence of PCA output.
D_ppca = geometry.RPn_geo_distance_matrix(X_ppca)
PH_ppca = ripser(D_ppca[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=1, coeff=3)
plot_diagrams(PH_ppca['dgms'])
plt.title('Klein Bottle Patch Projective PCA ($\mathbb{F}_3$)')
plt.savefig(pathname+'klein_bottle_PPCA_F3.png', dpi=300)
plt.show()
# These are goofy. The major H^1 class seems to become slightly less persistent
# when coefficients are in F3. But it doesn't disappear...

# %% codecell
# Apply MDS to PCA output.
X_mds = geodesic_mds.rp_mds(D_geo, X=X_ppca)
X_rand = geodesic_mds.rp_mds(D_geo)
#X_mds = real_projective.pmds(X_ppca, D_geo, max_iter=100)

# %% codecell
# Compute persistence of MDS output.
D_mds = geometry.RPn_geo_distance_matrix(X_mds)
PH_mds = ripser(D_mds, distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_mds['dgms'])
plt.title('Peristence of MDS Output')
plt.show()

# %% codecell
# Compute persistence of MDS output.
D_rand = geometry.RPn_geo_distance_matrix(X_rand)
PH_rand = ripser(D_rand, distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_rand['dgms'])
plt.title('Peristence of MDS Output')
plt.show()

# Save the data.
#np.savez(filename, xy=xy, xy_sub = xy_sub, D=D, D_sub=D_sub, PH_sub=PH_sub,
#    proj_coords=proj_coords)

# %% codecell
# Visualize the different outputs.
neg_idx = np.where(X_ppca[2,:] < 0)
X_ppca[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_ppca[0,:], X_ppca[1,:], cmap='YlGn')
ax.axis('equal')
ax.set_title('PPCA Embedding')
plt.show()

# %% codecell
# Visualize the different outputs.
neg_idx = np.where(X_mds[2,:] < 0)
X_mds[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_mds[0,:], X_mds[1,:], cmap='YlGn')
ax.axis('equal')
ax.set_title('MDS Embedding')
plt.show()

# %% codecell
# Visualize the different outputs.
neg_idx = np.where(X_rand[2,:] < 0)
X_rand[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_rand[0,:], X_rand[1,:], cmap='YlGn')
ax.axis('equal')
ax.set_title('Random Embedding')
plt.show()
