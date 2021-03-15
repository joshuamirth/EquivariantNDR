# Script for constructing data on the flat model of the Klein bottle.
# Modified from code provided by Joe Melby.
# %% codecell
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import geodesic_mds
import pipeline
import geometry
from scipy.spatial.distance import pdist    # For some reason I have to
                                            # import this, instead of just
                                            # running it?
from numpy.random import default_rng
from ppca import ppca

# %% codecell
#-----------------------------------------------------------------------------#
# Functions copied from Joe which give the geometry of the flat Klein
# bottle.

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

# %% codecell
# Construct a uniformly random grid of points in the unit square.
rng = default_rng(57)
N = 2000   # Total number of points
pathname = 'examples/klein_bottle/'
filename = 'flat_klein_bottle_N%d.npz' %N
print('Generating %d points on the flat klein bottle.' %N)
n_landmarks = 50
# x, y = np.meshgrid(rng.random(numx),rng.random(numy))
# xy = np.column_stack((x.ravel(),y.ravel()))
xy = rng.random((N, 2))

# %% codecell
# Construct the distance matrix of the points.
print('Computing distance matrix and landmark subset.')
D = pdist(xy, minDist)
D = sp.spatial.distance.squareform(D)
# Subsample the distance matrix with max/min.
sub_ind = pipeline.maxmin_subsample_distance_matrix(D,
    n_landmarks)['indices']
D_sub = D[sub_ind, :][:, sub_ind]
xy_sub = xy[sub_ind,:]

# %% codecell
c = xy[:,0]
cc = xy[:,1]
buff = 0.05
fig, ax = plt.subplots()
# ax.scatter(xy[:,0], xy[:,1], c=cc, cmap='viridis')
ax.scatter(xy[:,0], xy[:,1], c=c, cmap='cividis')
ax.scatter(xy_sub[:,0], xy_sub[:,1], c='red', marker='+')
ax.set_title('Klein Bottle Points with Landmarks ($N = %d$)' %N)
ax.set(xlim=(0-buff,1+buff), ylim=(0-buff,1+buff))
# plt.savefig(pathname+'klein_bottle_points.png', dpi=300)
plt.show()


# %% codecell
# Compute persistence of the landmarks.
PH_sub = ripser(D_sub, coeff=2, do_cocycles=True, maxdim=2,
    distance_matrix=True)
plot_diagrams(PH_sub['dgms'])
plt.title('Klein Bottle Landmarks ($\mathbb{F}_2$)')
# plt.savefig(pathname+'klein_bottle_landmarks_F3.png', dpi=300)
plt.show()

# %% codecell
# Get a prominent cocycle in dimension one.
print('Computing projective coordinates in dimension %d.' %len(sub_ind))
cocycles = PH_sub['cocycles'][1]
diagram = PH_sub['dgms'][1]
eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram, order=2)
print(birth, death)

# %% codecell
# Get a partition of unity.
eta = pipeline.threshold_cocycle(eta, D_sub, birth+.01)
part_func = pipeline.partition_unity(D, death-0.01, sub_ind, bump_type='quadratic')
proj_coords = pipeline.proj_coordinates(part_func, eta)
D_pc = geometry.RPn_geo_distance_matrix(proj_coords)
D_geo = pipeline.geo_distance_matrix(D_pc, k=8)

# %% codecell
# Compute PH of landmarks of high-dimensional data.
PH_pc = ripser(D_geo[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_pc['dgms'])
plt.show()

# %% codecell
# Apply PPCA to reduce to dimension 2.
stuff = ppca(proj_coords.T, 2)
X_ppca = stuff['X'].T
variance = stuff['variance']
plt.plot(range(2,10), 1-variance[0:8], 'bo--', linewidth=2, markersize=12)
plt.title('Percentage of cumulative variance')
plt.show()

# %% codecell
# Visualize PPCA output.
draw_circle = plt.Circle((0,0), 1.0+buff/2, fill=False)
neg_idx = np.where(X_ppca[2,:] < 0)
X_ppca[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_ppca[0,:], X_ppca[1,:], c=c, cmap='cividis')
# ax.scatter(X_ppca[0,:], X_ppca[1,:], c=cc, cmap='viridis')
ax.add_artist(draw_circle)
ax.set(xlim=(-1-buff,1+buff), ylim=(-1-buff,1+buff))
ax.set_aspect('equal', adjustable='box')
ax.set_title('PPCA Embedding')
plt.xlabel('y')
plt.ylabel('z')
plt.show()
# fig, ax = plt.subplots()
# ax.scatter(X_ppca[0,1], X_ppca[1,:], c=c, cmap='cividis')
# ax.set_title('Klein Bottle PPCA')
# ax.set(xlim=(0-buff,1+buff), ylim=(0-buff,1+buff))
# plt.savefig(pathname+'klein_bottle_points.png', dpi=300)
plt.show()

# %% codecell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_ppca[0,:], X_ppca[1,:], X_ppca[2,:], c=c, cmap='cividis')
ax.scatter(-X_ppca[0,:], -X_ppca[1,:], -X_ppca[2,:], c=c, cmap='cividis')
# ax.scatter(X_ppca[0,:], X_ppca[1,:], X_ppca[2,:], c=cc, cmap='viridis')
# ax.scatter(-X_ppca[0,:], -X_ppca[1,:], -X_ppca[2,:], c=cc, cmap='viridis')
ax.view_init(20, 0) #elev, azim pair. Default 30, -60.
ax.set_title('3D Plot of PPCA Output')
plt.show()


# %% codecell
# Compute persistence of PCA output.
D_ppca = geometry.RPn_geo_distance_matrix(X_ppca)
PH_ppca = ripser(D_ppca[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2)
plot_diagrams(PH_ppca['dgms'])
plt.show()

# %% codecell
# Apply MDS to PCA output.
X_mds = geodesic_mds.rp_mds(D_geo, X=X_ppca)

# %% codecell
# Compute persistence of MDS output.
D_mds = geometry.RPn_geo_distance_matrix(X_mds)
PH_mds = ripser(D_mds[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2)
plot_diagrams(PH_mds['dgms'])
plt.show()

# %% codecell
# Visualize MDS output.
draw_circle = plt.Circle((0,0), 1.0+buff/2, fill=False)
neg_idx = np.where(X_mds[2,:] < 0)
X_mds[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_mds[0,:], X_mds[1,:], c=c, cmap='cividis')
ax.add_artist(draw_circle)
ax.set(xlim=(-1-buff,1+buff), ylim=(-1-buff,1+buff))
ax.set_aspect('equal', adjustable='box')
ax.set_title('MDS Embedding (PCA IC)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %% codecell
X_rand = geodesic_mds.rp_mds(D_geo)

# %% codecell
# Compute persistence of MDS output.
D_rand = geometry.RPn_geo_distance_matrix(X_rand)
PH_rand = ripser(D_rand[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2)
plot_diagrams(PH_rand['dgms'])
plt.title('MDS Embedding (Random IC)')
plt.show()
# Save the data.
#np.savez(filename, xy=xy, xy_sub = xy_sub, D=D, D_sub=D_sub, PH_sub=PH_sub,
#    proj_coords=proj_coords)

# %% codecell
draw_circle = plt.Circle((0,0), 1.0+buff/2, fill=False)
neg_idx = np.where(X_rand[2,:] < 0)
X_rand[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_rand[0,:], X_rand[1,:], c=c, cmap='cividis')
ax.add_artist(draw_circle)
ax.set(xlim=(-1-buff,1+buff), ylim=(-1-buff,1+buff))
ax.set_aspect('equal', adjustable='box')
ax.set_title('MDS Embedding (random IC)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis = X_rand
ax.scatter(vis[0,:], vis[1,:], vis[2,:], c=c, cmap='cividis')
# ax.scatter(-vis[0,:], -vis[1,:], -vis[2,:], c=c, cmap='cividis')
ax.view_init(30, 30) #elev, azim pair. Default 30, -60.
ax.set_title('Output of MDS (random initial condition)')
plt.show()

# %%
ip = X_rand.T @ X_rand
n_idx = np.where(ip[:,50]<0)
X_rand[:,n_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_rand[0,:], X_rand[1,:], c=c, cmap='cividis')
draw_circle = plt.Circle((0,0), 1.0+buff/2, fill=False)
ax.add_artist(draw_circle)
ax.set(xlim=(-1-buff,1+buff), ylim=(-1-buff,1+buff))
ax.set_aspect('equal', adjustable='box')
ax.set_title('MDS Embedding (random IC)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
