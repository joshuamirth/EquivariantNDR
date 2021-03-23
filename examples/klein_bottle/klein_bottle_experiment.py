# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Klein Bottle Experiment
# %% [markdown]
# Construct a Klein bottle as a quotient of the unit square. Compute a distance matrix and projective coordinates. Then perform comparisons between:
# * PCA into different embedding dimensions (looking for right homology).
# * MDS into different embedding dimensions, starting with both PCA and random ICs.

# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import geodesic_mds
import pipeline
import geometry
from scipy.spatial.distance import pdist
from numpy.random import default_rng
from ppca import ppca
from importlib import reload


# %%
# Define functions for building quotient on Klein bottle (thanks, Joe Melby!)
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

# %% [markdown]
# Set a couple of parameters which will be significant throughout the experiment. It seems like reasonable results can be achieved with `N = 2000` and `n_landmarks = 50`.

# %%
N = 2000   # Total number of points
n_landmarks = 57
rng = default_rng(57)
pathname = 'examples/klein_bottle/'
filename = 'flat_klein_bottle_N%d.npz' %N

# %% [markdown]
# Construct the initial set of points and compute the distance matrix. The distance matrix `D` is the input data for our algorithm.

# %%
xy = rng.random((N, 2))
D = pdist(xy, minDist)
D = sp.spatial.distance.squareform(D)
sub_ind = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks, seed=[0])['indices']
D_sub = D[sub_ind, :][:, sub_ind]
xy_sub = xy[sub_ind,:]

# %% [markdown]
# Plot the output. Ideally this could be colored nicely for the paper.

# %%
c = xy[:,0]
cc = xy[:,1]
buff = 0.05
fig, ax = plt.subplots()
# ax.scatter(xy[:,0], xy[:,1], c=cc, cmap='viridis')
ax.scatter(xy[:,0], xy[:,1], c=c, cmap='Greys')
ax.scatter(xy_sub[:,0], xy_sub[:,1], c='red', marker='+')
ax.set_title('Klein Bottle Points with Landmarks ($N = %d$)' %N)
ax.set(xlim=(0-buff,1+buff), ylim=(0-buff,1+buff))
# plt.savefig(pathname+'klein_bottle_points.png', dpi=300)
plt.show()

# %% [markdown]
# ## Projective Coordinates
# 
# The distance matrix `D` can be used directly for embeddings with MDS; however, we can also use projective coordinates to get a theoretically more principalled initial condition. The next few cells build this projective coordinates in dimension `d = n_landmarks`.
# %% [markdown]
# The first step is to compute the persistence diagram in order to get a cocycle. Note that if we have correctly sampled a Klein bottle, then with $\mathbb{Z}/2\mathbb{Z}$ coefficients, the diagram should show two prominent $H^1$ classes and one prominent $H^2$ class. With $\mathbb{Z}/p\mathbb{Z}$ coefficients for $p$ odd prime there should instead be one prominent $H^1$ class and no $H^2$.

# %%
# Compute persistence of the landmarks.
p = 2
PH_sub = ripser(D_sub, coeff=p, do_cocycles=True, maxdim=2,
    distance_matrix=True)
plot_diagrams(PH_sub['dgms'])
plt.title('Klein Bottle Landmarks ($\mathbb{F}_%d$)' %p)
# plt.savefig(pathname+'klein_bottle_landmarks_F3.png', dpi=300)
plt.show()

# %% [markdown]
# Next we extract the prominent $H^1$ cocyle and compute projective coordinates from it. Note that there are two parameters here: we threshold the cocycle (meaning we use a representative of it only containing simplices occuring at time $t$) and we choose a radius for the partition of unity. If find that the first parameter should be near the birth of the cocycle and the latter near the death. I do not have a fully rigorous justification of this yet.
# 
# The essential output of this cell is the distance matrix `D_geo` and the coordinates themselves. `D_geo` is an approximation of the geodesic distance matrix for the data, not just the distance matrix of the points in high-dimensional projective space.

# %%
cocycles = PH_sub['cocycles'][1]
diagram = PH_sub['dgms'][1]
eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram, order=2)
eta = pipeline.threshold_cocycle(eta, D_sub, birth+.01)
part_func = pipeline.partition_unity(D, death-0.01, sub_ind, bump_type='quadratic')
proj_coords = pipeline.proj_coordinates(part_func, eta)
D_pc = geometry.RPn_geo_distance_matrix(proj_coords)
D_geo = pipeline.geo_distance_matrix(D_pc, k=8)

# %% [markdown]
# We check that the persistent homology of the projective coordinates matches that of the original data. This is to test that the preceding parameters are reasonably chosen.

# %%
PH_pc = ripser(D_geo[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_pc['dgms'])
plt.title('Klein Bottle Projective Coordinates ($\mathbb{F}_%d$)' %p)
plt.show()

# %% [markdown]
# ## Dimensionality Reduction Experiments
# %% [markdown]
# ### Experiment 1: PPCA Dimension
# 
# First we run PPCA down to dimension one and plot the retained variance. If there is an "elbow" in that plot we have a good candidate for an embedding dimension.

# %%
ppca_data = ppca(proj_coords.T, 1)
X_ppca = ppca_data['X'].T
variance = ppca_data['variance']
plt.plot(range(1,n_landmarks), 1-variance, 'bo--', linewidth=2, markersize=12)
plt.title('Percentage of cumulative variance retained')
plt.show()

# %% [markdown]
# This show a severe cutoff when we go below dimension eight. Let's get the PPCA embedding in dimension eight and check that it has good homology.

# %%
X_ppca8 = ppca(proj_coords.T, 8)['X'].T 
D_ppca8 = geometry.RPn_geo_distance_matrix(X_ppca8)
PH_ppca8 = ripser(D_ppca8[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_ppca8['dgms'])
plt.title('PPCA into $\mathbb{R}P^8$')
plt.show()

# %% [markdown]
# The homology here is correct, so this a topologically reasonable embedding. Now let's confirm that in dimension seven something is lost.

# %%
X_ppca7 = ppca(proj_coords.T, 7)['X'].T 
D_ppca7 = geometry.RPn_geo_distance_matrix(X_ppca7)
PH_ppca7 = ripser(D_ppca7[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_ppca7['dgms'])
plt.title('PPCA into $\mathbb{R}P^7$')
plt.show()

# %% [markdown]
# No! Despite the decrease in variance, the embedding remains reasonable topologically. We will keep going down one dimension at a time until we lose it.

# %%
p=2
X_ppca6 = ppca(proj_coords.T, 6)['X'].T 
D_ppca6 = geometry.RPn_geo_distance_matrix(X_ppca6)
PH_ppca6 = ripser(D_ppca6[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_ppca6['dgms'])
plt.title('PPCA into $\mathbb{R}P^6$')
plt.show()

X_ppca5 = ppca(proj_coords.T, 5)['X'].T 
D_ppca5 = geometry.RPn_geo_distance_matrix(X_ppca5)
PH_ppca5 = ripser(D_ppca5[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_ppca5['dgms'])
plt.title('PPCA into $\mathbb{R}P^5$')
plt.show()

X_ppca4 = ppca(proj_coords.T, 4)['X'].T 
D_ppca4 = geometry.RPn_geo_distance_matrix(X_ppca4)
PH_ppca4 = ripser(D_ppca4[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_ppca4['dgms'])
plt.title('PPCA into $\mathbb{R}P^4$')
plt.show()

X_ppca3 = ppca(proj_coords.T, 3)['X'].T 
D_ppca3 = geometry.RPn_geo_distance_matrix(X_ppca3)
PH_ppca3 = ripser(D_ppca3[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_ppca3['dgms'])
plt.title('PPCA into $\mathbb{R}P^3$')
plt.show()

X_ppca2 = ppca(proj_coords.T, 2)['X'].T 
D_ppca2 = geometry.RPn_geo_distance_matrix(X_ppca2)
PH_ppca2 = ripser(D_ppca2[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_ppca2['dgms'])
plt.title('PPCA into $\mathbb{R}P^2$')
plt.show()

# %% [markdown]
# Comparing the versions with $\mathbb{Z}/3\mathbb{Z}$ and $\mathbb{Z}/2\mathbb{Z}$ coefficients, we see that we get the correct homology down to $\mathbb{R}P^5$. The four-dimensional output gains a spurious $H^2$ class and loses the second $H^1$ class.
# %% [markdown]
# ### Experiment 2: MDS
# 
# Now that we have initial conditions, we can use MDS to try and recover the correct homology. The aim is to see if we can get a topologically correct embedding in a lower dimension.

# %%
X_mds3 = geodesic_mds.rp_mds(D_geo, dim=3, X=X_ppca3)
D_mds3 = geometry.RPn_geo_distance_matrix(X_mds3)
PH_mds3 = ripser(D_mds3[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_mds3['dgms'])
plt.title('PMDS into $\mathbb{R}P^3$')
plt.show()


# %%
X_mds4 = geodesic_mds.rp_mds(D_geo, dim=4, X=X_ppca4)
D_mds4 = geometry.RPn_geo_distance_matrix(X_mds4)


# %%
PH_mds4 = ripser(D_mds4[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_mds4['dgms'])
plt.title('PMDS into $\mathbb{R}P^4$')
plt.show()
PH_mds4 = ripser(D_mds4[sub_ind,:][:,sub_ind], distance_matrix=True, maxdim=2, coeff=3)
plot_diagrams(PH_mds4['dgms'])
plt.title('PMDS into $\mathbb{R}P^4$')
plt.show()


# %%
X_mds5 = geodesic_mds.rp_mds(D_geo, dim=5, X=X_ppca5)
D_mds5 = geometry.RPn_geo_distance_matrix(X_mds5)


# %%
PH_mds5 = ripser(D_mds5[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_mds5['dgms'])
plt.title('PMDS into $\mathbb{R}P^5$')
plt.show()
PH_mds5 = ripser(D_mds5[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=3)
plot_diagrams(PH_mds5['dgms'])
plt.title('PMDS into $\mathbb{R}P^5$')
plt.show()

# %% [markdown]
# Why is $\mathbb{R}P^5$ actually _worse_ than without the MDS embedding?

# %%
X_mds6 = geodesic_mds.rp_mds(D_geo, dim=6, X=X_ppca6)
D_mds6 = geometry.RPn_geo_distance_matrix(X_mds6)


# %%
PH_mds6 = ripser(D_mds6[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_mds6['dgms'])
plt.title('PMDS into $\mathbb{R}P^6$')
plt.show()
PH_mds6 = ripser(D_mds6[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=3)
plot_diagrams(PH_mds6['dgms'])
plt.title('PMDS into $\mathbb{R}P^6$')
plt.show()


# %%
X_mds7 = geodesic_mds.rp_mds(D_geo, dim=7, X=X_ppca7)
D_mds7 = geometry.RPn_geo_distance_matrix(X_mds7)


# %%
PH_mds7 = ripser(D_mds7[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_mds7['dgms'])
plt.title('PMDS into $\mathbb{R}P^7$')
plt.show()
PH_mds7 = ripser(D_mds7[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=3)
plot_diagrams(PH_mds7['dgms'])
plt.title('PMDS into $\mathbb{R}P^7$')
plt.show()


# %%
X_mds8 = geodesic_mds.rp_mds(D_geo, dim=8, X=X_ppca8)
D_mds8 = geometry.RPn_geo_distance_matrix(X_mds8)


# %%
PH_mds8 = ripser(D_mds8[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_mds8['dgms'])
plt.title('PMDS into $\mathbb{R}P^8$')
plt.show()
PH_mds8 = ripser(D_mds8[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=3)
plot_diagrams(PH_mds8['dgms'])
plt.title('PMDS into $\mathbb{R}P^8$')
plt.show()

# %% [markdown]
# Okay, so it isn't really much _worse_, but it likes to introduce this extra $H^2$ class that shouldn't be there.
# %% [markdown]
# ### Experiment 3: MDS with Random IC
# 
# Repeating the above, but now using a random initial condition. The question is how much worse than the normal MDS this will be.

# %%
rand_ic3 = rng.random(X_ppca3.shape)
rand_ic4 = rng.random(X_ppca4.shape)
rand_ic5 = rng.random(X_ppca5.shape)
rand_ic6 = rng.random(X_ppca6.shape)
rand_ic7 = rng.random(X_ppca7.shape)
rand_ic8 = rng.random(X_ppca8.shape)
X_rand3 = geodesic_mds.rp_mds(D_geo, dim=3, X=rand_ic3)
X_rand4 = geodesic_mds.rp_mds(D_geo, dim=4, X=rand_ic4)
X_rand5 = geodesic_mds.rp_mds(D_geo, dim=5, X=rand_ic5)
X_rand6 = geodesic_mds.rp_mds(D_geo, dim=6, X=rand_ic6)
X_rand7 = geodesic_mds.rp_mds(D_geo, dim=7, X=rand_ic7)
X_rand8 = geodesic_mds.rp_mds(D_geo, dim=8, X=rand_ic8)


# %%
D_rand3 = geometry.RPn_geo_distance_matrix(X_rand3)
D_rand4 = geometry.RPn_geo_distance_matrix(X_rand4)
D_rand5 = geometry.RPn_geo_distance_matrix(X_rand5)
D_rand6 = geometry.RPn_geo_distance_matrix(X_rand6)
D_rand7 = geometry.RPn_geo_distance_matrix(X_rand7)
D_rand8 = geometry.RPn_geo_distance_matrix(X_rand8)


# %%
p = 2
PH_rand3 = ripser(D_rand3[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_mds3['dgms'])
plt.title('PMDS into $\mathbb{R}P^3$')
plt.show()
PH_rand4 = ripser(D_rand4[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_mds4['dgms'])
plt.title('PMDS into $\mathbb{R}P^4$')
plt.show()
PH_rand5 = ripser(D_rand5[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_mds5['dgms'])
plt.title('PMDS into $\mathbb{R}P^5$')
plt.show()
PH_rand6 = ripser(D_rand6[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_mds6['dgms'])
plt.title('PMDS into $\mathbb{R}P^6$')
plt.show()
PH_rand7 = ripser(D_rand7[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_mds7['dgms'])
plt.title('PMDS into $\mathbb{R}P^7$')
plt.show()
PH_rand8 = ripser(D_rand8[sub_ind, :][:, sub_ind], distance_matrix=True, maxdim=2, coeff=p)
plot_diagrams(PH_mds8['dgms'])
plt.title('PMDS into $\mathbb{R}P^8$')
plt.show()

# %% [markdown]
# ## Analysis: Cost Improvement
# 
# The goal is to achieve a result similar to the original distance matrix. Therefore, we should check the difference between our end matrix and the original.

# %%
diff_MDS = np.linalg.norm(np.array([D_mds3, D_mds4, D_mds5, D_mds6, D_mds7, D_mds8]) - D_geo, axis=(1,2))
diff_PCA = np.linalg.norm(np.array([D_ppca3, D_ppca4, D_ppca5, D_ppca6, D_ppca7, D_ppca8]) - D_geo, axis=(1,2))
diff_rand = np.linalg.norm(np.array([D_rand3, D_rand4, D_rand5, D_rand6, D_rand7, D_rand8]) - D_geo, axis=(1,2))


# %%
print(diff_MDS)
print(diff_PCA)
print(diff_rand)


# %%
fig, ax = plt.subplots()
ax.plot(np.arange(3, 9), diff_MDS, 'bo--', linewidth=2, markersize=5, label='PMDS (PCA IC)')
ax.plot(np.arange(3, 9),diff_rand, 'ro--', linewidth=2, markersize=5, label='PMDS (random IC)')
ax.plot(np.arange(3, 9),diff_PCA, 'go--', linewidth=2, markersize=5, label='PPCA')
ax.legend()
ax.set_xlabel('Embedding Dimension')
ax.set_ylabel('$\|\| D - D_{goal} \|\|$')
plt.title('Difference with Goal Distance Matrix')
plt.show()

# %% [markdown]
# 

