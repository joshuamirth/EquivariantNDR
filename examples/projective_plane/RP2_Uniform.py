# %% codecell
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import real_projective
import cplx_projective
from examples import pipeline   # Note: to make this work, install
            # package in editable mode, `pip install -e .` in root directory.
from ppca import ppca
from visualization import my3dscatterplot
import pymanopt
import chordal_metric
import geodesic_metric
from importlib import reload
from scipy.spatial.distance import pdist, squareform    # For some reason I have to
                                            # import this, instead of just
                                            # running it?
# %% codecell
reload(real_projective)

# %% codecell
# Uniformly sample points from the sphere.
rng = np.random.default_rng(57)
N = 1000     # Number of points to sample.
n_landmarks = 100
RP = rng.standard_normal((3,N))
RP = RP/np.linalg.norm(RP, axis=0)
# Place all points onto their upper hemisphere representative.
neg_idx = np.where(RP[2,:] < 0)
RP[:,neg_idx] = -RP[:,neg_idx]
D = real_projective.projective_distance_matrix(RP.T)

# %% codecell
c = np.linalg.norm(RP[0:2,:],axis=0) # Distance from origin
fig, ax = plt.subplots()
ax.scatter(RP[0,:], RP[1,:], c=c, cmap='YlGn')
ax.axis('equal')
plt.show()

# %% codecell
idx = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks, seed=[57])['indices']
D_sub = D[idx,:][:,idx]
RP_sub = RP[:,idx]
PH_sub = ripser(D_sub, coeff=2, do_cocycles=True, maxdim=1,
    distance_matrix=True)
plot_diagrams(PH_sub['dgms'])
plt.show()

# %% codecell
eta, birth, death = pipeline.prominent_cocycle(PH_sub['cocycles'][1],
    PH_sub['dgms'][1], threshold_at_death=False)

# %% codecell
print(eta)
print('Birth: %2.4f' %birth)
print('Death: %2.4f' %death)

# %% codecell
# Get a partition of unity.
print('Computing partition of unity.')
part_func = pipeline.partition_unity(D, 0.26, idx, bump_type='quadratic')
print('Finding projective coordinates.')
proj_coords = pipeline.proj_coordinates(part_func, eta)
print('Computing distance matrix of projective coordinates.')
D_pc = real_projective.projective_distance_matrix(proj_coords.T)
print('Estimating geodesic distance matrix.')
D_geo = real_projective.geo_distance_matrix(D_pc, k=12)

# %% codecell
# Compute PH of landmarks of high-dimensional data.
print('Computing persistence of projective coordinates.')
PH_pc2 = ripser(D_geo[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_pc2['dgms'])
plt.title('Persistence of Projective Coordinates ($F_2$)')
plt.show()
PH_pc2 = ripser(D_geo[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=3)
plot_diagrams(PH_pc3['dgms'])
plt.title('Persistence of Projective Coordinates ($F_3$)')
plt.show()

# %% codecell
# Reduce dimension using PCA:
X_ppca = ppca(proj_coords.T, 2)['X']
# Compute persistence of PCA output.
D_ppca = real_projective.projective_distance_matrix(X_ppca)
PH_ppca2 = ripser(D_ppca[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_ppca2['dgms'])
plt.title('Persistence of PPCA Embedding ($F_2$)')
plt.show()
PH_ppca3 = ripser(D_ppca[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=3)
plot_diagrams(PH_ppca3['dgms'])
plt.title('Persistence of PPCA Embedding ($F_3$)')
plt.show()

# %% codecell
# Apply MDS with squared geodesic cost function.
X_mds = geodesic_metric.rp_mds(D_geo, X=X_ppca.T)

# %% codecell
# Compute persistence of MDS output.
D_mds = geodesic_metric.RPn_distance_matrix(X_mds)
PH_mds2 = ripser(D_mds[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_mds2['dgms'])
plt.title('Persistence of geodesic MDS Output ($F_2$)')
plt.show()
PH_mds3 = ripser(D_mds[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=3)
plot_diagrams(PH_mds3['dgms'])
plt.title('Persistence of geodeisc MDS Output ($F_3$)')
plt.show()

# %% codecell
neg_idx = np.where(X_ppca[:,2] < 0)
X_ppca[neg_idx,:] *= -1
fig, ax = plt.subplots()
ax.scatter(X_ppca[:,0], X_ppca[:,1], c=c, cmap='YlGn')
ax.axis('equal')
ax.set_title('PPCA Embedding')
plt.show()

# %% codecell
neg_idx = np.where(X_mds[2,:] < 0)
X_mds[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_mds[0,:], X_mds[1,:], c=c, cmap='YlGn')
ax.axis('equal')
ax.set_title('MDS (geodesic) Embedding.')
plt.show()
# %% codecell
D_goal = D_geo/np.max(D_geo)
X_chrd = chordal_metric.rp_mds(D_goal, X=X_ppca.T)

# %% codecell
neg_idx = np.where(X_chrd[2,:] < 0)
X_chrd[:,neg_idx] *= -1
fig, ax = plt.subplots()
ax.scatter(X_chrd[0,:], X_chrd[1,:], c=c, cmap='YlGn')
ax.axis('equal')
ax.set_title('MDS (chordal) Embedding.')
plt.show()

# %% codecell
# Compute persistence of MDS output.
D_out = geodesic_metric.RPn_distance_matrix(X_chrd)
PH_out2 = ripser(D_out[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_out2['dgms'])
plt.title('Persistence of chordal MDS ($F_2$)')
plt.show()
PH_out3 = ripser(D_out[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=3)
plot_diagrams(PH_out3['dgms'])
plt.title('Persistence of chordal MDS ($F_3$)')
plt.show()

# %% markdown
# # Analysis
#
# Compare the different methods here and classical MDS.
# %% codecell
# Do classical MDS of the distance matrix.
A = -0.5*D_goal**2
H = np.eye(N) - (1/N)*np.ones((N,N)) # centering matrix
B = H@A@H
L, G = np.linalg.eig(B)
print(np.abs(L[0:10]))
X = G*np.sqrt(L)
X_3 = np.real(X[:,0:3])
X_4 = np.real(X[:,0:4])
X_5 = np.real(X[:,0:5])

# %% codecell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3[:,0], X_3[:,1], X_3[:,2], c=c, cmap='YlGn')
plt.title('Metric MDS Embedding into $\mathbb{R}^3$')
plt.show()


# %% codecell
D_3 = sp.spatial.distance.cdist(X_3, X_3, metric='euclidean')
PH_3 = ripser(D_3[idx, :][:, idx], distance_matrix=True, maxdim=1)
plot_diagrams(PH_3['dgms'])
plt.title('Metric MDS Persistence ($\mathbb{R}^3$)')
plt.show()

# %% codecell
D_4 = sp.spatial.distance.cdist(X_4, X_4, metric='euclidean')
PH_4 = ripser(D_4[idx, :][:, idx], distance_matrix=True, maxdim=1)
plot_diagrams(PH_4['dgms'])
plt.title('Metric MDS Persistence ($\mathbb{R}^4$)')
plt.show()

# %% codecell
D_5 = sp.spatial.distance.cdist(X_5, X_5, metric='euclidean')
PH_5 = ripser(D_5[idx, :][:, idx], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_5['dgms'])
plt.title('Metric MDS Persistence ($\mathbb{R}^5$)')
plt.show()

# %% codecell
print(np.linalg.norm(D_3 - D_goal))
print(np.linalg.norm(D_4 - D_goal))
D_out = chordal_metric.RPn_chordal_distance_matrix(X_chrd)
print(np.linalg.norm(D_out - D_goal))
# %% markdown
# Weirdly, the Euclidean embedding _is_ recovering the distances better overall.
# %% codecell
D_out = (D_out + D_out.T)/2
np.fill_diagonal(D_out, 0)
# %% codecell
D_3vec = squareform(D_3)
D_4vec = squareform(D_4)
D_outvec = squareform(D_out)
D_vec = squareform(D_goal)

# %% codecell
fig, ax = plt.subplots()
ax.scatter(D_vec, D_3vec, marker='.')
ax.plot(D_vec, D_vec, color='#ff7f0e')
ax.set_xlabel('Goal Distance')
ax.set_ylabel('Metric MDS Distance')
plt.show()

# %% codecell
fig, ax = plt.subplots()
ax.scatter(D_vec, D_4vec, marker='.')
ax.plot(D_vec, D_vec, color='#ff7f0e')
ax.set_xlabel('Goal Distance')
ax.set_ylabel('Metric MDS Distance (R^4)')
plt.show()

# %% codecell
fig, ax = plt.subplots()
ax.scatter(D_vec, D_outvec, marker='.')
ax.plot(D_vec, D_vec, color='#ff7f0e')
ax.set_xlabel('Goal Distance')
ax.set_ylabel('Projective MDS Distance')
plt.show()

# %% codecell
resid_3 = (D_3vec - D_vec)
resid_out = (D_outvec - D_vec)

# %% codecell
fig, ax = plt.subplots()
ax.scatter(D_vec, resid_out**2, marker='*')
ax.scatter(D_vec, resid_out, marker='+')
ax.set_xlabel('Goal Distances')
ax.set_ylabel('Projective MDS (Squared) Residuals')
plt.show()

# %% codecell
fig, ax = plt.subplots()
ax.scatter(D_vec, resid_3**2, marker='*')
ax.scatter(D_vec, resid_3, marker='+')
ax.set_xlabel('Goal Distances')
ax.set_ylabel('Metric MDS (Squared) Residuals')
plt.show()
