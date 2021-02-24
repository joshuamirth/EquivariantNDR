# %% codecell
# Script for generating images of point.
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from examples import pipeline
import cplx_projective
import chordal_metric
import geodesic_metric
from scipy.spatial.distance import pdist, squareform    # For some reason I have to
                                            # import this, instead of just
                                            # running it?
# %% codecell
# Setup parameters
N = 12  # Create N*N images
amp = 0.5
sd = 1.0 # Standard deviation needs to be large enough for overlap.
K = 20  # Create K^2 total images.
n_landmarks = 100

# %% codecell
# Generate the images:
x = np.linspace(-1,1,N)
y = np.linspace(-1,1,N)
xx, yy = np.meshgrid(x,y)
z = np.zeros((K**2, xx.shape[0], xx.shape[1]))
mux = np.linspace(-1 - 2*sd, 1 + 2*sd, K)
muy = np.linspace(-1 - 2*sd, 1 + 2*sd, K)
mmx, mmy = np.meshgrid(mux, muy)
mm = np.column_stack((mmx.ravel(), mmy.ravel()))
for i in range(K**2):
    z[i,:,:] = amp/(2*np.pi*sd**2) * np.exp((-1/2)*(((xx-mm[i,0])/sd)**2 +
        ((yy-mm[i,1])/sd)**2)**2)

data = np.reshape(z, (K**2, N**2))    # Matrix with each row a data point.
# data = np.vstack(data, np.zeros(N**2))

# %% codecell
# Make a big plot showing all of the images:
fig, axs = plt.subplots(K,K)
for i in range(K):
    for j in range(K):
        axs[i][j].imshow(z[j+i*K,:,:], cmap='gray')
        axs[i][j].set_axis_off()
plt.show()

# %% codecell
c = np.linalg.norm(data, axis=1)
cc = np.reshape(c, (20,20))
fig, ax = plt.subplots()
ax.imshow(cc, cmap='cividis')
plt.show()

# %% codecell
# Compute the distance matrix and persistence.
D = sp.spatial.distance.cdist(data, data, metric='euclidean')
# TODO: improve this maxmin function.
sub_ind = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks, seed=[57])['indices']
D_sub = D[sub_ind, :][:, sub_ind]
PH = ripser(D, distance_matrix=True, maxdim=2)
plot_diagrams(PH['dgms'])
plt.show()
# If the parameters are chosen well the PH should have a nice H^2 class.

# %% codecell
# Set up a random initial condition.
rng = np.random.default_rng(57)
X_rand = rng.uniform(-1, 1, (4, n_landmarks+1))
# X_rand = rng.uniform(-1, 1, (4, K**2))
X_rand = X_rand / np.linalg.norm(X_rand, axis=0)

# %% codecell
# Note that these distance matrices are restricted to the landmark points.
# Because many distances are near zero, the weights computed for the geodesic
# distance end up containing infinite values when using the whole dataset. Oddly
# this doesn't seem to change the result much, but I am trying to avoid it.
D_chrd = D_sub / np.max(D_sub)
D_geo = (np.pi/2)*D_chrd

# %% codecell
X_chrd = chordal_metric.cp_mds(D_chrd, X=X_rand)

# %% codecell
reload(geodesic_metric)
X_geo = geodesic_metric.cp_mds(D_geo, X=X_rand)

# %% codecell
D_out_g = geodesic_metric.CPn_distance_matrix(X_geo)
D_out_c = chordal_metric.CPn_chordal_distance_matrix(X_chrd)

# %% codecell
PH_geo = ripser(D_out_g, distance_matrix=True, maxdim=2)
plot_diagrams(PH_out['dgms'])
plt.title('Persistence with Squared Geodesic Metric')
plt.show()

# %% codecell
PH_chrd = ripser(D_out_c, distance_matrix=True, maxdim=2)
plot_diagrams(PH_out['dgms'])
plt.title('Persistence with Chordal Metric')
plt.show()

# %% codecell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis = cplx_projective.hopf(X_geo)
ax.scatter(vis[0,:], vis[1,:], vis[2,:], c=c[sub_ind], cmap='cividis')
ax.set_title('Output of Squared Geodesic Metric')
plt.show()

# %% codecell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis = cplx_projective.hopf(X_chrd)
ax.scatter(vis[0,:], vis[1,:], vis[2,:], c=c[sub_ind], cmap='cividis')
ax.set_title('Output of Chordal Metric')
plt.show()

# %% codecell
def riemann_sphere(X):
    """X : ndarray of column vectors."""
    X_cplx = cplx_projective.complexify(X)
    RS = X_cplx[0,:] / X_cplx[1,:]
    RS = cplx_projective.realify(RS)
    return RS

# %% codecell
RS = riemann_sphere(X_geo)
fig, ax = plt.subplots()
ax.scatter(RS[0,:], RS[1,:], c=c[sub_ind], cmap='cividis')
ax.set_title('Riemann Sphere projection of Squared Geodesic')
plt.show()


# %% codecell
RS_chrd = riemann_sphere(X_chrd)
fig, ax = plt.subplots()
ax.scatter(RS_chrd[0,:], RS_chrd[1,:], c=c[sub_ind], cmap='cividis')
ax.set_title('Riemann Sphere projection of Chordal')
plt.show()

# %% codecell
E_geo = D_out_g - D_geo
E_chrd = D_out_c - D_chrd
print(np.mean(E_geo**2))
print(np.mean(E_chrd**2))
print(np.var(E_geo**2))
print(np.var(E_chrd**2))

# %% codecell
# TODO: fill in the full projective coordinates pipeline (requires lifting)
# Compute projective coordinates, using a prominent cocycle in dimension 2.
# cocycles = PH['cocycles'][2]
# diagram = PH['dgm'][2]
# part_func = pipeline.partition_unity(D, (death-birth)/2, sub_ind)
# eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram,
    # threshold_at_death=False)
# TODO: apply a Bockstein lift here.
# TODO: use the harmonic cocycle.
# TODO: implement complex projective coordinates (or pull from some existing
# code).
#proj_coors =
# Check persistence of projective coordinates.
# Construct geodesic distance matrix of projective coordinates.
# Apply PCA
# Apply MDS.
