"""Compare PPCA and various MDS methods on the example of "bent"
high-dimensional data on RP^5."""

# A whole bunch of import statements.
# %% codecell
import autograd.numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import geodesic_metric
import chordal_metric

# Import the data and list what is in it.
# %% codecell
stuff = np.load('examples/bezier_curve/bezier_curve_data.npz')
for i in stuff.keys():
    print(i)

# %% codecell
B = stuff['B']
D = stuff['D']
D_geo = stuff['D_geo']
X_ppca = stuff['X_ppca']
X_mds = stuff['X_mds']

# %% markdown
# Note that this `X_mds` was obtained a long time ago using the alternating
# update method. It is essentially the goal output.

# %% codecell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_ppca[:,0], X_ppca[:,1], X_ppca[:,2])
ax.scatter(-X_ppca[:,0], -X_ppca[:,1], -X_ppca[:,2])
ax.set_title('PPCA Output')
plt.show()
# This will plot the PPCA embedding. Note that it is very "wiggly."

# %% markdown
# First we will run the chordal metric version of MDS. This does not work well.
# %% codecell
D_goal = np.sin(D_geo)
X_out = chordal_metric.rp_mds(D_goal)

# %% codecell
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_out[0,:], X_out[1,:], X_out[2,:])
ax.scatter(-X_out[0,:], -X_out[1,:], -X_out[2,:])
ax.set_title('Chordal Metric Output')
plt.show()

# %% markdown
# Other initial conditions can be used, but none of them really give a "nice"
# output. They are all a wiggly circle with the wrong radius.

# %% markdown
# # Squared Distances
# Now perform the same experiment using the squared distances cost function.

# %%codecell
X_square = geodesic_metric.rp_mds(D_geo, X=X_ppca.T)

# %% codecell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_square[0,:], X_square[1,:], X_square[2,:])
ax.scatter(-X_square[0,:], -X_square[1,:], -X_square[2,:])
ax.view_init(30, -20) #elev, azim pair. Default 30, -60.
ax.set_title('Squared Geodesic MDS Output')
plt.show()

# %% codecell
# For reference, here is the version from the alternating MDS method:
X_mds = X_mds.T

# %% codecell
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_mds[0,:], X_mds[1,:], X_mds[2,:])
ax.scatter(-X_mds[0,:], -X_mds[1,:], -X_mds[2,:])
ax.view_init(30, 0) #elev, azim pair. Default 30, -60.
ax.set_title('Alternating MDS Output')
plt.show()

# %% codecell
D_square = real_projective.projective_distance_matrix(X_square.T)
D_diff = D_geo - D_square
plt.imshow(D_diff, cmap='cividis')
plt.title('Squared Cost Difference')
plt.show()
print(np.linalg.norm(D_geo - D_square))

# %% codecell
D_out = chordal_metric.RPn_chordal_distance_matrix(X_out)
D_diff = D_goal - D_out
plt.imshow(D_diff, cmap='cividis')
plt.title('Chordal Metric Difference')
plt.show()
print(np.linalg.norm(D_goal - D_out))

# This shows that not only does the squared version of the geodesic distance
# perform better than the chordal distance, it also gets closer to its goal
# matrix.

# Let's see what the persistence looks like.

# %% codecell
PH = ripser(D_out, distance_matrix=True, coeff=2, maxdim=1)
plot_diagrams(PH['dgms'])
plt.title('Persistence of Chordal Metric MDS')
plt.show()

# %% markdown
# Obviously this is homologically correct, but it is hard to justify the radius
# of that H^1 class.

# %% markdown
# By way of comparison, here is the persistence of the MDS (square) version.

# %% codecell

PH_square = ripser(D_square, distance_matrix=True, coeff=2, maxdim=1)
plot_diagrams(PH_square['dgms'])
plt.title('Persistence of Squared Cost MDS.')
plt.show()

# %% markdown
# By way of comparison, here are the PPCA and old MDS version:

# %% codecell
D_ppca = real_projective.projective_distance_matrix(X_ppca)
PH_ppca = ripser(D_ppca, distance_matrix=True, maxdim=1)
plot_diagrams(PH_ppca['dgms'])
plt.title('Persistence of PPCA.')
plt.show()

# %% codecell
D_mds = real_projective.projective_distance_matrix(X_mds.T)
PH_mds = ripser(D_mds, distance_matrix=True, maxdim=1)
plot_diagrams(PH_mds['dgms'])
plt.title('Persistence of Alternating MDS.')
plt.show()


# %% codecell
PH_geo = ripser(D_geo, distance_matrix=True, maxdim=1)
plot_diagrams(PH_geo['dgms'])
plt.show()

# As a final point of comparison, let's test the squared cost with a random initial condition.
# %% codecell
rng = np.random.default_rng()
X_rand = rng.standard_normal((3, 500))
X_rand = X_rand/np.linalg.norm(X_rand, axis=0)
X_mds_r = geodesic_metric.rp_mds(D, X=X_rand)


# %% codecell
def align_representatives(X):
    """Place all points in the upper hemisphere."""
    neg_idx = np.where(X[2,:] < 0)
    X[:,neg_idx] = -X[:,neg_idx]
    return X

# %% codecell
X_mds_r = align_representatives(X_mds_r)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_mds_r[0,:], X_mds_r[1,:], X_mds_r[2,:])
ax.scatter(-X_mds_r[0,:], -X_mds_r[1,:], -X_mds_r[2,:])
ax.view_init(30, -50) #elev, azim pair. Default 30, -60.
ax.set_title('Squared Cost with Random IC')
plt.show()
# These are now not very good. Were they better before?

# %% codecell
# OLD: deeper analysis of the chordal metric result.
# # So I think this might have more to do with the chordal metric looking a little odd than it does with the optimization.
# # How different are the three results?
# # %% codecell
# print(np.linalg.norm(D_out - D_out_2))
# print(np.linalg.norm(D_out - D_out_3))
# print(np.linalg.norm(D_out_3 - D_out_2))
# # %% codecell
# print(np.linalg.norm(D_out - D_goal)/np.linalg.norm(D_goal))
# print(np.linalg.norm(D_mds - D_geo)/np.linalg.norm(D_geo))
# print(np.linalg.norm(D_geo - D_ppca)/np.linalg.norm(D_geo))
# # What does the persistence of the geodesic metric on the output points look like?
# # %% codecell
# D_out_geo = real_projective.projective_distance_matrix(X_out.T)
# # %% codecell
# PH_out_geo = ripser(D_out_geo, distance_matrix=True, maxdim=1)
# plot_diagrams(PH_out_geo['dgms'])
# plt.show()
# # %% codecell
# print(np.linalg.norm(D_out_geo - D_geo)/np.linalg.norm(D_geo))
# # I don't really understand how this can be this...okay?
# # ## Chordal the Whole Way
# # %% codecell
# D_c = chordal_metric.RPn_chordal_distance_matrix(B.T)
# # %% codecell
# PH_c = ripser(D_c, distance_matrix=True, maxdim=1)
# plot_diagrams(PH_c['dgms'])
# plt.show()
# # %% codecell
# X_c = chordal_metric.rp_mds(D_c)
# # %% codecell
# D_c = chordal_metric.RPn_chordal_distance_matrix(X_c)
# # %% codecell
# PH_c = ripser(D_c, distance_matrix=True, maxdim=1)
# plot_diagrams(PH_c['dgms'])
# plt.show()
# # %% codecell
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_c[0,:], X_c[1,:], X_c[2,:])
# ax.scatter(-X_c[0,:], -X_c[1,:], -X_c[2,:])
# plt.show()
# # %% codecell
# # Now estimate "geodesic" distances within the chordal manifold.
# D_c_geo = real_projective.geo_distance_matrix(D_c)
# # %% codecell
# PH_c_geo = ripser(D_c_geo, distance_matrix=True, maxdim=1)
# plot_diagrams(PH_c_geo['dgms'])
# plt.show()
# # %% codecell
# X_c = chordal_metric.rp_mds(D_c_geo)
# # %% codecell
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_c[0,:], X_c[1,:], X_c[2,:])
# ax.scatter(-X_c[0,:], -X_c[1,:], -X_c[2,:])
# plt.show()
# # %% codecell
# D_out_c_geo = chordal_metric.RPn_chordal_distance_matrix(X_c)
# PH_out_c_geo = ripser(D_out_c_geo, distance_matrix=True, maxdim=1)
# plot_diagrams(PH_out_c_geo['dgms'])
# plt.show()
