# Script for constructing data on the projective plane.

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import real_projective
from examples import pipeline   # Note: to make this work, install
            # package in editable mode, `pip install -e .` in root directory.
from numpy.random import default_rng
from ppca import ppca

# Uniformly sample points from the sphere.
rng = default_rng(57)
N = 1000     # Number of points to sample.
n_landmarks = 100
RP = rng.standard_normal((3,N))
RP = RP/np.linalg.norm(RP, axis=0)
# Place all points onto their upper hemisphere representative.
neg_idx = np.where(RP[2,:] < 0)
RP[:,neg_idx] = -RP[:,neg_idx]
D = real_projective.projective_distance_matrix(RP.T)

# Plot the data. This is a projection onto the first two coordinates.
c = np.linalg.norm(RP[0:2,:],axis=0)
fig, ax = plt.subplots()
ax.scatter(RP[0,:], RP[1,:], c=c, cmap='YlGn')
ax.axis('equal')
plt.show()

# Compute persistence.
# For consistency, be sure to seed maxim with seed=57.
idx = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks, seed=[57])['indices']
D_sub = D[idx,:][:,idx]
RP_sub = RP[:,idx]
PH_sub = ripser(D_sub, coeff=2, do_cocycles=True, distance_matrix=True)
plot_diagrams(PH_sub['dgms'])
plt.show()

#diagram = np.loadtxt('rp2_diagram.txt', skiprows=1, delimiter=',')
#cocycles = np.loadtxt('rp2_diagram.txt', skiprows=1, delimiter=',')
#cocycles = PH_sub['cocycles'][1]
#diagram = PH_sub['dgms'][1]
#eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram,
#    threshold_at_death=False)

# Something weird is going on with ripser producing cocycles here. I saved the
# distance matrix and did this computation in ripser manually. This loads that
# output instead of recomputing it. (The issue may have something to do with the
# length of this cocyle...)
# np.savetxt('RP2_distance_matrix.csv', D_sub, delimiter=',')
# Manually, run this distance matrix through ripser-representatives, then find
# the prominent cocycle in dimension one from that output and copy the
# representative below as `eta`.
# The current eta (beginning [32,55]) corresponds to choosing N=1000,
# n_landmarks=100, default_rng(57), and maxmin seeded with seed=[57]. Adjusting
# any of these parameters will change eta, birth, and death.
eta = np.array([[32,55], [10,64], [17,61], [36,57], [57,61], [22,48], [39,82], [14,56], [11,48], [56,73], [11,78], [11,39], [10,36], [28,43], [6,39], [4,43], [48,84], [43,61], [6,89], [6,35], [6,55], [10,55], [13,39], [6,79], [39,47], [28,56], [0,73], [6,78], [12,28], [55,82], [57,64], [27,28], [56,100], [17,36], [36,98], [55,98], [48,73], [64,98], [11,93], [0,22], [6,21], [78,82], [22,93], [39,88], [11,21], [32,89], [10,18], [4,17], [55,88], [6,18], [10,61], [33,61], [10,40], [39,94], [4,57], [18,32], [61,98], [48,52], [32,39], [57,66], [43,100], [47,78], [0,14], [46,48], [22,56], [14,27], [6,63], [82,89], [73,93], [84,93], [61,87], [27,100], [56,75], [43,66], [26,55], [27,73], [13,78], [10,66], [6,85], [10,20], [36,43], [41,56], [6,44], [32,64], [78,84], [35,82], [39,46], [10,89], [28,80], [55,57], [17,66], [49,56], [41,43], [47,48], [39,53], [0,84], [79,82], [32,35], [26,36], [39,71], [4,12], [26,61], [17,64], [73,83], [12,100], [10,74], [39,84], [4,33], [20,57], [18,82], [61,99], [21,82], [18,98], [28,38], [0,11], [28,33], [46,78], [12,61], [13,55], [10,91], [14,48], [28,83], [61,80], [26,64], [22,78], [0,28], [4,10], [32,78], [0,100], [50,73], [6,8], [48,49], [32,36], [61,69], [78,88], [26,39], [14,83], [24,39], [32,63], [66,98], [78,94], [36,87], [6,93], [21,84], [21,22], [6,74], [11,79], [40,98], [89,98], [40,57], [22,50], [36,99], [52,93], [14,43], [0,49], [46,93], [88,89], [48,71], [11,50], [32,74], [32,79], [0,52], [32,40], [14,93], [4,80], [4,98], [17,28], [21,47], [0,75], [5,48], [27,41], [22,83], [4,56], [71,78], [10,63], [33,36], [47,93], [12,14], [20,98], [63,82], [18,88], [56,84], [18,57], [64,88], [6,96], [27,75], [28,51], [2,61], [53,55], [83,100], [4,87], [55,99], [6,50], [22,39], [22,27], [36,88], [4,27], [6,34], [6,48], [52,56], [39,98], [13,21], [39,99], [12,41], [57,91], [21,73], [43,60], [43,75], [19,28], [14,50], [17,20], [48,75], [10,35], [11,44], [64,82], [21,32], [18,26], [52,78], [10,60], [57,60], [74,98], [32,61], [39,90], [64,99], [20,43], [21,46], [35,88], [50,84], [7,56], [43,64], [61,88], [13,89], [6,40], [26,89], [53,78], [82,93], [44,82], [1,10], [49,93], [6,64], [13,48], [73,78], [0,46], [39,52], [48,94], [33,66], [0,41], [48,82], [82,85], [80,100], [53,61], [10,34], [19,73], [41,57], [16,61], [28,57], [28,92], [13,79], [10,42], [17,55], [38,100], [37,48], [28,50], [91,98], [33,100], [74,82], [39,58], [17,100], [56,60], [64,87], [12,66], [27,49], [79,88], [17,41], [24,78], [4,26], [48,100], [10,39], [21,88], [13,35], [57,89], [12,73], [48,58], [36,69], [4,69], [57,74], [11,56], [11,83], [4,99], [26,66], [35,98], [63,98], [36,53], [11,35], [32,34], [32,85], [56,66], [14,38], [47,79], [14,19], [66,87], [26,78], [36,82], [6,67], [2,36], [71,93], [12,36], [21,52], [16,28], [13,93], [2,39], [57,100], [83,84], [43,73], [40,82], [5,39], [93,100], [55,94], [13,18], [6,31], [27,61], [75,93], [20,32], [21,94], [36,80], [17,60], [12,75], [26,40], [50,100], [28,93], [24,48], [75,83], [47,55], [51,61], [8,32], [63,88], [28,87], [32,66], [8,82], [39,77], [32,91], [38,73], [10,41], [53,64], [32,44], [37,56], [22,44], [55,87], [28,48], [56,61], [21,71], [0,6], [19,100], [5,93], [4,38], [0,47], [19,22], [40,88], [5,78], [33,41], [42,57], [44,73], [79,84], [17,40], [28,69], [66,99], [22,79], [6,19], [66,80], [54,61], [49,83], [41,83], [93,94], [78,98], [26,35], [58,78], [60,98], [74,88], [4,51], [20,26], [11,55], [61,90], [11,85], [4,16], [50,52], [50,82], [33,64], [13,64], [48,95], [41,48], [27,66], [7,48], [34,82], [11,89], [41,80], [39,73], [14,21], [27,84], [6,83], [7,43], [0,7], [10,79], [28,68], [14,80], [38,61], [46,56], [53,89], [13,36], [2,4], [44,84], [0,37], [0,5], [10,70], [6,36], [51,100], [2,55], [17,91], [26,74], [61,82], [46,50], [3,28], [43,49], [18,99], [11,96], [39,57], [49,50], [48,88], [78,99], [35,47], [78,90], [17,18], [11,19], [4,32], [39,87], [47,50], [34,98], [57,63], [13,61], [52,83], [82,96], [1,32], [79,98], [18,53], [10,78], [32,93], [50,75], [8,10], [27,60], [6,70], [89,99], [26,63], [6,76], [10,59], [1,57], [1,6], [2,64], [27,52], [73,79], [47,89], [1,98], [56,86], [0,4], [79,94], [12,60], [66,88], [14,33], [46,79], [88,93], [73,96], [7,27], [42,98], [13,63], [38,41], [21,53], [44,47], [66,69], [37,93], [4,83], [13,44], [14,78], [11,27], [26,79], [41,98], [10,86], [61,68], [10,28], [0,71], [10,100], [32,48], [43,91], [92,100], [5,56], [41,93], [4,88], [20,56], [64,69], [6,91], [55,90], [49,78], [57,75], [35,57], [48,77], [48,53], [7,10], [10,85], [28,96], [58,93], [20,33], [28,98], [85,88], [20,87], [26,91], [89,94], [44,88], [14,17], [21,26], [21,24], [21,49], [7,57], [14,92], [20,88], [35,53], [29,48], [16,36], [22,96], [17,75], [36,90], [24,93], [0,82], [20,99], [14,44], [77,78], [4,53], [12,49], [42,43], [19,84], [36,56], [53,79], [0,60], [5,21], [73,92], [39,54], [40,99], [14,51], [34,57], [13,85], [32,96], [87,100], [32,67], [35,94], [41,50], [28,31], [12,20], [32,70], [24,55], [38,75], [29,39], [33,60], [21,98], [40,43], [2,66], [28,44], [28,99]])

birth = 0.244596
death = 1.05546
print('Birth: %2.4f' %birth)
print('Death: %2.4f' %death)# Get a partition of unity.

print('Computing partition of unity.')
# Note that the second parameter depends on the value of birth. It should be
# between birth and death. Usually birth+delta works fine.
part_func = pipeline.partition_unity(D, .26, idx, bump_type='quadratic')
print('Finding projective coordinates.')
proj_coords = pipeline.proj_coordinates(part_func, eta)
print('Computing distance matrix of projective coordinates.')
D_pc = real_projective.projective_distance_matrix(proj_coords.T)
print('Estimating geodesic distance matrix.')
D_geo = pipeline.geo_distance_matrix(D_pc, k=8)
# Compute PH of landmarks of high-dimensional data.
print('Computing persistence of projective coordinates.')
PH_pc = ripser(D_geo[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_pc['dgms'])
plt.show()

# Reduce dimension using PCA:
X_ppca = ppca(proj_coords.T, 2)['X']
# Compute persistence of PCA output.
D_ppca = real_projective.projective_distance_matrix(X_ppca)
PH_ppca = ripser(D_ppca[idx,:][:,idx], distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_ppca['dgms'])
plt.show()

# Apply MDS to PCA output.
X_mds = real_projective.pmds(X_ppca, D_geo, max_iter=100)

# Compute persistence of MDS output.
D_mds = real_projective.projective_distance_matrix(X_mds)
PH_mds = ripser(D_mds, distance_matrix=True, maxdim=1, coeff=2)
plot_diagrams(PH_mds['dgms'])
plt.show()

# Plot the PPCA and MDS embeddings.
neg_idx = np.where(X_ppca[:,2] < 0)
X_ppca[neg_idx,:] *= -1
fig, ax = plt.subplots()
ax.scatter(X_ppca[:,0], X_ppca[:,1], c=c, cmap='YlGn')
ax.axis('equal')
plt.show()

neg_idx = np.where(X_mds[:,2] < 0)
X_mds[neg_idx,:] *= -1
fig, ax = plt.subplots()
ax.scatter(X_mds[:,0], X_mds[:,1], c=c, cmap='YlGn')
ax.axis('equal')
plt.show()

# Save the MDS and PCA outputs.
np.savez('rp2_output.npz', D=D, D_mds=D_mds, D_ppca=D_ppca, D_geo=D_geo,
    X_ppca=X_ppca, X_mds=X_mds, RP=RP)
