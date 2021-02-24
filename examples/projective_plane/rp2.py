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
idx = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks)['indices']
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
eta = np.array([[15,65], [24,50], [20,50], [20,82], [26,61], [12,61], [17,45], [61,67], [15,46], [50,53], [1,53], [19,55], [12,44], [46,53], [14,36], [15,68], [2,36], [50,71], [50,76], [6,34], [19,37], [19,45], [15,34], [36,45], [34,84], [45,75], [34,74], [45,52], [53,82], [55,74], [34,72], [14,26], [12,92], [35,36], [46,84], [20,49], [36,61], [50,84], [12,20], [65,84], [19,34], [71,82], [15,31], [24,82], [6,55], [53,68], [48,50], [14,75], [1,15], [2,75], [45,63], [53,54], [9,15], [61,99], [50,72], [61,75], [14,67], [68,84], [12,14], [44,67], [76,82], [1,84], [37,74], [2,19], [1,20], [53,65], [1,24], [8,50], [34,80], [67,92], [2,17], [44,82], [59,61], [45,59], [34,53], [44,49], [49,53], [26,92], [15,54], [36,56], [9,19], [15,16], [51,61], [24,34], [37,52], [15,79], [48,82], [45,74], [15,50], [15,55], [24,46], [52,55], [36,42], [82,84], [36,37], [53,91], [36,92], [19,56], [32,34], [4,12], [36,94], [43,50], [44,99], [65,72], [44,50], [45,90], [9,74], [17,37], [26,44], [55,84], [6,65], [14,59], [16,53], [32,55], [26,45], [49,61], [55,72], [15,82], [12,48], [9,84], [2,52], [6,9], [34,43], [7,45], [15,88], [46,72], [12,42], [20,99], [35,75], [50,64], [55,80], [1,76], [32,45], [31,84], [2,26], [54,84], [36,95], [1,71], [65,74], [37,75], [49,71], [34,52], [24,65], [18,61], [6,37], [20,91], [19,88], [31,53], [5,61], [72,82], [92,99], [26,42], [26,35], [14,17], [24,49], [1,72], [20,67], [49,92], [15,91], [19,65], [24,68], [2,59], [8,82], [48,49], [12,71], [53,79], [50,86], [68,72], [15,58], [4,50], [39,45], [82,92], [75,92], [9,72], [11,36], [32,37], [4,82], [16,84], [34,76], [56,75], [17,61], [6,50], [14,99], [50,77], [18,20], [36,44], [61,82], [24,54], [41,53], [17,55], [61,90], [42,67], [14,51], [2,74], [44,51], [17,56], [1,44], [50,80], [18,44], [9,53], [15,49], [2,63], [49,76], [79,84], [5,45], [49,84], [18,53], [46,76], [20,54], [37,63], [20,46], [45,87], [19,25], [42,75], [64,82], [6,45], [56,74], [25,36], [4,67], [52,56], [84,91], [36,78], [74,88], [15,41], [15,89], [51,92], [6,68], [15,37], [50,92], [1,48], [53,55], [44,91], [6,46], [59,92], [4,49], [24,91], [17,35], [36,55], [24,55], [61,91], [43,82], [9,52], [84,88], [53,89], [5,14], [14,90], [4,26], [4,36], [3,15], [53,99], [65,80], [2,90], [75,94], [43,55], [55,63], [12,35], [31,72], [35,67], [37,80], [12,95], [12,53], [50,85], [7,55], [23,34], [44,75], [30,61], [6,88], [45,70], [6,31], [12,85], [45,67], [12,64], [45,80], [35,59], [9,32], [9,80], [14,49], [34,70], [26,95], [7,37], [46,71], [9,24], [20,51], [18,92], [37,72], [23,50], [19,31], [37,84], [15,25], [55,75], [44,59], [48,67], [20,26], [19,35], [61,93], [37,59], [15,38], [54,72], [19,94], [16,24], [7,34], [50,61], [2,67], [31,74], [19,58], [24,31], [4,99], [12,76], [34,86], [36,88], [48,99], [68,74], [65,76], [1,61], [21,50], [1,8], [71,91], [75,95], [32,65], [55,70], [15,18], [26,94], [1,6], [26,56], [45,93], [43,65], [58,84], [77,82], [49,72], [53,88], [54,76], [2,32], [40,50], [1,43], [82,85], [26,37], [12,24], [6,82], [82,86], [14,18], [71,99], [61,63], [42,59], [14,63], [68,76], [9,36], [6,56], [41,84], [72,88], [15,56], [76,91], [25,74], [17,34], [2,7], [28,36], [8,34], [43,46], [54,71], [67,95], [1,92], [20,41], [42,99], [8,49], [37,90], [39,61], [17,94], [45,51], [91,92], [52,88], [19,68], [2,5], [14,82], [16,72], [44,54], [49,64], [2,6], [17,92], [19,78], [34,63], [34,71], [5,92], [56,59], [53,58], [33,45], [24,79], [46,74], [12,28], [50,74], [52,65], [56,63], [23,55], [9,17], [42,49], [14,19], [42,82], [80,82], [84,89], [72,79], [46,80], [11,75], [48,91], [67,71], [8,12], [14,52], [5,44], [1,4], [18,24], [41,61], [55,76], [2,39], [17,42], [72,91], [18,84], [58,74], [38,53], [25,75], [11,26], [26,48], [35,52], [32,50], [2,12], [18,71], [32,56], [37,70], [68,80], [24,99], [9,43], [4,75], [42,50], [54,61], [20,68], [36,85], [24,41], [45,72], [18,48], [25,52], [29,36], [17,25], [3,53], [1,64], [42,51], [6,58], [15,99], [36,66], [90,92], [1,80], [14,39], [75,78], [61,83], [16,20], [6,79], [14,93], [36,58], [22,45], [43,68], [51,53], [53,67], [17,88], [21,82], [10,15], [6,25], [23,45], [37,39], [46,48], [8,46], [41,44], [12,45], [11,12], [25,84], [3,84], [4,51], [34,81], [32,88], [37,87], [44,46], [49,85], [37,43], [35,90], [30,44], [28,50], [14,30], [76,99], [9,75], [52,61], [10,53], [4,18], [50,96], [59,94], [12,84], [56,84], [37,53], [67,85], [55,59], [55,87], [24,88], [35,63], [12,94], [84,99], [6,54], [2,51], [52,94], [50,81], [48,54], [2,87], [75,88], [14,91], [80,88], [55,90], [68,71], [14,50]])

birth = 0.266739
death = 1.05087
print('Birth: %2.4f' %birth)
print('Death: %2.4f' %death)

# Get a partition of unity.
print('Computing partition of unity.')
part_func = pipeline.partition_unity(D, .28, idx, bump_type='quadratic')
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
