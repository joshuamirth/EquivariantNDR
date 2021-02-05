# Script for constructing data on the projective plane.

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
from numpy.random import default_rng
from ppca import ppca

# Uniformly sample points from the sphere.
rng = default_rng(57)
N = 1000     # Number of points to sample.
n_landmarks = 100
RP = rng.standard_normal((3,N))
RP = RP/np.linalg.norm(RP, axis=0)
RP[:,2] = np.abs(RP[:,2])   # Set all z-coordinates to positive.
D = real_projective.projective_distance_matrix(RP.T)

# Plot the data. This is a projection onto the first two coordinates.
# TODO:Make a nice plot of this. Should be properly square, and color-coded to
# show distance from origin for comparison with final result.
plt.scatter(RP[0,:], RP[1,:])
plt.show()

# Compute persistence.
idx = pipeline.maxmin_subsample_distance_matrix(D, n_landmarks)['indices']
D_sub = D[idx,:][:,idx]
RP_sub = RP[:,idx]
PH_sub = ripser(D_sub, coeff=2, do_cocycles=True, distance_matrix=True)
plot_diagrams(PH_sub['dgms'])
plt.show()

cocycles = PH_sub['cocycles'][1]
diagram = PH_sub['dgms'][1]
eta, birth, death = pipeline.prominent_cocycle(cocycles, diagram,
    threshold_at_death=False)

# Get a partition of unity.
part_func = pipeline.partition_unity(D, .25, sub_ind, bump_type='quadratic')
proj_coords = pipeline.proj_coordinates(part_func, eta)
D_pc = real_projective.projective_distance_matrix(proj_coords.T)
D_geo = real_projective.geo_distance_matrix(D_pc, k=8)
# Compute PH of landmarks of high-dimensional data.
PH_pc = ripser(D_geo, distance_matrix=True, maxdim=2, coeff=2)
plot_diagrams(PH_pc['dgms'])
plt.show()
