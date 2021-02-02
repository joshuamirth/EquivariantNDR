import numpy as np
#import ripser
#from persim import plot_diagrams
import time

# WARNING: this script is very memory intensive!

# TODO: we really want this in R^6 so it lives on CP^2.

# First construct a trefoil knot in R^3
# This can be changed to any simple closed curve we like, but it's a better
# demo if there's something interesting going on.
N = 12  # Must be even!
s = np.linspace(-1, 1, N)
t = np.linspace(0, 2*np.pi, N)
x = np.sin(t) + 2*np.sin(2*t)
y = np.cos(t) - 2*np.cos(2*t)
z = -np.sin(3*t)

# Now build the coordinates in R^4 
xyz = np.column_stack((x, y, z))
xxyyzz = np.vstack(int(N/2)*(xyz, xyz))
ss = np.column_stack(int(N/2)*(s, s))
ss = np.reshape(ss, (N**2, 1))
knot = np.hstack((xxyyzz, ss)) # Product of curve and unit interval in R^4

# Do the quotient to turn this into a suspension
# In principle I want these:
#u_mean = (1/N)*np.sum(R[(N-1)*N:N**2,:], axis=0)
#l_mean = (1/N)*np.sum(R[0:N,:], axis=0)
# But it looks like they are exactly [0, -1, 0, +/- 1]
mean_point = np.array([0, -1, 0])
for i in range(int(N**2/2)):
    knot[i,0:3] = (1 + knot[i,3]) * knot[i,0:3] - knot[i,3]*mean_point
for i in range(int(N**2/2), N**2):
    knot[i,0:3] = (1 - knot[i,3]) * knot[i,0:3] + knot[i,3]*mean_point

filename = 'knot_test_%d.npz' %N**2
np.savez(filename, knot=knot)
# Do a ripser computation to check that there is some 2-dimensional homology,
# i.e. that we have a 2-sphere.
#print('Done generating 2-knot.')
#start_time = time.time()
#dgms = ripser.ripser(knot, maxdim=2)['dgms']
#end_time = time.time()
#print('Running time for persistence with %d points: %5.2f seconds.' %(N, end_time - start_time))
#plot_diagrams(dgms, show=True)
#np.savetxt('sphere_test.pers', dgms[2])
