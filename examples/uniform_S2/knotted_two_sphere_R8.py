import numpy as np
#import ripser
#from persim import plot_diagrams
import time


# First construct a trefoil knot in R^7
# This can be changed to any simple closed curve we like, but it's a better
# demo if there's something interesting going on.
N = 16  # Must be even!
s = np.linspace(-1, 1, N)
t = np.linspace(0, 2*np.pi, N)
x1 = np.sin(t) + 2*np.sin(2*t)
x2 = np.cos(t) - 2*np.cos(2*t)
x3 = -np.sin(3*t)
x4 = np.cos(t)
x5 = np.sin(t)
x6 = np.sin(t)
x7 = np.cos(t)

# Now build the coordinates in R^8 
xyz = np.column_stack((x1, x2, x3, x4, x5, x6, x7))
xxyyzz = np.vstack(int(N/2)*(xyz, xyz))
ss = np.column_stack(int(N/2)*(s, s))
ss = np.reshape(ss, (N**2, 1))
knot = np.hstack((xxyyzz, ss)) # Product of curve and unit interval in R^4

# Do the quotient to turn this into a suspension
# In principle I want these:
u_mean = (1/N)*np.sum(knot[(N-1)*N:N**2,0:7], axis=0)
l_mean = (1/N)*np.sum(knot[0:N,0:7], axis=0)
# But it looks like they are exactly [0, -1, 0, +/- 1]
# mean_point = np.array([0, -1, 0])
for i in range(int(N**2/2)):
    knot[i,0:7] = (1 + knot[i,7]) * knot[i,0:7] - knot[i,7]*l_mean
for i in range(int(N**2/2), N**2):
    knot[i,0:7] = (1 - knot[i,7]) * knot[i,0:7] + knot[i,7]*u_mean

# Remove duplicate points in the quotient:
knot = knot[N-1:N**2-N,:]

filename = 'knot_R8_N%d.npz' %N**2
np.savez(filename, knot=knot)
# Do a ripser computation to check that there is some 2-dimensional homology,
# i.e. that we have a 2-sphere.
# WARNING: this is very memory intensive!
#print('Done generating 2-knot.')
#start_time = time.time()
#dgms = ripser.ripser(knot, maxdim=2)['dgms']
#end_time = time.time()
#print('Running time for persistence with %d points: %5.2f seconds.' %(N, end_time - start_time))
#plot_diagrams(dgms, show=True)
#np.savetxt('sphere_test.pers', dgms[2])
