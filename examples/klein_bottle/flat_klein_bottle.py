# Script for constructing data on the flat model of the Klein bottle.
# Modified from code provided by Joe Melby.

import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist    # For some reason I have to
                                            # import this, instead of just
                                            # running it?

#-----------------------------------------------------------------------------#
# Functions copied from Joe which give the geometry of the flat Klein
# bottle.

def cls(x,y):
    '''
    Outputs the equivalence class of the points represented by (x,y) in
    the fundamental domain of K.
    '''
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

# Construct a uniform grid of points in the unit square.

numx = 20
numy = 20
N = numx*numy   # Total number of points
filename = 'flat_klein_bottle_N%d.npz' %N
print('Generating %d points on the flat klein bottle and saving the '\
    'results to ' %N + filename + '.' )
n_landmarks = 100

x, y = np.meshgrid(np.linspace(0,1,numx),np.linspace(0,1,numy))
xy = np.column_stack((x.ravel(),y.ravel()))

# Construct the distance matrix of the points.

D = pdist(xy, minDist)
D = sp.spatial.distance.squareform(D)

# TODO:
# Subsample the distance matrix with max/min.
# Save the output data.
# np.savez(filename, xy=xy, xy_sub = xy_sub, D=D, D_sub=D_sub)
