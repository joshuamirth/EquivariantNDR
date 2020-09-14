# Hacked together script to do dim reduction on RPn
# import matlab.engine    # for LRCM MIN.
import numpy as np
# import scipy.io as io
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import floyd_warshall

import ppca


def circle_RPn(dimn=4,segment_points=100,num_segments=2):
    """Construct points on a "kinked" circle in RP^d.

    Constructs a curve of evenly-spaced points along the great circle
    from e_i to e_{i+1} in R^{d+1}, starting at e_0 and finishing at
    e_i with i = num_segments, then returns to -e_0.

    Parameters
    ----------
    dimn : int, optional
        Dimension of RP^d to work on (ambient euclidean space is dimn+1).
    segment_points : int, optional
        Number of points along each segment of curve.
    num_segments : int, optional
        Number of turns to make before returning to start point.

    Returns
    -------
    n : int
        Number of points in the resulting curve, which is
        segment_curve*(num_segments+1).
    X : ndarray
        Array of coordinate values in R^{d+1}.
    """
    
    if dimn < num_segments:
        raise ValueError('Dimension ' + str(dimn) + ' does not have enough ' +
            'dimensions for ' + str(num_segments) + ' segments.')
    num_points = segment_points*(num_segments+1)
    theta = np.linspace(0,np.pi/2,segment_points,endpoint=False)
    X = np.zeros((num_points,dimn+1))
    segment_curve = np.array([np.cos(theta),np.sin(theta)]).transpose()
    for i in range(0,num_segments):
        X[i*segment_points:(i+1)*segment_points,i:i+2] = segment_curve
    X[num_segments*segment_points:num_points,0] = -np.sin(theta)
    X[num_segments*segment_points:num_points,num_segments] = np.cos(theta)
    return {'n': num_points, 'X': X}

def
