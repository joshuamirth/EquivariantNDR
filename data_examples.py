"""Generate various example datasets for dimensionality reduction testing.

    All methods return data in the form of a numpy array, X, where each
    column corresponds to a single data point.

"""

import numpy as np
import random

def circleRPn(
    dim=4,
    segment_points=50,
    num_segments=4,
    noise=False,
    v=0.2,
    randomize=True
):
    """Construct points on a "kinked" circle in RP^d.

    Constructs a curve of evenly-spaced points along the great circle
    from e_i to e_{i+1} in R^{d+1}, starting at e_0 and finishing at
    e_i with i = num_segments, then returns to -e_0.

    It is recommended that dim==num_segments, otherwise the resulting
    data matrix will not be full rank, which can cause issues later.
    Similarly, the output data is randomly permuted so that the first n
    points are not on the same linear subspace, generically.

    Parameters
    ----------
    dim : int, optional
        Dimension of RP^d to work on (ambient euclidean space is dim+1).
    segment_points : int, optional
        Number of points along each segment of curve.
    num_segments : int, optional
        Number of turns to make before returning to start point.

    Returns
    -------
    X : ndarray
        Array of coordinate values in R^{d+1}.
    """
    if int(num_segments) != num_segments:
        raise ValueError("""Number of segments must be a positive integer.
            Supplied value was %2.2f.""" %num_segments)
    if num_segments < 1 or dim < 1:
        raise ValueError("""Number of segments and dimension must be positive
            integers. Supplied values were num_segments = %2.2f and dimension
            = %2.2f""" %(num_segments,dim))    
    if dim < num_segments:
        raise ValueError("""Value of dimension must be larger than number of
            segments. Supplied dimension was %i and number of segments was
            %i""" %(dim,num_segments))
    rng = np.random.default_rng(57)
    num_points = segment_points*(num_segments+1)
    theta = np.linspace(0,np.pi/2,segment_points,endpoint=False)
    X = np.zeros((num_points,dim+1))
    segment_curve = np.array([np.cos(theta),np.sin(theta)]).T
    for i in range(0,num_segments):
        X[i*segment_points:(i+1)*segment_points,i:i+2] = segment_curve
    X[num_segments*segment_points:num_points,0] = -np.sin(theta)
    X[num_segments*segment_points:num_points,num_segments] = np.cos(theta)
    if randomize:
        X = rng.permutation(X)
        X = X.T
    if noise:
        N = v*rng.random((dim+1,num_points))
        X = (X.T + N)/LA.norm(X.T+N,axis=0)
    return X

def bezier_RPn(ctrl_points,N=100,noise=0):
    """Define a weird curve for testing purposes.
    
    Parameters
    ----------
    ctrl_points : ndarray
        n*d array where each row is a control point of a Bezier curve
        in R^d. The first row is the start point of the curve, and the
        last row is the end point.
    N : int, optional
        Number of points to put on curve. Default is 1000.
    
    Returns
    -------
    B : ndarray
        Array (N*d) with each row a point on the curve. Normalized to
        lie on the sphere.

    """

    t = np.reshape(np.linspace(0,1,N),(N,1))
    deg = ctrl_points.shape[0]-1
    dim = ctrl_points.shape[1]
    P = np.reshape(ctrl_points[0,:],(1,dim))
    B = ((1-t)**deg)@P
    for i in range(1,deg):
        P = np.reshape(ctrl_points[i,:],(1,dim))
        B = B + comb(deg,i)*((t**i)*((1-t)**(deg-i)))@P
    P = np.reshape(ctrl_points[deg,:],(1,dim))
    B = B + (t**deg)@P
    if noise > 0 :
        ns = noise*(np.random.rand(N,dim)-.5)
        B = B+ns
    B = (B.T/LA.norm(B,axis=1))
    return B   

def line_patches(dim, NAngles, NOffsets, sigma):
    """Sample a set of line segments, as witnessed by square patches.

    Parameters
    ----------
    dim: int
        Patches will be dim x dim.
    NAngles: int
        Number of angles to sweep between 0 and pi.
    NOffsets: int
        Number of offsets to sweep from the origin to the edge of the
        patch.
    sigma: float
        The blur parameter.  Higher sigma is more blur.

    """

    N = NAngles*NOffsets
    P = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles+1)[0:NAngles]
    ps = np.linspace(-1, 1, NOffsets)
    idx = 0
    [Y, X] = np.meshgrid(
        np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim)
        )
    for i in range(NAngles):
        c = np.cos(thetas[i])
        s = np.sin(thetas[i])
        for j in range(NOffsets):
            patch = X*c + Y*s + ps[j]
            patch = np.exp(-patch**2/sigma**2)
            P[idx, :] = patch.flatten()
            idx += 1
    return P

def crossed_line_patches(dim, NAngles, NOffsets, sigma):
    """
    Sample a set of line segments, as witnessed by square patches
    Parameters
    ----------
    dim: int
        Patches will be dim x dim
    NAngles: int
        Number of angles to sweep between 0 and pi
    NOffsets: int
        Number of offsets to sweep from the origin to the edge of the patch
    sigma: float
        The blur parameter.  Higher sigma is more blur
    """

    N = NAngles*NOffsets
    P = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles+1)[0:NAngles]
    ps = np.linspace(-1, 1, NOffsets)
    idx = 0
    [Y, X] = np.meshgrid(np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim))
    for i in range(NAngles):
        c = np.cos(thetas[i])
        s = np.sin(thetas[i])
        for j in range(NOffsets):
            patch = X*c + Y*s + ps[j]
            patch = np.exp(-patch**2/sigma**2)
            P[idx, :] = patch.flatten()
            idx += 1
    return P
