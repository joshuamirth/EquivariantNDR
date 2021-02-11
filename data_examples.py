"""Generate various example datasets for dimensionality reduction testing.

    All methods return data in the form of a numpy array, X, where each
    column corresponds to a single data point.

"""

import numpy as np
import numpy.linalg as LA
import random
from scipy.special import comb

def circleRPn(
    dim=4,
    num_segments=0,
    segment_points=50,
    v=0,
    randomize=True
):
    """Points on a "kinked" circle in projective space.

    Constructs a curve of evenly-spaced points along the great circle
    from :math:`e_i` to :math:`e_{i+1}` in :math:`\mathbb{R}^{d+1}' for
    each i from 0 to `num_segments`, then returning to 0. The output
    data is randomly permuted so that the first n points are not on the
    same linear subspace, generically.

    Parameters
    ----------
    dim : int, optional
        Dimension of :math:`\mathbb{R}P^d` to work on (ambient
        euclidean space is `dim`+1).
    num_segments : int, optional
        Number of turns to make before returning to start point. Must be
        no larger than the `dim`. Default is equal to `dim`.
    segment_points : int, optional
        Number of points along each segment of curve.
    noise : float, optional
        If ``noise > 0``, noise is added to the curve. Each point is
        offset by a random vector with each component in
        ``[-noise,noise]``, then renormalized to the sphere.
    randomize : bool, optional
        If true, return the points in a random order, rather then
        sequentially. Default is true. Some dimensionality reduction
        steps behave poorly if too many consecutive points lie on the
        same low-dimensional subspace.

    Returns
    -------
    X : ndarray
        Array of coordinate values in :math:`\mathbb{R}^{d+1}`.
    """

    if num_segments < 1:
        num_segments = dim
    if int(num_segments) != num_segments:
        raise ValueError('Number of segments must be a positive integer. '\
            'Supplied value was %2.2f.' %num_segments)
    if dim < 1:
        raise ValueError('Dimension must be a positive integer. Supplied '\
            'values were num_segments = %2.2f and dimension = %2.2f' 
            %(num_segments,dim))
    if dim < num_segments:
        raise ValueError('Value of dimension must be larger than number of '\
            'segments. Supplied dimension was %i and number of segments was '\
            '%i' %(dim,num_segments))
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
    if v > 0:
        N = v*rng.random((dim+1,num_points))
        X = (X.T + N)/LA.norm(X.T+N,axis=0)
    return X

def bezier_RPn(ctrl_points,N=100,noise=0):
    """Define a weird curve for testing purposes.
    
    Parameters
    ----------
    ctrl_points : ndarray (d*n)
        Each column is a control point of a Bezier curve in
        :math:`\mathbb{R}^d`. The first column is the start point of
        the curve, and the last column is the end point.
    N : int, optional
        Number of points to put on curve. Default is 1000.
    noise : float, optional
    
    Returns
    -------
    B : ndarray (d*N)
        Array (N*d) with each column a point on the curve. Normalized to
        lie on the sphere.

    Examples
    --------
    >>> B = np.eye(3)
    >>> bezier_RPn(B,N=3)
    array([[1.     , 0.40824829, 0.        ],
        [0.        , 0.81649658, 0.        ],
        [0.        , 0.40824829, 1.        ]])

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

def line_patches(dim, NAngles=10, NOffsets=10, sigma=0.25,cross=False):
    """Sample a set of line segments, as witnessed by square patches.

    Constructs square greyscale images of a line. By varying the angle
    and offset of the line from the center a model of
    :math:`\mathbb{R}P^2` is formed in high-dimensional space. Noise can
    be added to the model by blurring the line segment. Setting `cross`
    creates a set of crossed lines, which model the Moore space.

    Parameters
    ----------
    dim: int
        Image patches will be ``dim*dim``.
    NAngles: int, optional
        Number of angles to sweep between 0 and pi. Default is 10.
    NOffsets: int, optional
        Number of offsets to sweep from the origin (the center of the
        patch) to the edge of the patch. Default is 10.
    sigma: float, optional
        The blur parameter. Higher sigma is more blur. Default is 0.1.
    cross: bool, optional
        If true, superimpose the line rotated by pi/2.

    Returns
    -------
    P : ndarray (dim*dim, N)
        Array of image patches. Each patch is flattened into a vector
        and given as a column of the data. The number of data points is
        ``N = NAngles*NOffsets``.

    """

    N = NAngles*NOffsets
    P = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles, endpoint=False)
    ps = np.linspace(-np.sqrt(2)/2, np.sqrt(2)/2, NOffsets)
    idx = 0
    [X, Y] = np.meshgrid(
        np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim)
        )
    for i in range(NAngles):
        c = np.cos(thetas[i])
        s = np.sin(thetas[i])
        for j in range(NOffsets):
            patch = X*s + Y*c + ps[j]
            patch = np.exp(-patch**2/sigma**2)
            if cross:
                xpatch = X*c - Y*s + ps[j]
                xpatch = np.exp(-xpatch**2/sigma**2)
                # TODO: determine if max or + is better model.
                #patch = (patch + xpatch)/2
                patch = np.maximum(patch,xpatch)
            P[idx, :] = patch.flatten()
            idx += 1
    P = P.T
    return P

