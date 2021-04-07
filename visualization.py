""" Methods for visualizing lens and projective space data. """

import numpy as np

def rotate_data(u, X):
    """Rotate data so a specified vector is at the north pole.

    Parameters
    ----------
    u : ndarrary (3,)
        "North pole" vector.
    X : ndarray (n, 3)
        Data array. Rows as data points, columns as features.

    Returns
    -------
    X_rot : ndarray (n, 3)
        Data rotated so that u = [0,0,1] and all equivalence classes are in upper hemisphere.

    """
    # Rotate data so u is at e_3 = [0,0,1].
    if np.allclose(u, np.array([0,0,1])):
        X_rot = np.copy(X)
    elif np.allclose(u, np.array([0,0,-1])):
        X_rot = -1*X
    else:
        # Axis of rotation is cross prodct `u x e_3` (normalized).
        v = np.array([u[1],-u[0],0])/np.sqrt(u[1]**2 + u[0]**2)
        if np.allclose(u[2], 0):
            phi = np.pi / 2
        else:
            phi = np.arctan(np.sqrt(u[0]**2 + u[1]**2)/u[2])    # polar angle of u
        # Apply Rodrigues' rotation formula to the data matrix.
        X_rot = (np.cos(phi)*X 
            + np.sin(phi)*np.cross(v, X) 
            + (1 - np.cos(phi))*(X@np.outer(v, v)))
    # Place all points on the northern hemisphere.
    idx = np.where(X_rot[:,2] < 0)
    X_rot[idx,:] *= -1
    return X_rot

def stereographic(X):
    """Compute stereographic projection from RP^2 to D^2.

    If RP^2 is modeled as vectors in the upper hemisphere of S^2 in R^3, this
    projects that model to the unit disk D^2 in R^2.

    Parameters
    ----------
    X : ndarray(n, 3)
        Data on RP^2.

    Returns
    -------
    X_proj : ndarray(n,2)
        Stereographic projection of X.

    """
    X_proj = np.array([-X[:,0]/(1 + X[:,2]), -X[:,1]/(1 + X[:,2])]).T
    return X_proj

def hopf(Y):
    """
    Map from CP^1 in C^2 = R^4 to the standard representation of S^2
    in R^3 using the Hopf fibration. This is useful for visualization
    purposes.

    Parameters
    ----------
    Y : ndarray (4, k)
        Array of `k` points in CP^1 < R^4 = C^2.

    Returns
    -------
    S : ndarray (3, k)
        Array of `k` points in S^2 < R^3.

    """

    if Y.shape[0] != 4:
        raise ValueError('Points must be in R^4 to apply Hopf map!.')
    S = np.vstack((
        [2*Y[0,:]*Y[1,:] + 2*Y[2,:]*Y[3,:]],
        [-2*Y[0,:]*Y[3,:] + 2*Y[1,:]*Y[2,:]],
        [Y[0,:]**2 + Y[2,:]**2 - Y[1,:]**2 - Y[3,:]**2]))
    return S
