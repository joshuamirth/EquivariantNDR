""" Projective space geometry tools. """
import numpy as np
import numpy.linalg as LA

# Miscellaneous Functions

def sqrt_validate(X):
    """Replace matrix with entries > 0."""
    tol = 1e-9
    x_min = np.min(X)
    if x_min > 0:
        return X
    elif x_min < -tol:
        print('WARNING: matrix contains nontrivial negative values.')
    bad_idx = X < 0
    X[bad_idx] = 0
    return X

def acos_validate(M,tol=1e-6):
    """Replace values outside of domain of acos with +/- 1.

    Parameters
    ----------
    M : ndarray (m,n)
        Input matrix.
    tol : float
        Raises a warning if the values of `M` lie outside of
        [-1-tol,1+tol]. Default is `1e-6`.

    Returns
    -------
    M : ndarray (m,n)
        Matrix with values > 1 replaced by 1.0 and values < -1 replaced
        by -1.0. Modifies the input matrix in-place.

    Examples
    --------

    """

    if  np.max(M) > 1 + tol or np.min(M) < -1 - tol:
        print('Warning: matrix contained a value of %2.4f. Input may be '\
            'outside of [-1,1] by more than floating point error.' %np.max(M))
    big_vals = M >= 1.0
    M[big_vals] = 1.0
    small_vals = M <= -1.0
    M[small_vals] = -1.0
    return M

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D."""
    W = np.sqrt((1 - np.cos(D)**2 + np.eye(D.shape[0]))**-1)
    return W


###############################################################################
# Real projective space geometry tools
###############################################################################

def RPn_validate(Y):
    """Check that Y is a valid element of RPn."""
    valid = np.isrealobj(Y)
    if Y.ndim > 1:
        valid *= np.allclose(LA.norm(Y, axis=0), np.ones(Y.shape[1]))
    else:
        valid *= np.allclose(LA.norm(Y), np.ones(Y.shape))
    return bool(valid)

def RPn_geo_distance_matrix(Y):
    """Construct the (exact) distance matrix of data Y on RP^d."""
    M = np.abs(Y.T@Y)
    acos_validate(M)
    D = np.arccos(M)    # Initial distance matrix
    np.fill_diagonal(D, 0)
    return D

def RPn_chordal_distance_matrix(X):
    D = np.sqrt(sqrt_validate(1 - (X.T@X)**2))
    np.fill_diagonal(D, 0)
    return D

###############################################################################
# Complex projective space geometry tools
###############################################################################

# Elements of complex projective space can be thought of as points on the
# 2n-sphere modulo an equivalence relation. I will think of the first
# n coordinates as the real part and the last n coordinates as the complex
# part. All functions will work with this real representation of the vectors.
# There is one conversion method in case of natural data with complex
# representation. Additionally, all data points are thought of as column
# vectors.

def CPn_validate(Y):
    """Check that Y is a valid element of CPn in the real representation."""
    valid = ( np.isrealobj(Y) * (np.mod(Y.shape[0], 2) == 0))
    if Y.ndim > 1:
        valid *= np.allclose(LA.norm(Y, axis=0), np.ones(Y.shape[1]))
    else:
        valid *= np.allclose(LA.norm(Y), np.ones(Y.shape))
    return valid

def realify(Y):
    """Convert data in n-dimensional complex space into 2n-dimensional real
    space.
    """
    Yreal = np.vstack((np.real(Y), np.imag(Y)))
    return Yreal

def complexify(Y):
    """Convert real 2n-dimensional points into n-dimensional complex vectors.
    """

    if np.mod(Y.shape[0], 2) != 0:
        raise ValueError('Cannot convert odd-dimensional vector to complex.')
    n = int(Y.shape[0]/2)
    Ycplx = Y[0:n] + 1j*Y[n:2*n]
    return Ycplx

def times_i(Y):
    """Multiply the real representation of a complex vector by i."""
    n = int(Y.shape[0]/2)
    i_mtx = np.vstack(
        (np.hstack((np.zeros((n, n)), -np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n)))))
    )
    iY = i_mtx@Y
    return iY

def CPn_geo_distance_matrix(Y):
    """Construct the (exact) distance matrix of data Y on CP^n."""
    n = int(Y.shape[0]/2)
    i_mtx = np.vstack(
        (np.hstack((np.zeros((n, n)), -np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n)))))
    )
    M = (Y.T@Y)**2 + (Y.T@(i_mtx@Y))**2
    M = np.sqrt(M)
    acos_validate(M)
    D = np.arccos(M)
    np.fill_diagonal(D, 0)
    return D

def CPn_chordal_distance_matrix(X):
    n = int(X.shape[0]/2)
    i_mtx = np.block([
        [np.zeros((n, n)), -np.eye(n)],
        [np.eye(n), np.zeros((n, n))]
        ])
    D = np.sqrt(sqrt_validate(1 - ((X.T @ X)**2 + (X.T @ (i_mtx@X))**2)))
    np.fill_diagonal(D, 0)
    return D

def norm_rotations(Y):
    """Compute a matrix S of complex numbers such that |<y_i, y_j>| is
    given by <y_i, s_ij y_j>."""
    sreal = Y.T @ Y
    simag = Y.T @ times_i(Y)
    norms = np.sqrt(sreal**2 + simag**2)
    sreal = sreal / norms
    simag = simag / norms
    return sreal, simag

def norm_compare(Y, Areal, Aimag):
    """Compare |<y_i, y_j>| with a<y_i,y_j>."""
    ip_real = Y.T@Y
    ip_imag = -Y.T@times_i(Y)
    ip_true = np.sqrt(ip_real**2 + ip_imag**2)
    ip_computed = Areal*ip_real - Aimag*ip_imag
    diff = np.linalg.norm(ip_true - ip_computed)
    imag_err = Areal*ip_imag + Aimag*ip_real
    return diff, imag_err

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
