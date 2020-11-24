""" File to carry out the steps in the EM-coords pipeline before
    dimensionality reduction."""

import numpy as np
import numpy.linalg as LA
from ripser import ripser

def prominent_cocycle(
    D,
    q = 2,
    threshold_at_death = True
):
    """Primary cocycle from H_1 persistence for lens coordinate
    representation.

    Computes the Vietoris-Rips persistent homology of a dataset
    (provided as a distance matrix) and returns the most persistent H_1
    cocycle. Also checks that this cocycle is sufficiently long to
    provide a valid lens coordinate classifying map and returns the
    corresponding covering radius.

    Parameters
    ----------
    D : ndarray (n*n)
        Distance matrix. Must be square. Symmetry, non-negativity, and
        triangle inequality are not enforced, but violations may lead to
        unexpected results.
    q : int, optional
        Coefficient field in which to compute homology. Must be prime.
        Default is 2.
    epsilon : Tolerance for covering radius. Default is 0.001. The
        persistent cocycle must die after 2*birth + epsilon to be valid.
    return_persistence : bool
        Set true to return the ripser output.
    threshold_at_death : bool, optional
        If true, remove edges from the cocycle which do not exist before
        the death value. In principle this is unnecessary because the
        partition of unity will be valued at zero for these edges, but
        for any other analysis it may be desirable to return a true
        cocycle.

    Returns
    -------
    eta : ndarray (?)
        Representative of the most persistent H_1 cocycle in the
        Vietoris–Rips persistent homology of D.
    valid_class : bool
        Whether the cohomology class if persistent enough to produce a
        valid classifying map. If false the data may lack any H_1
        cocycles or a different coefficient field may be required.
    cover_radius : float
        Birth time of valid cocycle + epsilon.
    persistence : dict, optional
        Full output from Ripser. Only returned if `return_persistence`
        is set.
        
    Examples
    --------
    The simple "house" simplicial complex has a cocycle.
    >>> D = np.array([
            [0. , 1. , 1.414 , 1. , 0.7],
            [1. , 0. , 1. , 1.414 , 1.5],
            [1.414. , 1. , 0. , 1. , 1.5],
            [1. , 1.414 , 1. , 0. , 0.7],
            [0.7, 1.5, 1.5, 0.7, 0. ]])
    >>> prominent_cocycle(D)
    (array([[1, 0, 1]]), False)

    Because this cocycle is born at `1` and dies at `1.414`, it is not
    persistent enough to necessarily give a valid class. However, in
    this case it is in fact a Cech cocycle.

    Raises
    ------
    NoHomologyError
        If there are no persistent H_1 cocycles at all.
    
    TODO
    ----
    * A better version of this function would return _all_ valid cocycles,
    allowing a user to choose which one to create map from.
    * Sort out how to return data more coherently.

    """
    
    PH = ripser(D,coeff=q,do_cocycles=True,maxdim=1,distance_matrix=True)
    cocycles = PH['cocycles'][1]
    diagram = PH['dgms'][1]
    persistence = diagram[:,1] - diagram[:,0]
    index = persistence.argsort()[-1] # Longest cycle is last.
    if index > len(cocycles):
        raise NoHomologyError('No PH_1 classes found. Either there is no '\
            'persistent homology in dimension 1 when computed with '\
            'Z/%dZ coefficients or the distance matrix was improperly '\
            'specified.' %q)
    eta = cocycles[index]
    birth = diagram[index,0]
    death = diagram[index,1]
    if threshold_at_death:
        eta = threshold_cocycle(eta,D,death)
    return eta, birth, death

def threshold_cocycle(cocycle,D,threshold):
    """Cocycle edges with length less than threshold.

    Take a cocycle and return it restricted only to edges which appear
    before `r = threshold` in the Vietoris-Rips persistent homology.
    This is necessary because Ripser returns representative cochains
    which are possibly from a later persistence value [1]_.

    Parameters
    ----------
    cocycle : ndarray (d,3)
        Cocycle representing choice of persistent H_1 class. The first
        two columns give the simplices as pairs [i,j]. The third column
        is the value in :math:`\mathbb{Z}_p` corresponding to each
        simplex.
    D : ndarray (n,n)
        Distance matrix for dataset.
    threshold : float
        Maximum length of edge to allow for a cocycle. Usually <= death
        time of corresponding feature in persistent homology.

    Returns
    thresh_cocycle : ndarray (k,3)
        Cocycle with long edges removed. Thus `k <= d`.

    References
    ----------
    .. [1] https://ripser.scikit-tda.org/notebooks/Representative%20Cocycles.html

    """

    bad_rows = []
    for i in range(cocycle.shape[0]):
        if D[tuple(cocycle[i,0:2])] >= threshold:
            bad_rows.append(i)
    threshold_cocycle = np.delete(cocycle,bad_rows,0)
    return threshold_cocycle

def partition_unity(D,radius,landmarks,conical=False):
    """Partition of unity subordinate to open ball cover.

    Parameters
    ----------
    D : ndarray (n*n)
        Distance matrix of entire dataset.
    radius : float
        Radius of balls to use in open cover of data.
    landmarks : int list (l)
        List of indices to use as centers of balls. Elements must be a
        subset of `{0,...,N-1}` but may be repeated.
    conical : bool, optional
        If true, use a cone-shaped partition of unity in which points
        nearer the center of an open set are more heavily weighted. If
        `False` uses a piecewise-constant partition of unity. Default is
        `False`.

    Returns
    -------
    S : ndarray (l*n)
        Matrix containing values of partition of unity. Each column sums
        to one, and the `j`-th column gives the value of the partition
        function subordinate to the open ball centered at landmark `j`.

    Examples
    --------
    >>> D = numpy.array([[0,1,2,1,0.7],
            [1,0,1,2,1.7],
            [2,1,0,1,1.7],
            [1,2,1,0,0.7],
            [0.7,1.7,1.7,0.7,0]])
    >>> radius = 1.05
    >>> landmarks = [0,1,2,3]
    >>> partition_unity(D,radius,landmarks)
    array([[0.33333333, 0.33333333, 0.        , 0.33333333, 0.5       ],
           [0.33333333, 0.33333333, 0.33333333, 0.        , 0.        ],
           [0.        , 0.33333333, 0.33333333, 0.33333333, 0.        ],
           [0.33333333, 0.        , 0.33333333, 0.33333333, 0.5       ]])

    """

    U = D[landmarks,:]
    S = U < radius
    if conical:
        S = S*(radius - U)
    if not np.all(np.sum(S,axis=0)):
        raise DivisionByZeroError('Open sets do not cover data when epsilon '\
            '= %d.' %epsilon)
    else:
        S = S/np.sum(S,axis=0)
        return S

def lens_coordinates(
    partition_function,
    cocycle,
    p
):
    """Coordinates of data matrix in lens space.
    
    Parameters
    ----------
    partition_function : ndarray (d*N)
        Function describing a partition of unity on the open cover given
        by a landmark subset. Each of the `d` rows corresponds to the
        function on the open ball around a landmark, where the `i`-th
        entry is the value of that function on the `i`-th data point.
    cocycle : ndarray (d*3)
        Cocycle representing choice of persistent H_1 class. The first
        two columns give the simplices as pairs [i,j]. The third column
        is the value in :math:`\mathbb{Z}_p` corresponding to each
        simplex. See documentation for ripser [1]_.
    p : int (prime power)
        Choice of quotient for lens space coordinates. Must match the
        coefficient field in which the cocycle takes values. (So if
        homology was computed with :math:`\mathbb{Z}_3` coefficients,
        need to set `p = 3`.)
        
    Returns
    -------
    X : ndarray (n*N)
        Coordinates of data in lens space :math:`L_p^d`. Each column is
        a complex vector with unit norm (a point on :math:`S^{2n-1}`).

    Examples
    --------
    >>> partition_function = numpy.array([[0.3333, 0.3333, 0., 0.3333, 0.5],
            [0.3333, 0.3333, 0.3333, 0., 0.],
            [0., 0.3333, 0.3333, 0.3333, 0.],
            [0.3333, 0., 0.3333, 0.3333, 0.5]])
    >>> cocycle = numpy.array([[1,0,1])
    >>> p = 2
    >>> lens_coordinates_jrm(D,partition_function,cocycle,p)
    array([[ 0.57735027+0.j, 0.57735027+0.j, -0.+0.j, 0.57735027+0.j, 0.70710678+0.j],
           [ 0.57735027+0.j, 0.57735027+0.j, 0.57735027+0.j, 0.+0.j,  0.+0.j],
           [ 0.+0.j, 0.57735027+0.j, 0.57735027+0.j, 0.57735027+0.j, 0.+0.j],
           [ 0.57735027+0.j, 0.+0.j, 0.57735027+0.j, 0.57735027+0.j, 0.70710678+0.j]])   

    References
    ----------
    .. [1] Ripser.py API documentation, https://ripser.scikit-tda.org/
    
    """

    d = partition_function.shape[0] # number of landmarks.
    N = partition_function.shape[1] # number of data points.
    used_columns = np.zeros(N)
    zeta = np.exp(2j*np.pi/p)
    X = np.zeros((d,N),dtype=complex) # set to complex to avoid unsafe cast.
    for i in range(d):
        tmp_eta = np.zeros(d)
        idx = np.where(cocycle[:,0]==i)  # what about other order?
        tmp_eta[cocycle[idx,1]] = cocycle[idx,2]
        neg_idx = np.where(cocycle[:,1]==i)
        tmp_eta[cocycle[neg_idx,0]] = p - cocycle[neg_idx,2]
        zeta_i = zeta**tmp_eta
        for k in range(N):
            if partition_function[i,k] != 0 and not used_columns[k]:
                used_columns[k] = 1
                X[:,k] = np.sqrt(partition_function[:,k])*zeta_i
        if sum(used_columns) == N:
            break
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

class NoHomologyError(Exception):
    def __init__(self, message):
        self.message = message


###############################################################################
# What I infer to be Luis' Lens PCA Algorithm
###############################################################################

def pca(X,tol=0.2):
    """ Classical prinicipal component analysis dimensionality
    reduction.

    Parameters
    ----------
    X : ndarray (d,n)
        Data Input data as a set of complex vectors in :math:`\mathbb{C}^d`.
        Each column is assumed to have unit norm.
    tol : float
        Maximum amount of variance allowed to be lost in the projection.

    Returns
    -------
    Y : ndarray (k,n)
        Data reduced to complex dimension `k`. Dimension is determined
        by tolerance setting. 

    Notes
    -----
    
    Examples
    --------

    """

    U, _, _ = np.linalg.svd(X, full_matrices=True)
    v_old = pca_variance(U[:, 0:1], X)
    variance = [v_old]
    k_break = U.shape[0]
    for i in range(2,len(U)+1):
        v_new = pca_variance(U[:, 0:i], X)
        if abs(v_old - v_new) < tol:
            k_break = i
            break
        variance.append(v_new)
        v_old = v_new
    U_tilde = U[:, 0:k_break]
    # variance.append( v_0 ) # lost variance in the projection
    # TODO: removing the above in favor of recording the variance
    # explained per dimension (instead of all in one chunk). Not sure if
    # this actually makes sense: there may be some incompatibility in
    # scale between the basic pca variance and the lens pca variance
    # below making the plots look weird.
    X = np.transpose(np.conj(U_tilde))@X 
    X = X / np.linalg.norm(X,axis=0)
    return X, variance

def pca_variance(U, X):
# def sqr_ditance_projection(U, X):
    """Function copied from Luis' code.

    Notes
    -----
    I don't understand this. It is supposed to, I think, compute some
    notion of variance captured by doing a naive PCA projection onto the
    first k components of the PCA basis followed by renormalizing. But
    _why_ this formula should mean that is beyond me. In particular,
    there doesn't seem to be any reason why the norm of these columns
    should, in general, be in the domain of arccos.

    Additionally, `norm` throws an error here when `U` is a single
    vector. (U^H X is a row vector and the axis command is undefined for
    a 1d array.)

    """

    norm_columns = LA.norm(U.conj().T @ X, axis=0)
    mean = np.mean(np.power(np.arccos(acos_validate(norm_columns)), 2))
    return mean

def lpca_variance(X):
# def sqr_ditance_orthogonal_projection(U, X):
    """ Much like the preceding function, this is copied from Luis'
    code, and I don't quite follow what it does or is supposed to do.
    The same issue arises calling `norm`, and there are typically errors
    forming the square root, as the norm is much larger than 1.

    """

    # Original:
    # print(np.linalg.norm(np.transpose(np.conj(U))@X, axis=0))
    # norm_colums = np.sqrt(1 - np.linalg.norm(np.transpose(np.conj(U))@X, axis=0)**2)
    # return np.mean(np.power(np.arccos( norm_colums ), 2))

    # Updated version that is probably correct:
    var = np.mean(np.arccos(acos_validate(np.abs(np.sqrt(1 - X[-1,:]**2)))))
    return var

def lpca(X,dim=2,p=2,tol=0.02):
    """ Hacked together LPCA code from Luis' example file.

    Parameters
    ----------
    X : ndarray (d,n)
        Input data as a set of complex vectors in :math:`\mathbb{C}^d`.
        Each column is required to have unit norm.
    dim : int
        (Complex) dimension to reduce down to. Default is 2.
    p : int
        Quotient group for lens space. Default is 2.
    tol : float
        Amount of variance allowed to be lost in the initial normal PCA
        projection.

    Returns
    -------
    Y : ndarray (dim,n)
        Data reduced to complex dimension `dim`.
    variance : float list (d - dim)
        Amount of variance lost in each dimension of reduction.

    Notes
    -----
    
    """

    # This first block seems to just reduce dimension as much as
    # possible by ordinary PCA.
    U, _, _ = np.linalg.svd(XX, full_matrices=True)
    v_old = pca_variance(U[:, 0:1], XX)
    variance = [v_old]
    k_break = U.shape[0]
    for i in range(2,len(U)+1):
        v_new = pca_variance(U[:, 0:i], XX)
        if abs(v_old - v_new) < tol:
            k_break = i
            break
        variance.append(v_new)
        v_old = v_new
    U_tilde = U[:, 0:k_break]
    # variance.append( v_0 ) # lost variance in the projection
    # TODO: removing the above in favor of recording the variance
    # explained per dimension (instead of all in one chunk). Not sure if
    # this actually makes sense: there may be some incompatibility in
    # scale between the basic pca variance and the lens pca variance
    # below making the plots look weird.
    XX = np.transpose(np.conj(U_tilde))@XX 
    XX = XX / np.linalg.norm(XX,axis=0)
    # Now a second block does actual Lens PCA down to desired dimension.
    i = 2
    while XX.shape[0] > dim:
        val, vec = np.linalg.eigh(XX@np.transpose(np.conj(XX)))
        vec_smallest = vec[:,0]
        XU,Xs,XV = np.linalg.svd(XX)
        Y = XU@XX
        variance.append(lpca_variance(Y))
        Y = np.delete(Y, (-1), axis=0)
        XX = Y / np.linalg.norm(Y, axis=0)
    return XX, variance

# Its actually maxmin subsampling. l_next = argmax_X(min_L(d(x,l)))
def minmax_subsample_distance_matrix(X, num_landmarks, seed=[]):
    '''
    This function computes minmax subsampling using a square distance matrix.

    :type X: numpy array
    :param X: Square distance matrix

    :type num_landmarks: int
    :param num_landmarks: Number of landmarks

    :type seed: list
    :param list: Default []. List of indices to seed the sampling algorith.
    '''
    num_points = len(X)

    if not(seed):
        ind_L = [np.random.randint(0,num_points)] 
    else:
        ind_L = seed
        num_landmarks += 1  # Why? I think this makes it return the wrong
                            # number of points if you apply a seed.

    distance_to_L = np.min(X[ind_L, :], axis=0)

    for i in range(num_landmarks-1):
        ind_max = np.argmax(distance_to_L)
        ind_L.append(ind_max)

        dist_temp = X[ind_max, :]

        distance_to_L = np.minimum(distance_to_L, dist_temp)
            
    return {'indices':ind_L, 'distance_to_L':distance_to_L}


def rotate_to_pole(v,old=True):
    """Rotation matrix aligning vector with last standard basis vector.

    Returns an orthogonal or unitary matrix `Q` such that :math:`Qv =
    \|v\|e_n` where :math:`e_n = [0,...,0,1]^T`. Handles both real and
    complex vectors. If the input vector is real, the output matrix is
    orthogonal, while if the input is complex, the output will be a unitary
    matrix.

    Parameters
    ----------
    v : ndarray (n,) or (n,1)
        Vector (real or complex) to rotate so that it aligns with
        :math:`e_n = [0,...,0,1]^T`.

    Returns
    -------
    Q : ndarray (n,n)
        Orthogonal (unitary) matrix satisfying :math:`Qv = \|v\|e_n`.

    Notes
    -----

    If the input vector is a row vector (shape (1,n) ndarray), then the
    multiplication `Q@v` is undefined. Instead the returned matrix `Q`
    satisfies `v@Q.T = [0,...,0,1]`.

    There is not a unique solution to :math:`Qv = e_n` in dimensions
    greater than three.

    """

    # Shape input into a column vector ((n,1) ndarray).
    vec_dim = len(v.shape)
    if vec_dim == 1:
        v = np.reshape(v, (-1,1)) 
    elif vec_dim == 2:
        m,n = v.shape
        if m == 1 or n == 1:
            v = np.reshape(v,(m*n,1))
        else:
            raise ValueError('Input must be a vector (ndarray of shape '\
                '(n,), (n,1), or (1,n)). Input given had shape (%d,%d).'\
                %(m,n))
    else:
        raise ValueError('Input must be a vector (ndarray of shape '\
                '(n,), (n,1), or (1,n)). Input given was a %d-dimensional '\
                'array.' %vec_dim)
    n = v.shape[0]
    if LA.norm(v) < 1e-15:
        raise ZeroDivisionError('Vector is (numerically) zero. Cannot '\
            'rotate into alignment with a standard basis vector.')
    else:
        v = v / LA.norm(v)
    e_n = np.zeros((n,1))
    e_n[-1] = 1
    c = v - v*e_n
    beta = LA.norm(c)
    if beta < 1e-15:
        Q = np.eye(n)/v[-1]   # Division handles complex case.
    else:
        Q = (np.eye(n)
            - ((1-v[-1])/beta**2) * c@c.conj().T
            - (1-np.conj(v[-1])) * e_n@e_n.conj().T
            + (e_n@c.conj().T - c@e_n.conj().T))
    return Q

def minmax_subsample_point_cloud(X, num_landmarks, distance):
    '''
    This function computes minmax subsampling using point cloud and a distance function.

    :type X: numpy array
    :param X: Point cloud. If X is a nxm matrix, then we are working with a pointcloud with n points and m variables.

    :type num_landmarks: int
    :param num_landmarks: Number of landmarks

    :type distance: function
    :param  distance: Distance function. Must be able to compute distance between 2 point cloud with same dimension and different number of points in each point cloud.
    '''
    num_points = len(X)
    ind_L = [np.random.randint(0,num_points)]  

    distance_to_L = distance(X[ind_L,:], X)

    for i in range(num_landmarks-1):
        ind_max = np.argmax(distance_to_L)
        ind_L.append(ind_max)
        
        dist_temp = distance(X[[ind_max],:], X)

        distance_to_L = np.minimum(distance_to_L, dist_temp)

    return {'indices':ind_L, 'distance_to_L':distance_to_L}

