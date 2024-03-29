""" File to carry out the steps in the EM-coords pipeline before
    dimensionality reduction."""

# General imports.
import numpy as np
import numpy.linalg as LA
import geometry
import time

# For geodesic distance matrix estimation.
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from sklearn.neighbors import NearestNeighbors

def geo_distance_matrix(D, epsilon=0.4, k=-1, normalize=True, verbose=False):
    """Approximate a geodesic distance matrix.

    Given a distance matrix uses either an epsilon neighborhood or a
    k-NN algorithm to find nearby points, then builds a distance matrix
    such that nearby points have their ambient distance as defined by
    the original distance matrix, while far away points are given the
    shortest path distance in the graph.

    Parameters
    ----------
    data : ndarray
        Data as an n*2 matrix, assumed to lie on RP^n (i.e. S^n).
    epsilon : float, optional
        Radius of neighborhood when constructing graph. Default is ~pi/8.
    k : int, optional
        Number of nearest neighbors in k-NN graph. Default is -1 (i.e.
        use epsilon neighborhoods). Here `k` includes the point itself.
    normalize: bool
        When `True`, distances are normalized to correspond to the original
        distance matrix, i.e. Dhat is scaled so the maximum distance is no
        larger than `max(D)`.

    Returns
    -------
    Dhat : ndarray
        Square distance matrix matrix of the graph. 

    Raises
    ------
    ValueError
        If the provided value of epsilon or k is too small, the graph
        may not be connected, giving infinite values in the distance
        matrix.

    Examples
    --------
    The four corners of the unit square.
    >>> d = np.sqrt(2)
    >>> D = np.array([[0, 1, d, 1],[1,0,1,d],[d,1,0,1],[1,d,1,0]])
    >>> G = geo_distance_matrix(D, k=3, normalize=False)
    [[0. 1. 2. 1.]
        [1. 0. 1. 2.]
        [2. 1. 0. 1.]
        [1. 2. 1. 0.]]    

    """

    # Use kNN. Sort twice to get nearest neighbour list.
    if k > 0:
        if verbose:
            print('Computing nearest neighbors...')
            tic = time.time()
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(D)
        A_graph = neigh.kneighbors_graph(D, mode='connectivity')
        A = A_graph.toarray()
        if verbose:
            toc = time.time()
            print('Finished in %g seconds' %(toc - tic))
    # Use epsilon neighborhoods.
    else:
        A = D<epsilon
    if verbose:
        print('Computing path-length distances...')
        tic = time.time()
    G = csr_matrix(D*A)                   # Matrix representation of graph
    Dg = floyd_warshall(G, directed=False)     # Path-length distance matrix
    if verbose:
        toc = time.time()
        print('Finished in %g seconds' %(toc - tic))
    if np.isinf(np.max(Dg)):
        raise ValueError('The distance matrix contains infinite values, ' +
            'indicating that the graph is not connected. Try a larger value ' +
            'of epsilon or k.')
    if normalize:
        Dhat = (np.max(D)/np.max(Dg))*Dg    # Normalize distances.
    else:
        Dhat = Dg
    np.fill_diagonal(Dhat, 0)
    return Dhat

def prominent_cocycle(
    cocycles,
    diagram,
    order = 1,
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
    cocycles : list of cocycles in dimension k (`PH['cocycles'][k]`)
    diagram : persistence diagram in dimension k (`PH['dgms'][k]`)
    order: int, optional
        Which cocycle to return. Default is the first (most prominent)
        cocycle, but may be set to higher values.
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
    birth : float
        Time of birth for cocycle.
    death : float
        Time of death for cocycle.

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

    """

    persistence = diagram[:,1] - diagram[:,0]
    index = persistence.argsort()[-1*order] # Longest cycle is last.
    if index > len(cocycles):
        raise NoHomologyError('No cohomology class found. Verify that there '\
            'are at least %d cocycles in this dimension.' %order)
    eta = cocycles[index]
    birth = diagram[index,0]
    death = diagram[index,1]
    return eta, birth, death

def threshold_cocycle(cocycle, D, threshold):
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
    # For 1-cocycles:
    if cocycle.shape[1] == 3:
        for i in range(cocycle.shape[0]):
            if D[tuple(cocycle[i,0:2])] >= threshold:
                bad_rows.append(i)
    # For 2-cocycles:
    elif cocycle.shape[1] == 4:
        for i in range(cocycle.shape[0]):
            if D[tuple(cocycle[i,0:2])] >= threshold:
                bad_rows.append(i)
            elif D[tuple(cocycle[i,1:3])] >= threshold:
                bad_rows.append(i)
            elif D[cocycle[i,0], cocycle[i,2]] >= threshold:
                bad_rows.append(i)
    threshold_cocycle = np.delete(cocycle,bad_rows,0)
    return threshold_cocycle

def partition_unity(D_land, radius, landmarks, bump_type='quadratic'):
    """Partition of unity subordinate to open ball cover.

    Parameters
    ----------
    D_land : ndarray (l*n)
        Matrix of distances from landmarks to entire dataset.
    radius : float
        Radius of balls to use in open cover of data.
    landmarks : int list (l)
        List of indices to use as centers of balls. Elements must be a
        subset of `{0,...,N-1}` but may be repeated.
    type : string, optional
        Type of bump function to use for partition of unity. Default is
        'triangular'. Other options include 'quadratic', 'logarithmic',
        and 'gaussian'.
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

    References
    ----------
    See section 3 of "Multi-Scale Projective Coordinates". In particular,
    this implementation has `epsilon_i = radius` for all `i`, and weights
    of `lambda = radius^2`.

    """

    # U = D[landmarks,:]
    S = D_land < radius
    # Radius must be large enough that every point is in one open ball.
    # Thus each column of `U` must contain at least one value less than
    # radius.
    cover_check = np.sum(S, axis=0)
    if np.min(cover_check) < 1:
        raise ValueError('Open sets do not cover data when epsilon '\
            '= %2.3f.' %radius)
    if bump_type == 'quadratic':
        # This is the default given in the paper.
        S = S*(radius - D_land)**2
    elif bump_type == 'triangular':
        S = S*(radius**2 - radius*D_land)
    elif bump_type == 'linear':
        S = S*(radius - D_land)
    else:
        raise NotImplementedError('This type of bump function not yet '\
            'implemented. Use "quadratic" instead.')
    S = S/np.sum(S, axis=0)
    return S

def convert_cocycle(cocycle, sub_ind):
    """Change labels in cocycle from index in full dataset to index in landmark subset. This is required when cocycle is computed using the n_perm option for Ripser istead of precomputing the landmarks."""

    new_cocycle = np.zeros(cocycle.shape, dtype=int)
    j = 0
    for i in sub_ind:
        idx = np.where(cocycle[:,0] == i)
        new_cocycle[idx, 0] = j
        neg_idx = np.where(cocycle[:,1] == i)
        new_cocycle[neg_idx, 1] = j
        j += 1
    new_cocycle[:,2] = cocycle[:,2]
    return new_cocycle

def proj_coordinates(partition_function, cocycle):
    """Coordinates of data matrix in real projective space.

    Parameters
    ----------
    partition_function : ndarray (d, N)
        Function describing a partition of unity on the open cover given
        by a landmark subset. Each of the `d` rows corresponds to the
        function on the open ball around a landmark, where the `i`-th
        entry is the value of that function on the `i`-th data point.
    cocycle : ndarray (d, 3) or (d, 2)
        Cocycle representing choice of persistent H_1 class. The first
        two columns give the simplices as pairs [i,j]. The third column
        is the value in :math:`\mathbb{Z}_p` corresponding to each
        simplex. Since we always use the field with two elements, the last
        column consists only of `1` and thus can be omitted. See documentation
        for ripser [1]_.

    Returns
    -------
    X : ndarray (d, N)
        Coordinates of the `N` data points on :math:`\mathbb{R}P^n`.

    """

    d = partition_function.shape[0] # number of landmarks.
    N = partition_function.shape[1] # number of data points.
    used_columns = np.zeros(N)
    X = np.zeros((d,N))
    p = cocycle.shape[1]
    for i in range(d):
        tmp_eta = np.zeros(d)
        idx = np.where(cocycle[:,0]==i)
        neg_idx = np.where(cocycle[:,1]==i)
        if p == 3:
            tmp_eta[cocycle[idx,1]] = cocycle[idx,2]
            tmp_eta[cocycle[neg_idx,0]] = 2 - cocycle[neg_idx,2]
        else:
            tmp_eta[cocycle[idx,1]] = 1
            tmp_eta[cocycle[neg_idx,0]] = 1
        for k in range(N):
            if partition_function[i,k] != 0 and not used_columns[k]:
                used_columns[k] = 1
                X[:,k] = np.sqrt(partition_function[:,k])*((-1)**(tmp_eta))
        if sum(used_columns) == N:
            break
    return X

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

def CPn_coordinates(partition_function, theta, nu):
    """Coordinates on complex projective space.

    Parameters
    ----------
    partition_function : ndarray (d, N)
        Function describing a partition of unity on the open cover given
        by a landmark subset. Each of the `d` rows corresponds to the
        function on the open ball around a landmark, where the `i`-th
        entry is the value of that function on the `i`-th data point.
    cocycle : ndarray (d, 4)
        Cocycle representing choice of persistent H^2 class. The first
        three columns give the 2-simplices as triples [i,j,k]. The final column
        is the value corresponding to that simplex. These values are real
        numbers, since we use the harmonic cocycle in H^2(X,R) for complex
        projective coordinates.

    Returns
    -------
    X : ndarray (d, N)
        Coordinates of the `N` data points on :math:`\mathbb{C}P^n`.

    """
    d = partition_function.shape[0] # number of landmarks.
    N = partition_function.shape[1] # number of data points.
    used_columns = np.zeros(N)
    Xcplx = np.zeros((d,N), dtype=complex)
    nu_vals = nu[:,2]
    nu_coc = nu[:,0:2].astype(int)
    theta_vals = theta[:,3]
    theta_coc = theta[:,0:3].astype(int)
    for i in range(d):
        # Setup nu values (i second term of edges)
        tmp_nu = np.zeros(d)
        neg_idx = np.where(nu_coc[:,0]==i)  # want i as the second entry
        idx = np.where(nu_coc[:,1]==i)
        tmp_nu[nu_coc[neg_idx,1]] = -nu_vals[neg_idx]
        tmp_nu[nu_coc[idx,0]] = nu_vals[idx]
        # Setup theta values (i middle term of simplices).
        tmp_theta_mtx = np.zeros((d,d))
        idx0 = np.where(theta_coc[:,0]==i)
        idx1 = np.where(theta_coc[:,1]==i)
        idx2 = np.where(theta_coc[:,2]==i)
        tmp_theta_mtx[theta_coc[idx0,1],theta_coc[idx0,2]] = -theta_vals[idx0]
        tmp_theta_mtx[theta_coc[idx1,0],theta_coc[idx1,2]] = theta_vals[idx1]
        tmp_theta_mtx[theta_coc[idx2,0],theta_coc[idx2,1]] = -theta_vals[idx2]
        tmp_theta_mtx += -tmp_theta_mtx.T
        for k in range(N):
            if partition_function[i,k] != 0 and not used_columns[k]:
                used_columns[k] = 1
                coef = tmp_nu + np.sum(tmp_theta_mtx * partition_function[:,k], axis=1)
                Xcplx[:,k] = np.sqrt(partition_function[:,k])*np.exp(2*np.pi*1j*coef)
        if sum(used_columns) == N:
            break
    X = geometry.realify(Xcplx)
    return X

def integer_lift(cocycle, p):
    """
    Turn a cocycle with Z_p coefficients into an integer cocycle.

    Parameters
    ----------
    cocycle: ndarray
        Cocycle with coefficients mod `p`.
    p : int
        Prime with which coefficients are computed. Recommended that `p > 2`.

    Returns
    -------
    Z_cocycle : ndarray
        Cocycle with coefficients lifted to `Z`.

    Notes:
    ------
    No check is performed to confirm that the cocycle is in the kernel.

    """

    Z_vals = np.copy(cocycle[:,-1])
    shifts = np.where(Z_vals > (p-1)/2)
    Z_vals [shifts] -= p
    Z_cocycle = np.column_stack((cocycle[:,0:-1], Z_vals))
    return Z_cocycle

def harmonic_cocycle(beta, D_mtx, p, filtration):
    """Compute the coboundary matrix for the Rips complex."""
    # Explicitly construct 2-skeleton of the Vietoris-Rips complex.
    # This lists the 0-, 1-, and 2-cells of the complex in lex order.
    n = D_mtx.shape[0]
    A = np.triu(D_mtx < filtration, k=1)
    edge1, edge2 = np.where(A == 1)
    edges = np.column_stack((edge2, edge1))
    tris = []
    for i in range(n):
        for j in range(i,n):
            for k in range(j,n):
                if A[i,j]*A[j,k]*A[i,k]:
                    tris.append([k,j,i])   # store in reverse order.
    tris = np.array(tris)
    edge_dict = simplex_index(edges)
    tri_dict = simplex_index(tris)
    # Construct the coboundary matrix:
    cobdry = np.zeros((tris.shape[0], edges.shape[0]))
    for i in range(tris.shape[0]):
        bdry = [str(tris[i,1:3]),
            str(np.array([tris[i,0], tris[i,2]])),
            str(tris[i,0:2])
            ]
        cobdry[i, edge_dict[bdry[0]]] = 1
        cobdry[i, edge_dict[bdry[1]]] = -1
        cobdry[i, edge_dict[bdry[2]]] = 1
    # First convert simplex names to a vector.
    beta_cells = []
    for i in range(beta.shape[0]):
        beta_cells.append(str(beta[i,0:3]))
    beta_idx = []
    for i in beta_cells:
        beta_idx.append(tri_dict[i])
    beta_val = beta[:,3]
    beta_vec = np.zeros(tris.shape[0])
    beta_vec[beta_idx] = beta_val
    # Solve the least squares problem.
    # nu' = argmin|| beta - d*nu||
    # theta = beta - d*nu'
    nu_val = np.linalg.pinv(cobdry) @ beta_vec
    theta_val = beta_vec - cobdry@nu_val
    theta = np.column_stack((tris[:,0], tris[:,1], tris[:,2], theta_val,))
    nu = np.column_stack((edges[:,0], edges[:,1], nu_val,))
    return theta, nu

def simplex_index(edges):
    """Convert between simplex names and location in array of edges."""
    edge_names = []
    edge_dict = {}
    for i in range(edges.shape[0]):
        edge_names.append(str(edges[i,:]))
        edge_dict[edge_names[i]] = i
    return(edge_dict)

def rotate_to_pole(v):
    """Rotation matrix aligning vector with last standard basis vector.

    Returns an orthogonal or unitary matrix `Q` such that :math:`Qv =
    \|v\|e_n` where :math:`e_n = [0,...,0,1]^T`. Handles both real and
    complex vectors. If the input vector is real, the output matrix is
    orthogonal, while if the input is complex, the output will be a unitary
    matrix.

    Parameters
    ----------
    v : ndarray (n,)
        Vector (real or complex) to rotate so that it aligns with
        :math:`e_n = [0,...,0,1]^T`. If the input array is not
        one-dimensional it is flattened.

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
    greater than three. `Q` is chosen to be orientation-preserving, i.e.
    `det(Q) = +1`.


    """

    v = v.flatten()
    n = v.shape[0]
    if LA.norm(v) < 1e-15:
        raise ZeroDivisionError('Vector is (numerically) zero. Cannot '\
            'rotate into alignment with a standard basis vector.')
    else:
        v = v / LA.norm(v)
    e_n = np.zeros(n)
    e_n[-1] = 1
    c = v - v*e_n
    beta = LA.norm(c)
    if beta < 1e-15:
        Q = np.eye(n)/v[-1]   # Division handles complex case.
    else:
        Q = (np.eye(n)
            - ((1 - v[-1])/beta**2) * np.outer(c,c.conj())
            - (1 - np.conj(v[-1])) * np.outer(e_n,e_n.conj())
            + (np.outer(e_n,c.conj()) - np.outer(c,e_n.conj())))
    return Q

def lpca(X,dim,p=2,tol=-1):
    """Lens principal component analysis algorithm.

    Based on the algorithm described in [1]_, reduces a point cloud in
    the lens space :math:`L^N_p` to a point cloud in :math:`L^n_p` with
    `n << N` by iteratively projecting onto the codimension 1 subspace
    which preserves the maximum amount of variance in the data.

    Parameters
    ----------
    X : ndarray (N,k)
        Data Input data as a set of complex vectors in
        :math:`\mathbb{C}^d`. Each column is assumed to have unit norm.
    dim : int
        Final output dimension for data. Here `dim` should be the
        dimension of the ambient complex space, so that the
        corresponding lens space has dimension `2*dim - 1`.
    p : int
        Cyclic group defining the choice of lens space.
    tol : float
        Maximum amount of variance allowed to be lost in the initial
        classical PCA projection. If negative no classical PCA step is
        used.

    Returns
    -------
    Y : ndarray (dim,k)
        Dimension-reduced point cloud of data. Each column is a complex
        unit vector representing a point in lens space.
    variance : ndarray (N-dim)
        Amount of variance lost with each projection.

    Examples
    --------
    A matrix with no n-th coordinates should project down one dimension
    with no loss in variance.
    >>> X = np.array([[1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    >>> Y, v = lpca(X,3)
    >>> Y
        array([[1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
    >>> v
        [0.0]

    Adding a small amount of wiggle in the last component produces the
    same projection, but with some variance lost.
    >>> X[3,3] = 0.1
    >>> X = X / numpy.linalg.norm(X, axis=0)
    >>> Y, v = lpca(X,3)
    >>> Y
        array([[0.-1.j, 0.+0.j, 0.+0.j, 1.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
    >>> v
        [0.0012417300361755113]

    Notes
    -----

    To speed up projection, a first-pass removing multiple projections
    can be made. The amount of variance allowed to be lost in this
    projection is given by setting `tol`.

    The implementation here differs slightly from that described in
    [1]_. To reduce the dimension by one, first the SVD of the data,
    `X`, is computed, :math:`USV^T = X`. The direction describing the
    least variance is the last column of `U`, the singular vector `u`
    corresponding to the smallest singular value. A rotation matrix `Q`
    is computed such that `Qu = e_n`. The entire dataset is then rotated
    by applying `Q` on the left so that the direction of least variance
    is `e_n`. The data is then projected off of the direction of least
    variance by deleting the last component. This approach makes the
    variance simpler to compute. The distance between a data vector `x`
    and its projected version `Px` is the angle between `x` and `Px`
    modulo the cyclic group action. If projection is removal of the last
    component, then `Px` is necessarily the optimal representative of
    the coset, so the group action can be ignored.  Additionally, the
    angle is given by :math:`\arccos(\sqrt{1 - \|x_n\|^2})` because the
    first (n-1) components of `x` and `Px` agree.  Iteratively reducing
    the dimension by one gives the principal lens coordinates. This
    method does not directly store the principal lens _components_,
    though they can be reconstructed from the sequence of singular
    vectors `u` and rotation matrices `Q` described above.

    References
    ----------
    .. [1] J. Perea and L. Polanco, "Coordinatizing Data with Lens
        Spaces and Persistent Cohomology," arXiv:1905:00350,
        https://arxiv.org/abs/1905.00350

    See Also
    --------
    `rotate_to_pole()`

    """

    Y = X / LA.norm(X,axis=0)
#    U, s, V = np.linalg.svd(Y)
    variance = []
# TODO: implement the initial pass cutting out several dimensions at
# once.
#    if tol > 0:
#        var_list = subspace_variance(U) # TODO: implement variance correctly
#        k = # TODO: find first element in var_list greater than tol.
#        U = U[:,0:k]
#        Y = U.conj().T@Y
#        Y = Y / np.linalg.norm(Y)
#        for i in range(k):
#            variance.append(var_list[i])
    while Y.shape[0] > dim:
        U,_,_ = LA.svd(Y)
        Q = rotate_to_pole(U[:,-1])
        Y = Q@Y
        var = np.mean(np.arccos(acos_validate(np.abs(
            np.sqrt(1 - Y[-1,:]*Y[-1,:].conj())
        )))**2)
        variance.append(var) # TODO: double-check the variance.
        Y = np.delete(Y, (-1), axis=0)
        # TODO: need to handle the case that the norm of a column,
        # post-deletion, is zero. (This does happen.) Note that a
        # small norm but nonzero vector is essentially projected out in
        # a random direction, so that may be a reasonable way to handle
        # numerically zero vectors.
        if np.any(LA.norm(Y, axis=0) < 1e-15):
            raise ZeroDivisionError('Cannot normalize data. Reduction to '\
                'dimension %d set a column to zero.' %Y.shape[0])
        Y = Y / LA.norm(Y, axis=0)
    # TODO: decide on type of return (tuple or dict).
    return Y, variance

# TODO
#def subspace_variance(U,Y,p):
#    return 0

class NoHomologyError(Exception):
    def __init__(self, message):
        self.message = message

###############################################################################
# Functions copied from Luis' notebooks
###############################################################################

# Its actually maxmin subsampling. l_next = argmax_X(min_L(d(x,l)))
def maxmin_subsample_distance_matrix(D, num_landmarks, seed=[]):
    '''
    This function computes minmax subsampling using a square distance matrix.

    :type D: numpy array
    :param D: Square distance matrix

    :type num_landmarks: int
    :param num_landmarks: Number of landmarks

    :type seed: list
    :param list: Default []. List of indices to seed the sampling algorith.
    '''
    num_points = len(D)

    if not(seed):
        ind_L = [np.random.randint(0,num_points)]
    else:
        ind_L = seed
        num_landmarks += 1  # Why? I think this makes it return the wrong
                            # number of points if you apply a seed.

    distance_to_L = np.min(D[ind_L, :], axis=0)

    for i in range(num_landmarks-1):
        ind_max = np.argmax(distance_to_L)
        ind_L.append(ind_max)

        dist_temp = D[ind_max, :]

        distance_to_L = np.minimum(distance_to_L, dist_temp)

    return {'indices':ind_L, 'distance_to_L':distance_to_L}

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
