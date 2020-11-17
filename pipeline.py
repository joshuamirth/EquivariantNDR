""" File to carry out the steps in the EM-coords pipeline before
    dimensionality reduction."""

import numpy as np
from ripser import ripser

def prominent_cocycle(
    D,
    q=2,
    epsilon=1e-3,
    return_persistence=False,
    threshold_at_death=True
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
        the death value. Almost always will want this to be true.

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
    if death <= 2*birth:
        valid_class = False
    else:
        valid_class = True
    if threshold_at_death:
        eta = threshold_cocycle(eta,D,death-epsilon)
    if return_persistence:
        return eta, valid_class, birth, PH
    else:
        return eta, valid_class, birth

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
        S = S*U
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

class NoHomologyError(Exception):
    def __init__(self, message):
        self.message = message

