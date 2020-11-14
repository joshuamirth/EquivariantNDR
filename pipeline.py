""" File to carry out the steps in the EM-coords pipeline before
    dimensionality reduction."""

import numpy as np
import scipy as sp
from ripser import ripser
from persim import plot_diagrams

def prominent_cocycle(D,q=2,epsilon=1e-3):
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
    # TODO: understand exactly what epsilon does.

    Returns
    -------
    eta : ndarray (?)
        Representative of the most persistent H_1 cocycle in the
        Vietoris–Rips persistent homology of D.
    valid_class : bool
        Whether the cohomology class if persistent enough to produce a
        valid classifying map. If false the data may lack any H_1
        cocycles or a different coefficient field may be required.
        

    Raises
    ------
    HomologyError
        If there are no persistent H_1 cocycles at all.
    
    """
    
    PH = ripser(D,coeff=q,do_cocycles=True,maxdim=2,distance_matrix=True)
    cocycles = PH['cocycles'][0]
    diagram = PH['dgms'][0]
    persistence = diagram[:,1] - diagram[:,0]
    index = persistence.argsort()[-1] # Longest cycle is last.
    if index > len(cocycles):
        raise HomologyError('No PH_1 classes found. Either there is no '\
            'persistent homology in dimension 1 when computed with '\
            'Z/%dZ coefficients or the distance matrix was improperly '\
            'specified.' %q)
    birth = diagram[index,0]
    death = diagram[index,1]
    if death < 2*birth + epsilon:
        valid_class = False
    else:
        valid_class = True
    return eta, valid_class

def partition_unity_jrm(D,radius,landmarks,conical=False):
    """Partition of unity subordinate to open ball cover.

    Parameters
    ----------
    D : ndarray (n*n)
        Distance matrix of dataset.
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

def lens_coordinates_jrm(
    partition_function,
    cocycle,
    p,
):
    """Coordinates of data matrix in lens space.
    
    Parameters
    ----------
    partition_function : ndarray (d*N)
        Function describing a partition of unity on the open cover given
        by a landmark subset. Each of the `d` rows corresponds to the
        function on the open ball around a landmark, where the `i`-th
        entry is the value of that function on the `i`-th data point.
    cocycle : ndarray (l*3)
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
        idx = np.where(cocycle[:,0]==i)
        # TODO: double check that cocycles from ripser come sorted in the
        # second column, otherwise need to implement that here.
        tmp_eta[idx] = cocycle[:,2][idx]
        zeta_i = zeta**tmp_eta
        for j in range(N):
            if partition_function[i,j] != 0 and not used_columns[j]:
                used_columns[j] = 1
                X[:,j] = np.sqrt(partition_function[:,j])*zeta_i
    return X


class HomologyError(Exception):
    def __init__(self, message):
        self.message = message

