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
    if death < 2*birth+epsilon:
        valid_class = False
    else:
        valid_class = True
    return eta, valid_class


class HomologyError(Exception):
    def __init__(self, message):
        self.message = message

