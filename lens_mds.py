""" Dimensionality reduction on Lens spaces using an MDS-type method. """
import autograd.numpy as np
import autograd.numpy.linalg as LA
import matplotlib.pyplot as plt

def lmds(Y,D,p,max_iter=20,verbosity=1,pmo_solve='cg'):
    """Lens multi-dimensional scaling algorithm.

    Parameters
    ----------
    Y : ndarray (m*d)
        Initial guess of data points. Each row corresponds to a point on
        the (2n-1)-sphere, so must have norm one and d must be even.
    D : ndarray (square)
        Distance matrix to optimize toward.
    p : int
        Cyclic group with which to act. Must be prime (this is not
        checked).
    max_iter: int, optional
        Number of times to iterate the loop.
    verbosity: int, optional
        Amount of output to display at each iteration.
    pmo_solve: string, {'cg','sd','tr','nm','ps'}
        Solver to use with pymanopt. Default is conjugate gradient.

    Returns
    -------
    X : ndarray
        Optimal location of data points.
    C : list (float)
        Computed cost at each loop of the iteration.
    T : list (float)
        Actual cost at each loop of the iteration.

    Notes
    -----
    The optimization can only be carried out w/r/t/ an approximation of
    the true cost function (which is not differentiable). The computed
    cost C should not match T, but should decrease when T does.

    """

    m = Y.shape[0]
    d = Y.shape[1] - 1
    W = distance_to_weights(D)
    S = optimal_rotation(Y)
    C = np.cos(D)
#   if d%2 == 0:
#       raise ValueError('Input data matrix must have an even number of ' +
#           'columns (must be on an odd-dimensional sphere). Given data had ' +
#           '%i columns.',%(d+1))
    
    # TODO: verify that input is valid.
    omega = g_action_matrix(p,d)
    cost = setup_cost(Y,omega,S,D,W)
    cost_list = [cost(Y.T)]
    true_cost = setup_cost(projective_distance_matrix(Y),S)
    true_cost_list = [true_cost(Y.T)]
    manifold = Oblique(rank,num_points) # Short, wide matrices.
    solver = ConjugateGradient()

def g_action_matrix(p,d):
    """Create a matrix corresponding to the action of Z_p on S^d. 
        
    Parameters
    ----------
    p : int
        Root of unity to use for quotient, i.e. p = 5 corresponds to
        5th roots of unity.
    d : int
        Dimension of sphere. Must be odd.

    Returns
    -------
    omega : ndarray
        Matrix defining the group action of Z_p on S^d. Always a
        block-diagonal (trilinear), orthogonal matrix.
    
    """
    if d%2 == 0:
        raise ValueError('Sphere must be odd dimensional.')
    theta = 2*np.pi/p
    rot_block = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    omega = np.zeros((d+1,d+1))
    for i in range(0,d+1,2):
        omega[i:i+2,i:i+2] = rot_block
    return omega

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D."""
    # TODO: currently identical to pmds version. Remark if changes.
    W_inv = (1 - np.cos(D)**2)     
    W = np.sqrt((W_inv+np.eye(D.shape[0],D.shape[1]))**-1 - np.eye(D.shape[0],D.shape[1]))
    return W

def optimal_rotation(Y,omega,p):
    """Choose the correct representative from each equivalence class.

    Parameters
    ----------
    Y : ndarray (d*n)
        Data, with one COLUMN for each of the n data points in
        d-dimensional space.
    w : ndarray (d*d)
        Rotation matrix for p-th root of unity.
    p : int
        Order of cyclic group.

    Returns
    -------
    S : ndarray (n*n)
        Correct power of w to use for each inner product. Satisfies
            S + S.T = 0 (mod p)

    """

    # The maximum inner product should be the cosine of half the
    # diameter of the space, and the diameter is 2*pi/p.
    minYY = np.arccos(acos_validate(Y.T@Y))
    S = np.zeros(np.shape(Y.T@Y))
    for i in range(1,p):
        tmpYY = np.arccos(acos_validate(Y.T@LA.matrix_power(omega,i)@Y))
        S[tmpYY<minYY] = i
        minYY = np.minimum(minYY,tmpYY)
    return S

def lens_inner_product(Y,omega,S):
    """Computes the inner product Y.T@Y with the correct representatives.

    Parameters
    ----------
    Y : ndarray (d*n)
        Data with each point as a column.
    w : ndarray (d*d)
        Root of unity.
    S : ndarray (n*n)
        Matrix containing list of correct power of w for each inner product.

    Returns
    -------
    YY : ndarray (n*n)
        Inner product matrix Y.T@Y where all use correct representative.

    """

    # TODO: vectorize this and remove for loops.
    n = Y.shape[1]
    YY = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            YY[i,j] = Y.T[i,:] @ LA.matrix_power(omega,int(S[i,j])) @ Y[:,j]           
    return YY

def setup_cost(Y,omega,S,D,W):
    def F(Y):
        YY = lens_inner_product(Y,omega,S)
        return 0.5*LA.norm(W*YY - W*np.cos(D))**2
    return F

def acos_validate(M):
    """Replace values in M outside of domain of acos with +/- 1."""
    big_vals = M >= 1.0
    M[big_vals] = 1.0
    small_vals = M <= -1.0
    M[small_vals] = -1.0
    return M

