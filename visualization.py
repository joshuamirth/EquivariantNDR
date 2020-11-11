""" Methods for visualizing lens and projective space data. """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


""" Currently mostly copied from other places. """

def imscatter(X, P, dim, zoom=1):
    """
    Plot patches in specified locations in R2

    Parameters
    ----------
    X : ndarray (N, 2)
        The positions of each patch in R2
    P : ndarray (N, dim*dim)
        An array of all of the patches
    dim : int
        The dimension of each patch

    """
    #https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
    ax = plt.gca()
    for i in range(P.shape[0]):
        patch = np.reshape(P[i, :], (dim, dim))
        x, y = X[i, :]
        im = OffsetImage(patch, zoom=zoom, cmap = 'gray')
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(X)
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])

def plotPatches(P, zoom = 1):
    """
    Plot patches in a best fitting rectangular grid
    """
    N = P.shape[0]
    d = int(np.sqrt(P.shape[1]))
    dgrid = int(np.ceil(np.sqrt(N)))
    ex = np.arange(dgrid)
    x, y = np.meshgrid(ex, ex)
    X = np.zeros((N, 2))
    X[:, 0] = x.flatten()[0:N]
    X[:, 1] = y.flatten()[0:N]
    imscatter(X, P, d, zoom)

