""" Methods for visualizing lens and projective space data. """

import numpy as np
import matplotlib.pyplot as plt

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

# Copied from persim, with small modifications.
def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    cechline=False,
    lifetime=False,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams. 

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    if cechline:
        ax.plot([x_down, x_up], [2*x_down, 2*x_up], linestyle="dotted", c='r')

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()