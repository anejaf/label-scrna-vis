import matplotlib.pyplot as plt
import numpy as np

from scipy import spatial, interpolate
from matplotlib import patches as ptc


def confidence_ellipse(data, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.
    https://scipython.com/book/chapter-7-matplotlib/examples/
    bmi-data-with-confidence-ellipses/
    """

    centre = np.mean(data, axis=0)
    cov = np.cov(data.T)

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    print(2 * 3 * np.sqrt(eigvals))
    width, height = 2 * 3 * np.sqrt(eigvals)
    return ptc.Ellipse(xy=centre, width=width, height=height,
                       angle=np.degrees(theta), **kwargs)


def convex_hull(data, **kwargs):

    c_hull = spatial.ConvexHull(data)
    points = data[c_hull.vertices]
    centr = np.mean(points, axis=0)
    lens = np.linalg.norm(centr - points, axis=1)

    dxs = (centr[0] - points[:, 0]) / lens
    dys = (centr[1] - points[:, 1]) / lens

    x3s = points[:, 0] - 0.5 * np.array(dxs)
    y3s = points[:, 1] - 0.5 * np.array(dys)

    # tck, u = interpolate.splprep(data[c_hull.vertices].T, s=0, per=1)
    tck, u = interpolate.splprep(np.array([x3s, y3s]), s=0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    coords_new = interpolate.splev(u_new, tck, der=0)

    poly = ptc.Polygon(np.array(coords_new).T, **kwargs)
    return poly


def plot_region(cells, region):

    """
    plots tsne projection of cells
    all cells in blue, region in yellow circles

    :param cells: all cells to plot (CellsDf)
    :param region: row indices of cells in the region
    """

    plt.plot(
        cells.tsne_proj[:, 0], cells.tsne_proj[:, 1],
        'o', color=[0.993248, 0.906157, 0.143936], alpha=.5)
    plt.plot(
        cells.tsne_proj[region][:, 0], cells.tsne_proj[region][:, 1],
        'o', color=[0.267004, 0.004874, 0.329415], alpha=.5)


def plot_clusters(cells, cluster_col):

    """
    plots tsne projection of cells
    colors are based on the column with name cluster_col in cells
    """

    col = np.array(cells.get_col(cluster_col)).astype(float)
    inf_idx = col == np.inf
    col[inf_idx] = col[~inf_idx].max() + 1

    return plt.scatter(
        cells.tsne_proj[:, 0], cells.tsne_proj[:, 1],
        c=col, s=20, cmap='viridis_r',
        alpha=.7)


def plot_col(cells, arr):

    """
    plots tsne projection of cells
    colors based on the array provided (num_cells == len(arr))
    """

    col = np.array(arr).astype(float)
    inf_idx = col == np.inf
    if np.sum(inf_idx == cells.shape[0]):
        col[inf_idx] = 1
    else:
        col[inf_idx] = col[~inf_idx].max() + 1
    col = col / np.max(col)

    return plt.scatter(
        cells.tsne_proj[:, 0], cells.tsne_proj[:, 1],
        c=col, s=20, cmap='viridis_r',
        alpha=.7)


def plot_col_bin(cells, arr, colore, label=''):

    """
    plots tsne projection of cells
    colors based on the array provided (num_cells == len(arr))
    """

    trues = np.where(arr == True)
    falses = np.where(arr == False)

    plt.scatter(
        cells.tsne_proj[falses][:, 0], cells.tsne_proj[falses][:, 1],
        c=[0.993248, 0.906157, 0.143936], s=20, alpha=.7)
    plt.scatter(
        cells.tsne_proj[trues][:, 0], cells.tsne_proj[trues][:, 1],
        c=colore, s=20, alpha=.7, label=label)
