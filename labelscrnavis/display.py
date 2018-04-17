import matplotlib.pyplot as plt
import numpy as np


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

    return plt.scatter(
        cells.tsne_proj[:, 0], cells.tsne_proj[:, 1],
        c=np.array(arr), s=20, cmap='viridis_r',
        alpha=.7)
