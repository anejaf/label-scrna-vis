import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from numba import jit

from labelscrnavis import utils
from labelscrnavis import evaluate_cells
from labelscrnavis import projection


def decide_mindist(dists, neighbors=5, display=False):

    topn = np.partition(dists, neighbors + 1)[:, :neighbors+1]
    maxn = sorted(np.max(topn, axis=1), reverse=True)

    dy = np.gradient(maxn)
    dy2 = np.gradient(dy)

    if display:
        fig = plt.figure()
        plt.plot(xs, abs(dy2), color='green')
        plt.axvline(x=np.argmax(abs(dy2)))
    return maxn[np.argmax(abs(dy2))]


def score_neighborhood(cell_idx, cells, adj_arr, gene_group=None):

    rel = list(np.where(adj_arr[cell_idx])[0])

    score = np.average(
        np.array(
            evaluate_cells.score_cells(
                cells[rel], gene_group=gene_group)))
    pval = np.array(
        evaluate_cells.pvalue_hypergeo(
            cells[rel], cells, gene_group))

    return score, pval


def score_cell_neighbors(
        cells, gene_group, neighbors,
        mindist=None, distance='euclidean',
        use_tsne=False, pca_components=None):

    X = cells.data_mat()

    if pca_components is not None:
        X = projection.pca(X, n_components=pca_components)
    if use_tsne:
        X = cells.tsne_proj

    dists = utils.dist_nd(X, X, distance)

    if mindist is None:
        mindist = decide_mindist(dists, neighbors=neighbors)
    print(mindist)

    adj_arr = dists < mindist

    res = np.array([
        score_neighborhood(cidx, cells, adj_arr, gene_group=gene_group)
        for cidx in cells.index])
    cells.add_col('n_score', res[:, 0])
    cells.add_col('n_enrich', -1. * np.log(res[:, 1]))

    return cells


@jit
def loop(cells, adj_arr):

    res = np.zeros((cells.shape[0], 2))
    scores_all = np.average(cells, axis=1)

    for i in range(cells.shape[0]):

        rel_cells = list(np.where(adj_arr[i])[0])
        scores = np.average(cells[rel_cells], axis=1)

        k = np.sum(scores > 0)
        n = np.sum(scores_all > 0)
        pval = stats.hypergeom.sf(k, cells.shape[0], n, len(rel_cells))

        res[i][0] = np.average(scores)
        res[i][1] = pval

    return res


def score_cell_neighbors_fast(
        cells, gene_group, neighbors,
        mindist=None, distance='euclidean',
        use_tsne=False, pca_components=None):

    X = cells.data_mat()

    if pca_components is not None:
        X = projection.pca(X, n_components=pca_components)
    if use_tsne:
        X = cells.tsne_proj

    dists = utils.dist_nd(X, X, distance)

    if mindist is None:
        mindist = decide_mindist(dists, neighbors=neighbors)
    print(mindist)

    adj_arr = dists < mindist

    res = np.array(loop(cells.data_mat(genes=gene_group), adj_arr))
    cells.add_col('n_score', res[:, 0])
    cells.add_col('n_enrich', -1. * np.log(res[:, 1]))

    return cells


