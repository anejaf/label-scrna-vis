import numpy as np
from scipy import stats


def decide_maxdist(dists, neighbors=5):

    """
    Decide maximum distance for neighborhood definition

    :param dists: matrix of distances (numpy array)
    :param neighbors: desired number of neighbors
    """

    topn = np.partition(dists, neighbors + 1)[:, :neighbors+1]
    maxn = sorted(np.max(topn, axis=1), reverse=True)
    xs = np.arange(len(maxn))

    maxn = np.array(maxn)
    p0 = np.array([xs[0], maxn[0]])
    p1 = np.array([xs[-1], maxn[-1]])
    dsts = abs(
        (p1[1] - p0[1]) * xs -
        (p1[0] - p0[0]) * maxn +
        p1[0] * p0[1] - p1[1] * p0[0]) / np.linalg.norm(p1-p0)
    return maxn[np.argmax(dsts)]


def loop_scores(bin_scores, adj_arr):

    """
    Score cells in a simple loop,
    score(cell) = p-value for siginificance of a cell's neighborhood
    (Simple loop for faster computations)

    :param bin_scores: binary scores for cells
    :param adj_arr: adjecancy array of cells (bin. numpy array: 1 if cells
    are in each other's neighborhoods, else 0)
    """

    num_cells = bin_scores.shape[0]
    res = np.zeros((num_cells,))

    for i in range(num_cells):

        rel_cells = list(np.where(adj_arr[i])[0])

        k = np.sum(bin_scores[rel_cells])
        n = np.sum(bin_scores)
        pval = stats.hypergeom.sf(k, num_cells, n, len(rel_cells))

        res[i] = pval

    return res


def score_cell_neighbors_fast(
        cells, gene_group, neighbors,
        maxdist=None, distance='euclidean',
        fc=1.5,
        use_tsne=False, pca_components=None):

    """
    Score cells with estimating significance of cells' neighborhoods

    :param cells: CellsDf
    :param gene_group: desired gene_group to score for (list of genes)
    :param neighbors: desired number of neighbors in cell's neighborhood
    :param maxdist: provide maximum distance for cell's neighborhood instead
                    of number of neighbors
    :param distance: distance to use (default: euclidean distance)
    :param fc: fold change
    :param use_tsne: whether to use tsne projection for distance computation
                     2D (boolean, default: True)
    :param pca_components: use pca_projection for distance computation
                           (int or None)
    """

    dists = cells.dists(
        distance=distance,
        use_tsne=use_tsne,
        pca_components=pca_components,
        store=False)

    if maxdist is None:
        maxdist = decide_maxdist(dists,
                                 neighbors=neighbors)

    adj_arr = dists < maxdist
    weights = np.average(cells.data_mat(genes=gene_group), axis=1)
    bin_scores = (weights / np.average(weights)) > fc
    
    res = np.array(loop_scores(bin_scores, adj_arr))
    return res


class CellScorer():

    """
    Cell scorer

    This class implements a method proposed in the diploma thesis
    for scoring cells for certain gene group

    Parameters
    ----------
    distance : string (default='euclidean', 'jaccard' or 'cosine')
        type of distance to use for computation of distances between cells
    
    use_tsne: boolean (default=True, False)
        perform computations in space of tsne projection of cells (in 2D)

    pca_components: int or None (default=None)
        perform computations in space of pca projection of cells

    neighbors: int (default=20)
        desired number of neighbors for cells' neighborhoods

    maxdist: float or None (default=None)
        use cell-to-cell distance for defining cells' neighborhoods instead
        of desired number of neighbors

    fc: float (default=1.5)
        fold change to define which cells have a distinctive expression of
        chosen gene group
            cell has a distinctive expression of a chosen gene group if
            w_g(c) > avg(w_g(c')) > fc, where w_g(c) is score of a cell c
            for a chosen gene group g
    """

    def __init__(self,
                 distance='euclidean',
                 use_tsne=True,
                 pca_components=None,
                 neighbors=20,
                 maxdist=None,
                 fc=1.5):

        self.distance = distance
        self.use_tsne = use_tsne
        self.pca_components = pca_components
        self.neighbors = neighbors
        self.maxdist = maxdist
        self.fc = fc

    def score(self, cells, gene_group):

        res = score_cell_neighbors_fast(
            cells, gene_group, self.neighbors, distance=self.distance,
            use_tsne=self.use_tsne, pca_components=self.pca_components,
            maxdist=self.maxdist, fc=self.fc)
        return res

    def binary_score(self, cells, gene_group, p=1e-49):

        res = self.score(cells, gene_group)
        return res < p
