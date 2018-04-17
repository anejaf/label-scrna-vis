import numpy as np
from scipy import stats
import random

from labelscrnavis import maps


def score_cells(
        cells, gene_group=None, score_method='avg', binary=False):

    """
    score cells from a gene group with score method,
    adds a column in CellsDf.df with name 'score_method_gene_group'
    
    :param cells: CellsDf
    :param gene_group: list of genes or a gene group (default: all)
    :param score_method: currently available max and avg (default: avg)
    """

    if isinstance(gene_group, list):
        descr = '_'.join([score_method, str(hash(','.join(gene_group)))])
    else:
        descr = '_'.join([score_method, (gene_group or 'all')])

    scr_m = maps.score_methods.get(score_method, None)

    if descr in cells.cols:
        return cells.get_col(descr)

    if scr_m is None:
        print('Wrong score method')
        return

    if binary:
        res = scr_m(cells.bin_mat(genes=gene_group))
    else:
        res = scr_m(cells.data_mat(genes=gene_group))

    cells.add_col(descr, res)
    return res


def score_rnd_regions(
        cells, region_size=10, gene_group=None, score_method='avg',
        reps=1000):

    """
    score random regions of cells for empirical estimation of pvalue
    
    :param cells: CellsDf
    :param region_size: size of region to score
    :param gene_group: list of genes or a gene group (default: all)
    :param score_method: currently available max and avg (default: avg)
    :param reps: repetitions (how many rnd regions to score, default: 1000)
    """

    scores = score_cells(
        cells, gene_group=gene_group, score_method=score_method)
    
    rnd_scores = [
        np.average(
            scores[random.sample(range(cells.num_cells), region_size)])
        for rep in range(reps)]

    return np.array(rnd_scores)


def pvalue_hypergeo(region, cells, gene_group=None):

    """
    estimation of pvalue with hypergeometric distribution

    :param region, cells: CellsDf
    """

    k = np.sum(score_cells(region, gene_group=gene_group) > 0)
    M = cells.shape[0]
    n = np.sum(score_cells(cells, gene_group=gene_group) > 0)
    N = region.shape[0]

    return stats.hypergeom.sf(k, M, n, N)


def pvalue_rnd(region, cells, gene_group=None, reps=10000):

    """
    estimation of pvalue with scoring random regions

    :param region, cells: CellsDf
    """

    avg = np.average(score_cells(region, gene_group=gene_group))
    rnds = score_rnd_regions(
        cells, gene_group=gene_group, region_size=region.shape[0], reps=reps)

    return np.sum(rnds >= avg) / (reps * 1.0)


def pvalue_rnd_ttest(region, cells, gene_group=None, reps=10000):

    avg_orig = score_cells(region, gene_group=gene_group)
    avg_rnds = score_rnd_regions(
        cells, gene_group=gene_group, region_size=region.shape[0], reps=reps)

    t_stats = stats.ttest_ind(avg_orig, avg_rnds.T, equal_var=False)

    return 1. / reps * np.sum(t_stats.pvalue)
