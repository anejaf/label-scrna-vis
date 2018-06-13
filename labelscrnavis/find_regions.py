from labelscrnavis import cell_scorer


def assign_region(
        cells, pthresh, markers_list, cell_type='ct1',
        neighbors=20, maxdist=None, fc=1.5):

    c_scr = cell_scorer.CellScorer(
        use_tsne=True, neighbors=neighbors, maxdist=maxdist, fc=fc)
    res = c_scr.score(cells, markers_list)

    cells.add_col("{}".format(cell_type), res)

    return cells


def assign_regions(
        cells, pthresh, markers_dict,
        neighbors=20, maxdist=None, fc=1.5,
        col_name='reg'):

    """
    Assign regions to cells

    :param cells: CellsDf
    :param pthresh: top value for p-value
    :param markers_dict: dictionary of type cell_type: [marker1, marker2, ...]
    :param neighbors: desired size of neighborhood
    :param maxdist: define neighborhood by maximum distance
    :param col_name: name of new column in CellsDf containing cells' regions
    """

    for group in markers_dict:
        cells = assign_region(
            cells, pthresh, markers_dict[group], cell_type=group,
            neighbors=neighbors, maxdist=maxdist, fc=fc)

    markers = list(markers_dict.keys())
    cells.add_col(
        col_name,
        cells.df[markers].idxmin(
            axis=1).where(cells.df[markers].min(axis=1) < pthresh))

    return cells
