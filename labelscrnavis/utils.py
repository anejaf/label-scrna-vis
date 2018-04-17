import numpy as np
from scipy import spatial


def sort_arr_idxs(arr, reverse=False):

    """ sort array and return element indices of initial array """

    arr_sorted = sorted(zip(arr, np.arange(len(arr))), reverse=reverse)
    return np.array(arr_sorted)[:, 1].astype(int)


def sort_dict_keys(d, reverse=False, func=None):

    """ return dictionary keys in sorted order by values """

    if func is None:
        func = lambda x: x
    return sorted(d, key=lambda k: func(d[k]), reverse=reverse)


def border_dict_keys(d, mode=None, func=None):

    """ return dictionary keys of max or min (param mode) values """

    if func is None:
        func = lambda x: x
    if mode is None:
        mode = max
    min_value = mode(func(list(d.values())))
    return [k for k in d if func(d[k]) == min_value]


def dist_nd(cells, region, distance):

    """ calculate distance between two matrices """

    return spatial.distance.cdist(cells, region, distance)


def average_score(mat, rowvar=True):
    axis = 0
    if rowvar:
        axis = -1
    return np.average(mat, axis=axis)


def nanmax_score(mat, rowvar=True):
    axis = 0
    if rowvar:
        axis = -1
    return np.nanmax(mat, axis=axis)
