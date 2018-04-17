import numpy as np

from labelscrnavis import utils


gene_groups = {
    'Bcell': ['CD19', 'CD79A'],
    'Tcell': ['CD4', 'CD3D'],
    'Retinal_m': ['VSX2', 'OTX2', 'SXGN', 'ISL1', 'GRM6', 'APOE', 'PAX6',
                  'RHO', 'ARR3'],
    'Bipolar_m': ['TACR3', 'SYT2', 'NETO1', 'IRX6', 'PRKAR2B', 'GRIK1',
                  'KCNG4', 'CABP5', 'VSX1', 'PRKCA'],
    'Monocyte': ['S100A9', 'CD14'],
    'NKcell': ['NKG7'],
    'Megakaryocyte': ['PPBP'],
}

linkages = {
    'single': np.min,
    'complete': np.max,
    'average': np.mean,
}

score_methods = {
    'avg': utils.average_score,
    'max': utils.nanmax_score,
}
