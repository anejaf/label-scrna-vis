import numpy as np

from labelscrnavis import utils


gene_groups = {
    'B Cell': ['CD19', 'CD20'],
    'T Cell': ['CD3', 'CD4', 'CD8'],
    'Dendritic Cell': ['CD11C', 'CD123'],
    'NK Cell': ['CD56'],
    'Stem Cell/Precursor': ['CD34'],
    'Macrophage/Monocyte': ['CD14', 'CD33'],
    'Granulocyte': ['CD66B'],
    'Platelet': ['CD41', 'CD61', 'CD62'],
    'Erythrocyte': ['CD235A'],
    'Epithelial Cell': ['CD146'],
    'Endothelial Cell': ['CD326'],
}

linkages = {
    'single': np.min,
    'complete': np.max,
    'average': np.mean,
}

score_methods = {
    'avg': utils.average_score,
    'max': utils.nanmax_score,
    'bin': utils.binary_score,
}
