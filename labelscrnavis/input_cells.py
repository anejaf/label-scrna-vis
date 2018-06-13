import pandas
import numpy as np
import copy
import networkx as nx

from labelscrnavis import maps
from labelscrnavis import projection
from labelscrnavis import utils


class CellsDf():

    """
    Class holding cells' data and other properties, describing data

    Iterating over class presents iterating over all cells (returns CellsDf)
    Indexing into class is indexing into each cell (returns CellsDf)

    Useful properties:
    * df: pandas DataFrame with data for all cells (each cell represents one
    row, each column represents one gene)
    * available_genes: all available genes in data
    * data_mat: numpy matrix of cell data, can specify genes (list), gene group
    (see maps) and rows of cells
    * bin_mat: numpy matrix of binary cell data (default thresh=0)
    """

    _info_cols = [
        'Type',
        'Replicate',
        'ID',
        'Barcode\r',
        'Cluster ID',
        'Cell ID\r',
    ]

    def __init__(self, df, info_cols=None, available_genes=None,
                 G=None, dists={}, relevant_genes=None):
        if info_cols is None:
            info_cols = CellsDf._info_cols
        if available_genes is None:
            available_genes = [col for col in list(df.keys())
                               if col not in info_cols]
        if relevant_genes is None:
            relevant_genes = available_genes
        self.df = df
        self.available_genes = available_genes
        self.relevant_genes = relevant_genes
        self._G = G
        self._dists = dists
        self.maximum = df.shape[0]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.maximum:
            self.n += 1
            return self.common(
                self.df.iloc[self.n - 1])
        raise StopIteration

    def __getitem__(self, key):
        return self.common(self.df.iloc[key])

    @classmethod
    def from_tab(cls, filepath, info_cols=None, skip_rows=2):
        df = pandas.read_csv(filepath, sep='\t', lineterminator='\n')
        return cls(df.iloc[skip_rows:, :].reset_index(drop=True),
                   info_cols=info_cols)

    @classmethod
    def from_tab_transposed(cls, filepath):
        """ deprecated """
        df = pandas.read_csv(filepath, sep='\t', lineterminator='\n')
        df['GENE\r'] = df['GENE\r'].map(lambda x: x.rstrip('\r'))
        df.rename(columns={'GENE\r': 'GENE'}, inplace=True)
        return cls(df.iloc[2:, :].set_index('GENE').T)

    @property
    def cols(self):
        return list(self.df.keys())

    @property
    def shape(self):
        return self.df.shape

    @property
    def num_genes(self):
        return len(self.available_genes)

    @property
    def num_cells(self):
        return self.df.shape[0]

    @property
    def tsne_proj(self):
        if 'tsne_x' and 'tsne_y' not in self.cols:
            tsne_proj = projection.tsne(projection.pca(self.data_mat()))
            self.df['tsne_x'] = tsne_proj[:, 0]
            self.df['tsne_y'] = tsne_proj[:, 1]
            return tsne_proj
        return np.array([self.get_col('tsne_x'), self.get_col('tsne_y')]).T

    @property
    def index(self):
        return self.df.index

    def dists(self,
              distance='euclidean',
              use_tsne=False,
              pca_components=None,
              store=False):
        if use_tsne:
            key = 'tsne'
            X = self.tsne_proj
        elif pca_components is not None:
            key = 'pca' + str(pca_components)
            X = projection.pca(self.data_mat(), n_components=pca_components)
        else:
            key = 'orig'
            X = self.data_mat()
        dists = self._dists.get(key, None)
        if dists is None:
            dists = utils.dist_nd(X, X, distance)
        if store:
            self._dists[key] = dists
        return dists

    def G(self,
          maxdist, dists=None,
          distance='euclidean', tsne=False, pca=None,
          store=False):
        if dists is None:
            dists = self.dists(
                distance=distance, use_tsne=tsne, pca_components=pca)
        if self._G is None:
            adj_arr = dists < maxdist
            G = nx.Graph(adj_arr)
        else:
            G = self._G
        if store:
            self._G = G
        return G

    def common(self, df):
        """ returns a similar object, with only Df changed"""
        return self.__class__(
            df,
            available_genes=copy.copy(self.available_genes),
            relevant_genes=copy.copy(self.relevant_genes),
            G=copy.copy(self._G),
            dists=copy.copy(self._dists))

    def set_col_value(self, col, row, value):
        self.df[col][row] = value
        return

    def data_mat(self, cols=None, genes=None, rows=None):
        """
        returns a matrix of cell data by selected cols/genes and/or rows
        """
        if cols is None:
            cols = self.available_genes
        if genes is not None:
            if isinstance(genes, list):
                cols = genes
                cols = cols #+ [col.upper() for col in cols]
            else:
                cols = maps.gene_groups.get(genes, self.available_genes)
        cols = [col for col in cols if col in self.cols]
        return self.df.as_matrix(columns=cols).astype('float64')

    def bin_mat(self, cols=None, genes=None, rows=None, thresh=0):
        """
        returns a matrix of cell data in binary
        :param thresh: for controlling the threshold (default: 0)
        """
        return self.data_mat(cols=cols, genes=genes, rows=rows) > thresh

    def get_col(self, col):
        """ get Df's column by name """
        return self.df[col]

    def add_row(self, row):
        """ add a row to Df """
        return self.common(self.df.append(row.df).reset_index(drop=True))

    def add_col(self, col_name, col_data):
        """ add a column to df """
        if col_name in self.df:
            print("Warning: column with this name already exists, \
            replacing this column new data...")
            del self.df[col_name]
        self.df[col_name] = col_data
        return

    def recompute_tsne(self):
        tsne_proj = projection.tsne(projection.pca(self.data_mat()))
        self.df['tsne_x'] = tsne_proj[:, 0]
        self.df['tsne_y'] = tsne_proj[:, 1]
        return tsne_proj
