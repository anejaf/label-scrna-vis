from sklearn import decomposition, manifold


def pca(mat, n_components=20):

    """ shortcut to perform pca """

    pca_model = decomposition.PCA(
        n_components=n_components, copy=True, whiten=False,
        svd_solver='auto', tol=0.0, iterated_power='auto',
        random_state=None)
    return pca_model.fit_transform(mat)


def tsne(mat, perplexity=30.0, n_iter=300):

    """ shortcut to perform tsne """

    tsne_model = manifold.TSNE(
        n_components=2, perplexity=perplexity, early_exaggeration=4.0,
        learning_rate=1000.0, n_iter=n_iter, n_iter_without_progress=30,
        min_grad_norm=1e-07, metric='euclidean', init='random',
        random_state=None, method='barnes_hut', angle=0.5)
    return tsne_model.fit_transform(mat)
