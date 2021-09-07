import numpy as np
import pandas as pd
import torch
import pickle
from functools import reduce


def read_tsv(path):
    return pd.read_csv(path, delimiter="\t", header=0, index_col=0)


def get_class_weights(Y, names):
    """
    Return inverse of occurence of each class as weight.
    Crashing if a class has 0 occurences is ok,
    as these classes should have been filtered out upfront.
    """
    w = np.bincount(Y)
    ws = sum(w)
    assert len(names) == len(w)
    d = dict(zip(names, ws / w))
    return d


def dataloaders(X1, X2, Y1, Y2, top_mvg=4000, cbiopickle=None):
    """load features and classes from two sources,
    subset both to their shared features and classes.
    Return three pytorch data sets.
    train and val from X1, test from X2.
    Optionally return another data set if z-scores from cbioportal are present.

    Genes are in columns, samples in rows.
    """
    X1 = read_tsv(X1)
    Y1 = read_tsv(Y1)
    X2 = read_tsv(X2)
    Y2 = read_tsv(Y2)

    # also subset to cbio genes if present
    if cbiopickle:
        with open(cbiopickle, "rb") as f:
            X3 = pickle.load(f)
        common = reduce(np.intersect1d, (X1.columns, X2.columns, X3.columns))
    else:
        # get intersection of features names and ordering
        common = np.intersect1d(X1.columns, X2.columns, assume_unique=True)

    # select the top n most variable features from the source distribution
    X1 = X1[common]
    var = np.var(X1)
    var = var.sort_values(ascending=False, na_position="last")
    var = var[0:top_mvg]

    X1 = X1[var.index]
    X2 = X2[var.index]
    assert (X1.columns == X2.columns).all()
    if cbiopickle:
        X3 = X3[var.index]
        assert (X1.columns == X3.columns).all()

    # get intersection of features names and ordering for Y too
    # remove classes that are not shared
    common = np.intersect1d(Y1.columns, Y2.columns, assume_unique=True)
    Y1 = Y1[common]
    Y2 = Y2[common]
    assert (Y1.columns == Y2.columns).all()

    # remove all observations from Xs and Ys that do not belong to a class
    # TBD: could we keep all classes for source and only remove from target?
    keepers1 = Y1.values.sum(axis=1) != 0
    keepers2 = Y2.values.sum(axis=1) != 0
    X1 = X1.iloc[keepers1, :]
    Y1 = Y1.iloc[keepers1, :]
    assert len(X1) == len(Y1)
    X2 = X2.iloc[keepers2, :]
    Y2 = Y2.iloc[keepers2, :]
    assert len(X2) == len(Y2)

    ynames = Y1.columns
    Y1 = Y1.values.argmax(axis=1)
    Y2 = Y2.values.argmax(axis=1)
    w = get_class_weights(Y1, ynames)

    import stratified_split

    train_x, train_y, test_x, test_y = stratified_split.my_train_test_split(X1, Y1)

    ds1 = torch.utils.data.TensorDataset(
        torch.Tensor(X1.values),
        torch.Tensor(Y1).type(torch.long),
    )

    train = torch.utils.data.TensorDataset(
        torch.Tensor(train_x.values),
        torch.Tensor(train_y).type(torch.long),
    )
    val = torch.utils.data.TensorDataset(
        torch.Tensor(test_x.values),
        torch.Tensor(test_y).type(torch.long),
    )

    ds2 = torch.utils.data.TensorDataset(
        torch.Tensor(X2.values),
        torch.Tensor(Y2).type(torch.long),
    )

    if cbiopickle:
        ds3 = torch.utils.data.TensorDataset(
            torch.Tensor(X3.values),
        )
        return ds1, train, val, ds2, w, (ds3, X3.index)

    return ds1, train, val, ds2, w


if __name__ == "__main__":
    # config
    x1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_X.tsv"
    y1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_y.tsv"
    x2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_X.tsv"
    y2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_y.tsv"
    # end config
    result = dataloaders(x1, x2, y1, y2)
