import numpy as np
import pandas as pd
import torch

# config
x1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_X.tsv"
y1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_y.tsv"
x2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_X.tsv"
y2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_y.tsv"
# end config


def read_tsv(path):
    return pd.read_csv(path, delimiter="\t", header=0, index_col=0)


def dataloaders(X1, X2, Y1, Y2, top_mvg=4000):
    """load features and classes from two sources,
    subset both to their shared features and classes.
    Return three data loaders.
    train and val from X1, test from X2

    Genes are in columns, samples in rows.
    """
    X1 = read_tsv(X1)
    Y1 = read_tsv(Y1)
    X2 = read_tsv(X2)
    Y2 = read_tsv(Y2)

    # get intersection of features names and ordering
    _, x1_ind, x2_ind = np.intersect1d(
        X1.columns, X2.columns, assume_unique=True, return_indices=True
    )

    X1 = X1.iloc[:, x1_ind]
    X2 = X2.iloc[:, x2_ind]
    assert (X1.columns == X2.columns).all()

    # select the top n most variable features
    var = np.var(X1)
    var = var.sort_values(ascending=False, na_position="last")
    var = var[0:top_mvg]

    X1 = X1[var.index]
    X2 = X2[var.index]
    assert (X1.columns == X2.columns).all()

    # get intersection of features names and ordering for Y too
    # remove classes that are not shared
    _, x1_ind, x2_ind = np.intersect1d(
        Y1.columns, Y2.columns, assume_unique=True, return_indices=True
    )

    Y1 = Y1.iloc[:, x1_ind]
    Y2 = Y2.iloc[:, x2_ind]
    assert (Y1.columns == Y2.columns).all()

    # remove all observations from Xs and Ys that do not belong to a class anymore
    keepers1 = Y1.values.sum(axis=1) != 0
    keepers2 = Y2.values.sum(axis=1) != 0
    X1 = X1.iloc[keepers1, :]
    Y1 = Y1.iloc[keepers1, :]
    assert len(X1) == len(Y1)
    X2 = X2.iloc[keepers2, :]
    Y2 = Y2.iloc[keepers2, :]
    assert len(X2) == len(Y2)

    # HACK
    #return X1, Y1, X2, Y2

    Y1 = Y1.values.argmax(axis=1)
    Y2 = Y2.values.argmax(axis=1)

    ds1 = torch.utils.data.TensorDataset(
        torch.Tensor(X1.values),
        torch.Tensor(Y1).type(torch.long),
    )

    ds2 = torch.utils.data.TensorDataset(
        torch.Tensor(X2.values),
        torch.Tensor(Y2).type(torch.long),
    )

    return ds1, ds2


if __name__ == "__main__":
    #ds1, ds2 = dataloaders(x1, x2, y1, y2)
    x1, y1, x2, y2 = dataloaders(x1, x2, y1, y2)
