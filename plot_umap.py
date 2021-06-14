import umap
import umap.plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_embedding2D(ds1, ds2, F, reduction="pca"):
    """Given two data sets and a trained feature extractor,
    plot 2D representation of source distribution as well
    as classes
    """
    emb1 = F(ds1.tensors[0]).detach().numpy()
    emb2 = F(ds2.tensors[0]).detach().numpy()
    emb = np.concatenate((emb1, emb2), axis=0)
    nrow1 = emb1.shape[0]
    nrow2 = emb2.shape[0]
    classes1 = ds1.tensors[1].detach().numpy().tolist()
    classes2 = ds2.tensors[1].detach().numpy().tolist()
    classes = classes1 + classes2
    distrs1 = [0] * nrow1
    distrs2 = [1] * nrow2
    distrs = distrs1 + distrs2
    distrs = [*distrs1, *distrs2]
    assert emb.shape[0] == len(distrs)

    if reduction == "pca":
        reducer = PCA(n_components=2)
    elif reduction == "umap":
        reducer = umap.UMAP()
    else:
        raise ("no valid method selected")

    embedding = reducer.fit_transform(emb)
    plot_embedding2D_(embedding, distrs, classes, reduction)

    embedding = reducer.transform(emb1)
    plot_embedding2D_(embedding, distrs1, classes1, reduction=reduction + "_src1_")

    embedding = reducer.transform(emb2)
    plot_embedding2D_(embedding, distrs2, classes2, reduction=reduction + "_src2_")


def plot_embedding2D_(embedding, distrs, classes, reduction):
    plt.close()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in distrs],
        alpha=0.4,
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title(reduction + "projection by source in feature space", fontsize=14)
    plt.savefig(reduction + "_distrs.pdf")
    plt.close()

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in classes],
        alpha=0.4,
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title(reduction + "projection by pathway in feature space", fontsize=14)
    plt.savefig(reduction + "_classes.pdf")
    plt.close()
