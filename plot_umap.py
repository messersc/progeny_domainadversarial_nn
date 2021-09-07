import umap

# import umap.plot
import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_embedding2D(ds1, ds2, F, reduction="pca", file_prefix=None):
    """Given two data sets and a trained feature extractor,
    plot 2D representation of source distribution as well
    as classes
    """
    emb1 = F(ds1.tensors[0]).detach().numpy()
    emb2 = F(ds2.tensors[0]).detach().numpy()
    emb = np.concatenate((emb1, emb2), axis=0)
    nrow1 = emb1.shape[0]
    nrow2 = emb2.shape[0]

    classes1 = ds1.tensors[1].detach().numpy()
    if len(classes1.shape) > 1:
        classes1 = classes1.argmax(axis=1).tolist()
    else:
        classes1 = classes1.tolist()

    try:
        classes2 = ds2.tensors[1].detach().numpy()
        if classes2.shape[1] > 1:
            classes2 = classes2.argmax(axis=1).tolist()
        else:
            classes2 = classes2.tolist()
    except IndexError:
        classes2 = [max(classes1) + 1] * nrow2
    classes = classes1 + classes2

    # generate labels for sources
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

    if file_prefix is None:
        file_prefix = reduction

    embedding = reducer.fit_transform(emb)
    plot_embedding2D_(embedding, distrs, classes, file_prefix)

    embedding = reducer.transform(emb1)
    plot_embedding2D_(embedding, distrs1, classes1, file_prefix=file_prefix + "_src1_")

    embedding = reducer.transform(emb2)
    plot_embedding2D_(embedding, distrs2, classes2, file_prefix=file_prefix + "_src2_")


def plot_embedding2D_(embedding, distrs, classes, file_prefix):
    plt.close()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette("colorblind", 11)[x] for x in distrs],
        alpha=0.4,
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title("Projection by source in feature space", fontsize=14)
    plt.savefig(file_prefix + "_distrs.pdf")
    plt.close()

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette("colorblind", 11)[x] for x in classes],
        alpha=0.4,
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title("Projection by pathway in feature space", fontsize=14)
    plt.savefig(file_prefix + "_classes.pdf")
    plt.close()
