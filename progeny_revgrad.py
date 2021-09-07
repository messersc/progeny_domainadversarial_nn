#!/usr/bin/env python
# coding: utf-8

# # Implementation of DANN
# Reference: https://arxiv.org/pdf/1505.07818.pdf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import datetime

from gmm_linear import GMMLinear
from acc import get_accuracy
from gradient_reversal import GradientReversal

MODEL_NAME = "DANN"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# config
x1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_X.tsv"
y1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_y.tsv"
x2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_X.tsv"
y2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_y.tsv"
cbio = "/vol/home-vol3/wbi/messersc/dev/footprints/03_torch_model/tcga_data/coadread_tcga_pan_can_atlas_2018/data_RNA_Seq_v2_mRNA_median_Zscores.pickle"

dim_feature_space = 10
batch_size = 16

max_epoch = 55
lambda_scale = 1
# end config


class FeatureExtractor(nn.Module):
    """
    Feature Extractor
    """

    def __init__(self, n_in=4000, dim_feature_space=512):
        super().__init__()
        self.net = nn.Sequential(
            GMMLinear(n_in, 500, 3),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(500),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(20, 20),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
            nn.Linear(20, dim_feature_space),
        )

    def forward(self, x):
        embedding = self.net(x)
        return embedding


class Classifier(nn.Module):
    """
    Classifier
    """

    def __init__(self, dim_feature_space=512, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim_feature_space, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_classes),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class Discriminator(nn.Module):
    """
    Simple Discriminator w/ MLP
    """

    def __init__(self, dim_feature_space=512, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            GradientReversal(),
            nn.Linear(dim_feature_space, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


F = FeatureExtractor(dim_feature_space=dim_feature_space).to(DEVICE)
C = Classifier(dim_feature_space=dim_feature_space).to(DEVICE)
D = Discriminator(dim_feature_space=dim_feature_space).to(DEVICE)

F_opt = torch.optim.Adam(F.parameters())
C_opt = torch.optim.Adam(C.parameters())
D_opt = torch.optim.Adam(D.parameters())


# data loaders
import data_loader

ds1, train, val, ds2, class_weights, (ds3, tcganames) = data_loader.dataloaders(
    x1, x2, y1, y2, 4000, cbio
)
class_names = class_weights.keys()
class_weights = torch.Tensor(list(class_weights.values()))

# sample from tgt for each batch of src
n_batches = len(ds1) // batch_size
assert n_batches != 0

# create data loader
dl1 = DataLoader(
    ds1, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
)

train_dl = DataLoader(
    train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
)

val_dl = DataLoader(
    val, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
)

dl2 = DataLoader(
    ds2, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
)

# speed 2 as target set
tgt_set = iter(dl2)


def sample_tgt(step, n_batches):
    global tgt_set
    if step % n_batches == 0:
        tgt_set = iter(dl2)
    return tgt_set.next()


# Annealing lambda?
def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2.0 / (1 + np.exp(-10.0 * p)) - 1.0


bce = nn.BCELoss()
xe = nn.CrossEntropyLoss(weight=class_weights)

D_src = torch.ones(batch_size, 1).to(DEVICE)  # Discriminator Label to real
D_tgt = torch.zeros(batch_size, 1).to(DEVICE)  # Discriminator Label to fake
D_labels = torch.cat([D_src, D_tgt], dim=0)

step = 0
ll_c, ll_d = [], []
ll_val = []
acc_lst = []
acc_tgt_lst = []

for epoch in range(1, max_epoch + 1):
    for idx, (src, labels) in enumerate(train_dl):
        tgt, _ = sample_tgt(step, n_batches)
        # Training Discriminator
        src, labels, tgt = (
            src.to(DEVICE),
            labels.to(DEVICE),
            tgt.to(DEVICE),
        )

        x = torch.cat([src, tgt], dim=0)
        h = F(x)
        #        y = D(h.detach())

        #        Ld = bce(y, D_labels)
        #        D.zero_grad()
        #        Ld.backward()
        #        D_opt.step()

        c = C(h[:batch_size])
        y = D(h)
        Lc = xe(c, labels)
        Ld = bce(y, D_labels)
        lambda_ = lambda_scale  ### * get_lambda(epoch, max_epoch)
        Ltot = Lc + lambda_ * Ld

        F.zero_grad()
        C.zero_grad()
        D.zero_grad()
        Ltot.backward()
        C_opt.step()
        F_opt.step()
        D_opt.step()

        step += 1

    dt = datetime.datetime.now().strftime("%H:%M:%S")
    print(
        "Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, lambda: {:.4f} ---- {}".format(
            epoch, max_epoch, step, Ld.item(), Lc.item(), lambda_, dt
        )
    )
    ll_c.append(Lc)
    ll_d.append(Ld)
    # check val_loss
    F.eval()
    C.eval()
    with torch.no_grad():
        val_pred = nn.Sequential(F, C)(val.tensors[0])
        ll_c_val = xe(val_pred, val.tensors[1])
        ll_val.append(ll_c_val)
        print(
            "***** C Loss on validation set: {:.4f}, Epoch: {}".format(ll_c_val, epoch)
        )
    F.train()
    C.train()


F.eval()
C.eval()
D.eval()

model = nn.Sequential(F, C)
acc = get_accuracy(model, dl1)
print("***** Accuracy on source: {:.4f}, Step: {}".format(acc, step))
acc_lst.append(acc)

acc = get_accuracy(model, val_dl)
print("***** Accuracy on validation set: {:.4f}, Step: {}".format(acc, step))

acc = get_accuracy(model, dl2)
print("***** Accuracy on target: {:.4f}, Step: {}".format(acc, step))
acc_tgt_lst.append(acc)

# Visualize losses

import matplotlib.pyplot as plt

# XE loss
plt.plot(range(len(ll_c)), ll_c)
plt.savefig("loss_classifier_TRAIN.pdf")
plt.close()
# XE loss on validation
plt.plot(range(len(ll_val)), ll_val)
plt.savefig("loss_classifier_VAL.pdf")
plt.close()
# Discriminator loss
plt.plot(range(len(ll_d)), ll_d)
plt.savefig("loss_discriminator.pdf")
plt.close()
# Accuracy
plt.plot(range(len(acc_lst)), acc_lst)
plt.savefig("loss_accuracy.pdf")
plt.close()


from sklearn.metrics import confusion_matrix, classification_report

predictions = model(ds2.tensors[0]).detach().argmax(1)
print(
    classification_report(
        ds2.tensors[1].detach(), predictions, target_names=list(class_names)
    )
)

predictions = model(ds1.tensors[0]).detach().argmax(1)
print(
    classification_report(
        ds1.tensors[1].detach(), predictions, target_names=list(class_names)
    )
)

predictions = model(ds3.tensors[0]).detach().numpy()
predictions = pd.DataFrame(data=predictions, index=tcganames, columns=class_names)
predictions.transpose().to_csv("tcga_predictions.csv", sep="\t")


# plot some 2D representations

import plot_umap

plot_umap.plot_embedding2D(ds1, ds2, F, "pca")
plot_umap.plot_embedding2D(ds1, ds2, F, "umap")

plot_umap.plot_embedding2D(ds1, ds3, F, "pca", "pca_TCGA")
plot_umap.plot_embedding2D(ds1, ds3, F, "umap", "umap_TCGA")
