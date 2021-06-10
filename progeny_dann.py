#!/usr/bin/env python
# coding: utf-8

# # Implementation of DANN
# Reference: https://arxiv.org/pdf/1505.07818.pdf

# TODO
# * sample from tgt, as currently with zip the longer data loader has elements discarded

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import datetime
import os, sys

from matplotlib.pyplot import imshow, imsave

from gmm_linear import GMMLinear
from acc import get_accuracy

MODEL_NAME = "DANN"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# config
x1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_X.tsv"
y1 = "/vol/home-vol3/wbi/messersc/dev/footprints/v1.0/Progeny_y.tsv"
x2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_X.tsv"
y2 = "/vol/home-vol3/wbi/messersc/dev/SPEED2_data_download/06_speed2_dataloader/Speed2_y.tsv"

dim_feature_space = 10
batch_size = 16

max_epoch = 100
lambda_scale = 1.0
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
            nn.Linear(200, 20),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(20),
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

    def __init__(self, dim_feature_space=512, num_classes=9):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim_feature_space, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
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
            nn.Linear(dim_feature_space, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
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

ds1, ds2 = data_loader.dataloaders(x1, x2, y1, y2)

# sample from tgt for each batch of src
n_batches = len(ds1) // batch_size
assert n_batches != 0

# create data loader
ds1 = DataLoader(
    ds1, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
)


ds2 = DataLoader(
    ds2, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
)
tgt_set = iter(ds2)


bce = nn.BCELoss()
xe = nn.CrossEntropyLoss()
D_src = torch.ones(batch_size, 1).to(DEVICE)  # Discriminator Label to real
D_tgt = torch.zeros(batch_size, 1).to(DEVICE)  # Discriminator Label to fake
D_labels = torch.cat([D_src, D_tgt], dim=0)


# Annealing lambda?
def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2.0 / (1 + np.exp(-10.0 * p)) - 1.0


def sample_tgt(step, n_batches):
    global tgt_set
    if step % n_batches == 0:
        tgt_set = iter(ds2)
    return tgt_set.next()


step = 0
ll_c, ll_d = [], []
acc_lst = []
acc_tgt_lst = []

for epoch in range(1, max_epoch + 1):
    # for idx, ((src, labels), (tgt, _)) in enumerate(zip(ds1, ds2)): # TODO
    for idx, (src, labels) in enumerate(ds1):
        tgt, _ = sample_tgt(step, n_batches)
        # Training Discriminator
        src, labels, tgt = (
            src.to(DEVICE),
            labels.to(DEVICE),
            tgt.to(DEVICE),
        )

        x = torch.cat([src, tgt], dim=0)
        h = F(x)
        y = D(h.detach())

        Ld = bce(y, D_labels)
        D.zero_grad()
        Ld.backward()
        D_opt.step()

        c = C(h[:batch_size])
        y = D(h)
        Lc = xe(c, labels)
        Ld = bce(y, D_labels)
        lambda_ = lambda_scale * get_lambda(epoch, max_epoch)
        Ltot = Lc - lambda_ * Ld

        F.zero_grad()
        C.zero_grad()
        D.zero_grad()

        Ltot.backward()

        C_opt.step()
        F_opt.step()

        if step % 100 == 0:
            dt = datetime.datetime.now().strftime("%H:%M:%S")
            print(
                "Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, lambda: {:.4f} ---- {}".format(
                    epoch, max_epoch, step, Ld.item(), Lc.item(), lambda_, dt
                )
            )
            ll_c.append(Lc)
            ll_d.append(Ld)

        if step % 400 == 0:
            F.eval()
            C.eval()
            with torch.no_grad():
                ## the following code gives incorrect results, because not the
                ## whole dataset is used in each epoch - incomplete batches are
                ## dropped by the data loader to not crash batchnorm1d with
                ## batches of 1 samples
                # corrects = torch.zeros(1).to(DEVICE)
                # for idx, (src, labels) in enumerate(ds1):
                #    src, labels = src.to(DEVICE), labels.to(DEVICE)
                #    c = C(F(src))
                #    _, preds = torch.max(c, 1)
                #    corrects += (preds == labels).sum()
                # acc = corrects.item() / len(ds1.dataset)
                # print("***** Accuracy: {:.4f}, Step: {}".format(acc, step))
                # acc_lst.append(acc)

                acc = get_accuracy(nn.Sequential(F, C), ds1)
                print("***** Accuracy on source: {:.4f}, Step: {}".format(acc, step))
                acc_lst.append(acc)

                acc = get_accuracy(nn.Sequential(F, C), ds2)
                print("***** Accuracy on target: {:.4f}, Step: {}".format(acc, step))
                acc_tgt_lst.append(acc)

            F.train()
            C.train()
        step += 1

acc = get_accuracy(nn.Sequential(F, C), ds1)
print("***** Accuracy on source: {:.4f}, Step: {}".format(acc, step))
acc_lst.append(acc)

acc = get_accuracy(nn.Sequential(F, C), ds2)
print("***** Accuracy on target: {:.4f}, Step: {}".format(acc, step))
acc_tgt_lst.append(acc)

# ## Visualize Sample

# In[26]:


import matplotlib.pyplot as plt


# In[27]:


# XE loss
plt.plot(range(len(ll_c)), ll_c)


# In[28]:


# Discriminator loss
plt.plot(range(len(ll_d)), ll_d)


# In[29]:


# Accuracy
plt.plot(range(len(acc_lst)), acc_lst)


# In[30]:


max(acc_lst)


# In[ ]:
