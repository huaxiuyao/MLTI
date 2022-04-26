import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import random
from torch.distributions import Beta

from torch.autograd import Variable
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class FCNet(nn.Module):
    def __init__(self, args, x_dim, hid_dim):
        super(FCNet, self).__init__()
        self.args = args
        self.net = nn.Sequential(
            self.fc_block(x_dim, hid_dim),
            self.fc_block(hid_dim, hid_dim),
        )
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim

    def fc_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(),
        )

    def mixup_data(self, xs, ys, xq, yq, lam):
        query_size = xq.shape[0]

        shuffled_index = torch.randperm(query_size)
        xs = xs[shuffled_index]
        ys = ys[shuffled_index]

        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, yq, ys, lam

    def forward(self, x):
        return self.net(x)

    def forward_within(self, inp_support, label_support, inp_query, label_query, lam_mix):

        hidden1_support = self.net[0](inp_support)
        hidden1_query = self.net[0](inp_query)

        hidden1_query, reweighted_query, reweighted_support, lam = self.mixup_data(hidden1_support, label_support, hidden1_query,
                                                           label_query, lam_mix)

        hidden2_query = self.net[1](hidden1_query)

        return hidden2_query, reweighted_query, reweighted_support, lam


    def forward_crossmix(self, x):

        return self.net[1](x)