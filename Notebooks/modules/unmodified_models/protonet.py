# -*- coding: utf-8 -*-
import torch
from .model import BaseModel


def euclidean_dist(X, Y):
    """
    Description: Takes two matrices where each row is a separate point.
                 Return the Euclidean distance between each pair of points.

    Inputs:
        :X: torch.Tensor, dimension N x D
        :Y: torch.Tensor, dimension M x D

    Outputs:
        :dists: torch.Tensor, dimension N x M
    """
    n = X.size(0)
    m = Y.size(0)
    d = X.size(1)

    X = X.unsqueeze(1).expand(n, m, d)
    Y = Y.unsqueeze(0).expand(n, m, d)

    dists = torch.pow(X - Y, 2).sum(2)
    return dists


class PrototypicalNet(BaseModel):
    def __init__(self, encoder, device='cpu'):
        super(PrototypicalNet, self).__init__(encoder, device='cpu')
        self.encoder = encoder
        self.device = device

    def forward(self, support_group, queries):
        # support group is a dictionary of support point tensors keyed by class, create prototypes
        prototypes = []
        # encode the data points
        for s_k, x_k in support_group.items():
            #support_group[s_k] = self.encoder(x_k)
            # calculate the prototype for each class
            prototypes.append(self.encoder(x_k).mean(0))
        prototypes = torch.stack(prototypes)
        query_set = self.encoder(queries)
        # calculate a distance between each average and query point
        distances = -torch.cdist(query_set, prototypes)
        return distances

