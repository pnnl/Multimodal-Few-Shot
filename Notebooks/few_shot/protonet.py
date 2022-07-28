# -*- coding: utf-8 -*-
import torch
from few_shot.model import BaseModel


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

    def forward(self, support_group, queries, support_spectra, spectra, batch_index): # Added
        # support_group is a dictionary of support point tensors keyed by class, create prototypes
        prototypes = []
        support_encoded = {} # Added
        # encode the data points
        for s_k, x_k in support_group.items():
            normalizer = 0 # Added
            support_peak = 0 # Added
            spectral_support_peak = 0 # Added
            support_encoded[s_k] = self.encoder(x_k).mean(0) # Added
            support_peak = torch.max(support_encoded[s_k]) # Added
            spectral_support_peak = torch.max(support_spectra[s_k]) # Added
            normalizer = support_peak / spectral_support_peak # Added
            spectral_norm = support_spectra[s_k] * normalizer # Added
            support_encoded[s_k] = torch.cat([support_encoded[s_k],spectral_norm]) # Added
            # calculate the prototype for each class
            prototypes.append(support_encoded[s_k]) # Added
        #print(prototypes)
        prototypes = torch.stack(prototypes)
        query_set = self.encoder(queries)
        multi_set = torch.empty((len(batch_index),prototypes[0].shape[0])) # Added
        for i, name in enumerate(batch_index): # Added
            normalizer = 0 # Added
            query_peak = 0 # Added
            spectral_peak = 0 # Added
            query_peak = torch.max(query_set[i]) # Added
            spectral_peak = torch.max(spectra[name]) # Added
            normalizer = query_peak / spectral_peak # Added
            spectral_norm = spectra[name] * normalizer # Added
            multi_set[i] = torch.cat([query_set[i],spectral_norm]) # Added
        # calculate a distance between each average and query point
        multi_double = multi_set.double() # Added
        #print('prototypes')
        #print(prototypes)
        #print('multi')
        #print(multi_double)
        distances = abs(torch.cdist(multi_double, prototypes)) # Added
        return distances

