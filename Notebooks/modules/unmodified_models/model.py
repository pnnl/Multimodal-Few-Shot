# -*- coding: utf-8 -*-
import torch


class BaseModel(torch.nn.Module):
    def __init__(self, encoder, device):
        """
        This class lays out basic fewshot models construction.
        :param encoder:
        :param device:
        """
        super(BaseModel, self).__init__()
        self.encoder = encoder
        self.device = device

    def forward(self, support_set, query_set):
        raise NotImplementedError

