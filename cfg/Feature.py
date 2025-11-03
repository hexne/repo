import torch
from torch import nn


class GetFeatures(nn.Module):
    def __init__(self, index : int, out_channels : int):
        super().__init__()
        self.index = index
        self.out_channels = out_channels
    def forward(self, features):
        return features[self.index]

class MergeFeatures(nn.Module):
    def __init__(self, indexs, out_channels : int):
        super().__init__()
        self.indexs = indexs
        self.out_channels = out_channels

    def forward(self, features):
        selected = [features[i] for i in self.indexs]
        merged = torch.cat(selected, dim=1)
        return merged


