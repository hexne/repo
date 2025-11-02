from torch import nn


class GetFeatures(nn.Module):
    def __init__(self, index : int, out_channels : int):
        super().__init__()
        self.index = index
        self.out_channels = out_channels
    def forward(self, features):
        return features[self.index]

