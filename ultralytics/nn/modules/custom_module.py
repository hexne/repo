import torch
from torch import nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x)
        y = self.fc(y)
        return x * y

class Conv3D(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=(1, k, k), stride=(1, s, s), padding=(0, p, p), bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, c1, H, W]
        x = x.unsqueeze(2)              # → [B, c1, 1, H, W]
        x = self.conv(x)                # → [B, c2, 1, H/2, W/2]
        return x.squeeze(2)             # → [B, c2, H/2, W/2]

class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.linear = nn.Linear(c1, c2)

    def forward(self, x):
        # x: [B, c1, H, W]
        x = x.permute(0, 2, 3, 1)       # → [B, H, W, c1]
        x = self.linear(x)              # → [B, H, W, c2]
        x = x.permute(0, 3, 1, 2)       # → [B, c2, H, W]
        return x


class BiFPNConcat(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPNConcat, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class DataSwitch(nn.Module):
    def __init__(self, c1):
        super().__init__()

    def forward(self, x):
        print(f"shape is {x.shape}")
        return x
