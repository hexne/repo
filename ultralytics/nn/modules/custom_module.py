import torch
from torch import nn
import torch.nn.functional as F
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        reduced = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"in shape: {x.shape}")
        y = self.pool(x)
        y = self.fc(y)
        # ret = x * y
        # print(f"out shape: {ret.shape}")
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


class BiFPN2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # print(f"BiFPN2 in {x[0].shape}, {x[1].shape}")
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        fused = weight[0] * x[0] + weight[1] * x[1]
        # print(f"BiFPN2 out {fused.shape}")
        return fused


# 融合版
class BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN, self).__init__()
        self.d = dimension  # 保留接口一致性，实际未使用
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4

    def forward(self, x):
        # x: list of 3 tensors, each [B, C, H, W]
        # print(f"in {x[0].shape}, {x[1].shape}, {x[2].shape}")
        w = F.relu(self.w)
        weight = w / (w.sum() + self.eps)
        # print(f"{x[0].shape}, {x[1].shape},  {x[2].shape}")

        fused = weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]  # [B, C, H, W]
        # print(f"fused {fused.shape}")

        return fused

# 拼接版本
# class BiFPN(nn.Module):
#     def __init__(self, dimension=1):
#         super(BiFPN, self).__init__()
#         self.d = dimension  # 通常为 1（通道维度）
#
#     def forward(self, x):
#         # x: list of 3 tensors, each [B, C, H, W]
#         return torch.cat(x, dim=self.d)  # [B, 3*C, H, W]


class DataSwitch(nn.Module):
    def __init__(self, c1):
        super().__init__()

    def forward(self, x):
        # print(f"DataSwitch {x.shape}")
        return x

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class DWT(nn.Module):
    def __init__(self, c1, c2):
        super(DWT, self).__init__()
        self.dwt = DWTForward(J=1, wave='sym4', mode='periodization')  # 推荐使用 sym4 + zero
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, bias=False)
        self.eps = 1e-4

    def forward(self, x):
        # print(f"DWT in {x.shape}")
        Yl, Yh = self.dwt(x)  # Yl: LL, Yh[0]: [B, C, 3, H/2, W/2] or list of 3 tensors

        # 兼容两种格式：Yh[0] 是 list 或是合并张量
        if isinstance(Yh[0], torch.Tensor) and Yh[0].dim() == 5 and Yh[0].shape[2] == 3:
            LH, HL, HH = Yh[0][:, :, 0], Yh[0][:, :, 1], Yh[0][:, :, 2]
        elif isinstance(Yh[0], (list, tuple)) and len(Yh[0]) == 3:
            LH, HL, HH = Yh[0]
        else:
            raise ValueError(f"Expected Yh[0] to contain 3 tensors, got {type(Yh[0])} with shape {getattr(Yh[0], 'shape', 'N/A')}")

        x_cat = torch.cat([Yl, LH, HL, HH], dim=1)  # [B, 4*c1, H/2, W/2]
        ret =  self.conv(x_cat)
        # print(f"DWT out {ret.shape}")
        return ret
        return self.conv(x_cat)  # [B, c2, H/2, W/2]
