import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

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

class BiFPNConcat(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPNConcat, self).__init__()
        self.d = dimension  # 通常为 1（通道维度）

    def forward(self, x):
        # x: list of 3 tensors, each [B, C, H, W]
        # print(f"BiFPNConcat in {x[0].shape}, {x[1].shape}")
        return torch.cat(x, dim=self.d)  # [B, 3*C, H, W]


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


class PConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU-friendly Haar DWT，下采样到 H//2, W//2
class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义 Haar 小波核
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5],
                           [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5],
                           [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])
        # 堆叠成 (4,1,2,2)
        weight = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer('weight', weight)

    def forward(self, x):
        B, C, H, W = x.shape
        # 每个通道独立做 DWT
        weight = self.weight.repeat(C, 1, 1, 1)  # (4*C,1,2,2)
        out = F.conv2d(x, weight, stride=2, groups=C)  # (B,4*C,H//2,W//2)
        return out


class DWTBackbone(nn.Module):
    def __init__(self, c1):
        super().__init__()
        # 用 1x1 conv 做通道映射，保持原始接口一致
        self.conv1 = nn.Conv2d(c1*4, 64, 1, bias=False)
        self.conv2 = nn.Conv2d(64*4, 128, 1, bias=False)
        self.conv3 = nn.Conv2d(128*4, 256, 1, bias=False)
        self.conv4 = nn.Conv2d(256*4, 512, 1, bias=False)
        self.conv5 = nn.Conv2d(512*4, 1024, 1, bias=False)

        self.dwt = HaarDWT()

    def forward(self, x):
        # print(f"DWTBackbone: {x.shape}")
        p1 = self.conv1(self.dwt(x))
        p2 = self.conv2(self.dwt(p1))
        p3 = self.conv3(self.dwt(p2))
        p4 = self.conv4(self.dwt(p3))
        p5 = self.conv5(self.dwt(p4))

        # print(f"DWTBackbone: out {p1.shape}, {p2.shape}, {p3.shape}, {p4.shape}, {p5.shape}")
        return [p3, p4, p5]



def get_image(image, pos, size):
    B, _, H, W = image.shape
    pos_x, pos_y = pos

    # 将归一化坐标映射到像素坐标
    cx = (pos_x * W).clamp(0, W - 1)
    cy = (pos_y * H).clamp(0, H - 1)

    # 计算左上角坐标，贴边处理
    x1 = (cx - size // 2).clamp(0, W - size)
    y1 = (cy - size // 2).clamp(0, H - size)

    # 构造采样网格
    lin = torch.linspace(0, size - 1, steps=size, device=image.device)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    # 加上偏移量
    grid[:, :, :, 0] += x1.view(B, 1, 1)
    grid[:, :, :, 1] += y1.view(B, 1, 1)

    # 归一化到 [-1, 1]
    grid[:, :, :, 0] = grid[:, :, :, 0] / (W - 1) * 2 - 1
    grid[:, :, :, 1] = grid[:, :, :, 1] / (H - 1) * 2 - 1

    # 采样
    patch = F.grid_sample(image, grid, mode='bilinear', align_corners=True)
    return patch  # [B, 1, size, size]


class CatBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.coord_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, h, w] → [B, C, 1, 1]
            nn.Conv2d(in_channels, 2, 1)  # [B, C, 1, 1] → [B, 2, 1, 1]
        )
        self.conv1 = nn.Conv2d(1, 64, 3, 2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 64, 3, 2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(1, 64, 3, 2, padding=1, bias=False)


    def forward(self, x):
        image, feat = x  # image: [B, C_img, H, W], feat: [B, C_feat, h, w]
        B, _, H, W = image.shape

        # 特征图 → 坐标
        coord = self.coord_conv(feat)  # [B, 2, 1, 1]
        coord = coord.permute(0, 2, 3, 1)  # → [B, 1, 1, 2]
        coord = coord.view(B, 2).sigmoid()  # 归一化到 [0, 1]
        pos_x, pos_y = coord[:, 0], coord[:, 1]  # [B], [B]

        # 原图中间通道切片
        mid_img = image[:, image.shape[1] // 2:image.shape[1] // 2 + 1, :, :]  # [B, 1, H, W]

        # 裁剪子图
        image1 = get_image(mid_img, (pos_x, pos_y), H // 4)
        image2 = get_image(mid_img, (pos_x, pos_y), H // 8)
        image3 = get_image(mid_img, (pos_x, pos_y), H // 16)
        # print(f"CatBackbone {image1.shape}, {image2.shape}, {image3.shape}")

        ret1 = self.conv1(image1)
        ret2 = self.conv2(image2)
        ret3 = self.conv3(image3)
        # print(f"conv after {ret1.shape}, {ret2.shape}, {ret3.shape}")
        return [ret1, ret2, ret3]
        return [self.conv1(image1), self.conv2(image2), self.conv3(image3)]


class GetFeature(nn.Module):
    def __init__(self, index : int, out_channels : int):
        super().__init__()
        self.index = index
        self.out_channels = out_channels

    def forward(self, features):
        ret = features[self.index]
        # print(f"GetFeature: {ret.shape}")
        return ret
        return features[self.index]