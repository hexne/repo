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

# YOLO里常见的Conv封装
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):  # 自动padding
    if p is None:
        p = k // 2
    return p


class SEBlock(nn.Module):
    ''' Squeeze-and-Excitation 通道注意力模块 '''
    def __init__(self, channels, reduction=2):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class DSConvSE(nn.Module):
    ''' Depthwise Conv → SE → Pointwise Conv(1x1) '''
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        # 深度卷积
        self.dwconv = Conv(c1, c1, k, s, g=c1, act=True)
        # SE 注意力
        self.se = SEBlock(c1, reduction=2)
        # Pointwise 1x1 卷积
        self.pconv = Conv(c1, c2, k=1, s=1, p=0, g=1, act=True)

    def forward(self, x):
        x = self.dwconv(x)   # [B, c1, h, w]
        x = self.se(x)       # 通道加权
        x = self.pconv(x)    # [B, c2, h, w]
        return x






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



import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class CropBlock(nn.Module):
    """
    SE → Depthwise Conv(stride=2) → Pointwise Conv(1x1 → out_channels)
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.se = SEBlock(in_channels, reduction=2)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.se(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def crop_square_grid(img, pos_xy, crop_size):
    """
    使用 grid_sample 一次性裁剪 batch 内所有样本。
    img: [B, C, H, W]
    pos_xy: [B, 2] 像素坐标 (x,y)
    crop_size: int
    return: [B, C, crop_size, crop_size]
    """
    B, C, H, W = img.shape
    # 归一化坐标到 [-1,1]
    cx = pos_xy[:, 0] / (W - 1) * 2 - 1
    cy = pos_xy[:, 1] / (H - 1) * 2 - 1

    # 构造采样网格
    lin = torch.linspace(-1, 1, crop_size, device=img.device)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [crop_size, crop_size, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, crop_size, crop_size, 2]

    # 平移到目标点
    grid[..., 0] += cx.view(B, 1, 1)
    grid[..., 1] += cy.view(B, 1, 1)

    # 采样
    crops = F.grid_sample(img, grid, mode="bilinear", align_corners=True)
    return crops


class CatBackbone(nn.Module):
    def __init__(self, in_channels_img, in_channels_feat, out_channels=64):
        super().__init__()
        self.coord_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_feat, 2, 1)
        )
        self.block1 = CropBlock(in_channels_img, out_channels)
        self.block2 = CropBlock(in_channels_img, out_channels)
        self.block3 = CropBlock(in_channels_img, out_channels)

    def forward(self, x):
        image, feat = x  # image: [B, C_img, H, W], feat: [B, C_feat, h, w]
        B, C_img, H, W = image.shape

        # 生成坐标
        coord = self.coord_conv(feat).view(B, 2).sigmoid()
        pos_x = coord[:, 0] * (W - 1)
        pos_y = coord[:, 1] * (H - 1)
        pos_xy = torch.stack([pos_x, pos_y], dim=1)  # [B,2]

        # 多尺度裁剪
        s1, s2, s3 = H // 4, H // 8, H // 16
        patch1 = crop_square_grid(image, pos_xy, s1)
        patch2 = crop_square_grid(image, pos_xy, s2)
        patch3 = crop_square_grid(image, pos_xy, s3)

        # 特征提取
        ret1 = self.block1(patch1)
        ret2 = self.block2(patch2)
        ret3 = self.block3(patch3)

        return [ret1, ret2, ret3]



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
