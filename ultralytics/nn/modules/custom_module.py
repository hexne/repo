# custom_module_safe.py
# Robust custom modules for Ultralytics YOLO integration.
# Keeps original class/function signatures but adds robustness, dtype/device safety,
# deterministic grid_crop fallback, and a resilient CatBackbone.coord_head implementation.

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*grid_sampler_2d_backward_cuda does not have a deterministic implementation.*",
    category=UserWarning
)

import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

# ----------------------------
# Squeeze-and-Excitation (single unified definition)
# ----------------------------
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
        y = self.pool(x)
        y = self.fc(y)
        return x * y


# ----------------------------
# Conv3D helper
# ----------------------------
class Conv3D(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=(1, k, k), stride=(1, s, s), padding=(0, p, p), bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv(x)
        return x.squeeze(2)


# ----------------------------
# Simple MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.linear = nn.Linear(c1, c2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ----------------------------
# BiFPN2 (two-input weighted fusion)
# ----------------------------
class BiFPN2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, f"BiFPN2 expects list/tuple of 2 tensors, got {type(x)} len={len(x)}"
        a, b = x
        if a.shape[1:] != b.shape[1:]:
            raise RuntimeError(f"BiFPN2 inputs must have same (C,H,W). Got {a.shape} and {b.shape}")
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        fused = weight[0] * a + weight[1] * b
        return fused


# ----------------------------
# BiFPN (three-input weighted fusion)
# ----------------------------
class BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, f"BiFPN expects list/tuple of 3 tensors, got {type(x)} len={len(x)}"
        s0, s1, s2 = x
        if s0.shape[1:] != s1.shape[1:] or s0.shape[1:] != s2.shape[1:]:
            raise RuntimeError(f"BiFPN inputs must share shape (C,H,W): {s0.shape}, {s1.shape}, {s2.shape}")
        w = F.relu(self.w)
        weight = w / (w.sum() + self.eps)
        fused = weight[0] * s0 + weight[1] * s1 + weight[2] * s2
        return fused


# ----------------------------
# BiFPNConcat
# ----------------------------
class BiFPNConcat(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPNConcat, self).__init__()
        self.d = dimension

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) >= 2, "BiFPNConcat expects a list/tuple of tensors"
        base_dim = x[0].dim()
        for t in x:
            if t.dim() != base_dim:
                raise RuntimeError(f"BiFPNConcat input rank mismatch: {t.dim()} vs {base_dim}")
        return torch.cat(x, dim=self.d)


# ----------------------------
# DataSwitch (noop)
# ----------------------------
class DataSwitch(nn.Module):
    def __init__(self, c1):
        super().__init__()

    def forward(self, x):
        return x


# ----------------------------
# DWT wrapper (robust)
# ----------------------------
class DWT(nn.Module):
    def __init__(self, c1, c2):
        super(DWT, self).__init__()
        try:
            from pytorch_wavelets import DWTForward
            self._use_pwavelets = True
            self.dwt = DWTForward(J=1, wave='sym4', mode='periodization')
        except Exception:
            self._use_pwavelets = False
            self.dwt = None
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, bias=False)
        self.eps = 1e-6

    def forward(self, x):
        if not self._use_pwavelets:
            Yl = F.avg_pool2d(x, kernel_size=2, stride=2)
            LH = x[..., ::2, 1::2] if x.size(-1) > 1 else x[..., ::2, ::2]
            HL = x[..., 1::2, ::2] if x.size(-2) > 1 else x[..., ::2, ::2]
            HH = x[..., 1::2, 1::2] if x.size(-2) > 1 and x.size(-1) > 1 else x[..., ::2, ::2]
            LH = F.adaptive_avg_pool2d(LH, Yl.shape[-2:])
            HL = F.adaptive_avg_pool2d(HL, Yl.shape[-2:])
            HH = F.adaptive_avg_pool2d(HH, Yl.shape[-2:])
        else:
            Yl, Yh = self.dwt(x)
            if isinstance(Yh[0], torch.Tensor) and Yh[0].dim() == 5 and Yh[0].shape[2] == 3:
                LH = Yh[0][:, :, 0]
                HL = Yh[0][:, :, 1]
                HH = Yh[0][:, :, 2]
            elif isinstance(Yh[0], (list, tuple)) and len(Yh[0]) == 3:
                LH, HL, HH = Yh[0]
            else:
                raise ValueError(f"DWT expected Yh[0] to contain 3 tensors, got {type(Yh[0])} with shape {getattr(Yh[0], 'shape', 'N/A')}")
        x_cat = torch.cat([Yl, LH, HL, HH], dim=1)
        return self.conv(x_cat)


# ----------------------------
# PConv (robust to c2 not divisible by 4)
# ----------------------------
class PConv(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
        self.c2 = c2

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        parts = [yw0, yw1, yh0, yh1]
        out = torch.cat(parts, dim=1)
        if out.shape[1] != self.c2:
            c_out = out.shape[1]
            if c_out < self.c2:
                pad_ch = self.c2 - c_out
                pad_tensor = out[:, :pad_ch, :, :].clone()
                out = torch.cat([out, pad_tensor], dim=1)
            else:
                out = out[:, :self.c2, :, :]
        return self.cat(out)


# ----------------------------
# SEBlock and DSConvSE
# ----------------------------
class SEBlock(nn.Module):
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
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k//2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.SiLU()
        self.se = SEBlock(c1, reduction=2)
        self.pconv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.dwconv(x)))
        x = self.se(x)
        x = self.act2(self.bn2(self.pconv(x)))
        return x


# ----------------------------
# HaarDWT (dtype/device safe)
# ----------------------------
class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5],
                           [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5],
                           [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]], dtype=torch.float32)
        weight = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer('weight', weight)

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.weight.to(x.dtype).to(x.device)
        weight = weight.repeat(C, 1, 1, 1)
        out = F.conv2d(x, weight, stride=2, groups=C)
        return out


# ----------------------------
# DWTBackbone
# ----------------------------
class DWTBackbone(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.conv1 = nn.Conv2d(c1*4, 64, 1, bias=False)
        self.conv2 = nn.Conv2d(64*4, 128, 1, bias=False)
        self.conv3 = nn.Conv2d(128*4, 256, 1, bias=False)
        self.conv4 = nn.Conv2d(256*4, 512, 1, bias=False)
        self.conv5 = nn.Conv2d(512*4, 1024, 1, bias=False)
        self.dwt = HaarDWT()

    def forward(self, x):
        p1 = self.conv1(self.dwt(x))
        p2 = self.conv2(self.dwt(p1))
        p3 = self.conv3(self.dwt(p2))
        p4 = self.conv4(self.dwt(p3))
        p5 = self.conv5(self.dwt(p4))
        return [p3, p4, p5]


# ----------------------------
# grid_crop deterministic fallback
# ----------------------------
def grid_crop(img, pos_xy, crop_size):
    B, C, H, W = img.shape
    sz = max(int(crop_size), 8)
    if sz % 2 == 1:
        sz += 1
    # normalized coords not used directly for integer-centered deterministic crop,
    # but kept to maintain signature compatibility for other callers.
    cx = torch.clamp(pos_xy[:, 0].round().long(), 0, W - 1)
    cy = torch.clamp(pos_xy[:, 1].round().long(), 0, H - 1)

    half = sz // 2
    min_x = (cx - half).min().item()
    min_y = (cy - half).min().item()
    max_x = (cx + half - 1).max().item()
    max_y = (cy + half - 1).max().item()

    pad_left = max(0, -min_x)
    pad_top = max(0, -min_y)
    pad_right = max(0, max_x - (W - 1))
    pad_bottom = max(0, max_y - (H - 1))

    if pad_left or pad_top or pad_right or pad_bottom:
        img_padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        cx = cx + pad_left
        cy = cy + pad_top
    else:
        img_padded = img

    _, _, Hp, Wp = img_padded.shape
    crops = []
    for i in range(B):
        x_c = int(cx[i].item())
        y_c = int(cy[i].item())
        x1 = x_c - half
        y1 = y_c - half
        x1 = max(0, min(x1, Wp - sz))
        y1 = max(0, min(y1, Hp - sz))
        x2 = x1 + sz
        y2 = y1 + sz
        crop = img_padded[i:i+1, :, y1:y2, x1:x2]
        if crop.shape[2] != crop_size or crop.shape[3] != crop_size:
            crop = F.interpolate(crop, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        crops.append(crop)
    out = torch.cat(crops, dim=0)
    return out


# ----------------------------
# CropBlock
# ----------------------------
class CropBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.se = SEBlock(in_channels, reduction=2)
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.se(x)
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


# ----------------------------
# CatBackbone with robust coord_head (adaptive handling)
# ----------------------------
class CatBackbone(nn.Module):
    """
    CatBackbone:
      - Signature unchanged: __init__(in_channels_img, in_channels_feat)
      - coord_head outputs a channel tensor that is interpreted robustly as per-axis
        heatmaps pooled to a fixed small grid (fallback to 4x4).
      - Uses deterministic grid_crop above.
    """
    def __init__(self, in_channels_img, in_channels_feat):
        super().__init__()
        mid = max(in_channels_feat // 4, 16)
        self._in_channels_feat = in_channels_feat
        # flexible coord_head: outputs arbitrary even number of channels (ideally 2*(Gx*Gy))
        self.coord_head = nn.Sequential(
            nn.Conv2d(in_channels_feat, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2 * 16, kernel_size=1)  # default to 2*16 channels but code handles other counts
        )
        self.block1 = CropBlock(in_channels_img)
        self.block2 = CropBlock(in_channels_img)
        self.block3 = CropBlock(in_channels_img)
        self._debug_counter = 0

    def forward(self, x):
        image, feat = x  # image: [B,C_img,H,W], feat: [B,C_feat,hf,wf]
        B, C_img, H, W = image.shape

        # coord heatmap prediction
        hmap = self.coord_head(feat)  # [B, C_h, hf, wf]
        # stabilize spatial shape by pooling to small fixed grid (4x4)
        hmap = F.adaptive_avg_pool2d(hmap, (4, 4))  # [B, C_h, 4, 4]
        _, C_h, Hh, Wh = hmap.shape
        # flatten spatial
        hmap_flat = hmap.view(B, C_h, -1)  # [B, C_h, 16]

        # Expect channels to be split into two groups (axis x and axis y). If not, fallback gracefully.
        if C_h >= 2 and C_h % 2 == 0:
            half = C_h // 2
            hx = hmap_flat[:, :half, :].mean(dim=1)  # [B, 16]
            hy = hmap_flat[:, half:, :].mean(dim=1)  # [B, 16]
        else:
            # fallback: divide channels roughly in half
            half = max(1, C_h // 2)
            hx = hmap_flat[:, :half, :].mean(dim=1)
            hy = hmap_flat[:, half:, :].mean(dim=1)

        # argmax per axis over the 4x4 grid (16 cells)
        idx_x = hx.argmax(dim=-1)  # [B]
        idx_y = hy.argmax(dim=-1)  # [B]

        grid_side = 4
        gx_x = (idx_x % grid_side).float() / (grid_side - 1)
        gy_x = (idx_x // grid_side).float() / (grid_side - 1)
        gx_y = (idx_y % grid_side).float() / (grid_side - 1)
        gy_y = (idx_y // grid_side).float() / (grid_side - 1)

        # fuse estimates from both axis groups (reduce quantization)
        pos_x = ((gx_x + gx_y) * 0.5) * (W - 1)
        pos_y = ((gy_x + gy_y) * 0.5) * (H - 1)
        pos_xy = torch.stack([pos_x, pos_y], dim=1)  # [B,2]

        # crop sizes
        s1 = max(H // 4, 1)
        s2 = max(H // 8, 1)
        s3 = max(H // 16, 1)

        # deterministic crops
        patch1 = grid_crop(image, pos_xy, s1)
        patch2 = grid_crop(image, pos_xy, s2)
        patch3 = grid_crop(image, pos_xy, s3)

        ret1 = self.block1(patch1)
        ret2 = self.block2(patch2)
        ret3 = self.block3(patch3)

        # low-frequency debug logging
        if self.training:
            self._debug_counter += 1
            if self._debug_counter % 500 == 0:
                with torch.no_grad():
                    mean_pos = pos_xy.mean().item()
                    std_pos = pos_xy.std().item()
                    n1 = ret1.norm().item()
                    n2 = ret2.norm().item()
                    n3 = ret3.norm().item()
                    print(f"[CatBackbone debug] pos mean/std: {mean_pos:.4f}/{std_pos:.4f}, patch norms: {n1:.4f},{n2:.4f},{n3:.4f}")

        return [ret1, ret2, ret3]


# ----------------------------
# GetFeature
# ----------------------------
class GetFeature(nn.Module):
    def __init__(self, index: int, out_channels: int):
        super().__init__()
        self.index = index
        self.out_channels = out_channels

    def forward(self, features):
        return features[self.index]



################## 基于改进 YOLOv8 算法的 CT 图像肺结节检测研究 论文复现 ###########################
# custom_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class PolarizedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PSA(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        self.c1 = c1
        c2 = c2 or c1
        self.channels = c1

        # 简化版PSA，避免复杂的维度问题
        self.conv1 = Conv(c1, c1 // 4, 1)
        self.conv2 = Conv(c1 // 4, c1, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 8, 1),
            nn.ReLU(),
            nn.Conv2d(c1 // 8, c1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # 通道注意力
        attn = self.attention(x)
        x = x * attn

        # 空间注意力简化
        x = self.conv1(x)
        x = self.conv2(x)

        return identity + x


class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 简化的可变形注意力
        self.offset_conv = nn.Conv2d(dim, 2 * num_heads, 3, padding=1)
        self.value_conv = nn.Conv2d(dim, dim, 3, padding=1)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成偏移量
        offsets = self.offset_conv(x)  # (B, 2*num_heads, H, W)
        offsets = offsets.view(B, self.num_heads, 2, H, W)

        # 生成value
        v = self.value_conv(x)  # (B, C, H, W)
        v = v.view(B, self.num_heads, self.head_dim, H, W)

        # 简化的可变形采样（使用网格采样）
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=x.device),
                                            torch.arange(W, device=x.device), indexing='ij')
        coords = torch.stack([x_coords, y_coords], dim=0).float()  # (2, H, W)
        coords = coords.unsqueeze(0).unsqueeze(0)  # (1, 1, 2, H, W)

        # 应用偏移
        deformed_coords = coords + offsets * 0.1  # 小幅度偏移

        # 归一化坐标到 [-1, 1]
        deformed_coords = deformed_coords.permute(0, 1, 3, 4, 2)  # (B, num_heads, H, W, 2)
        deformed_coords[..., 0] = 2.0 * deformed_coords[..., 0] / (W - 1) - 1.0
        deformed_coords[..., 1] = 2.0 * deformed_coords[..., 1] / (H - 1) - 1.0

        # 对每个head进行采样
        output = []
        for i in range(self.num_heads):
            sampled = F.grid_sample(
                v[:, i],  # (B, head_dim, H, W)
                deformed_coords[:, i],  # (B, H, W, 2)
                align_corners=True,
                mode='bilinear'
            )
            output.append(sampled)

        x = torch.cat(output, dim=1)  # (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class DAT(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        self.c1 = c1
        c2 = c2 or c1

        # 简化的可变形注意力
        self.deform_attn = DeformableAttention(c1)
        self.conv = Conv(c1, c2, 1)

    def forward(self, x):
        identity = x
        x = self.deform_attn(x)
        x = self.conv(x)
        return identity + x  # 残差连接