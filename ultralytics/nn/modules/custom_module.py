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
# Conv3D helper (unchanged interface)
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
# Simple MLP (unchanged)
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
# BiFPN2 (two-input weighted fusion) -- add assertions and numeric stability
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
        # require same spatial and channel dims
        if a.shape[1:] != b.shape[1:]:
            raise RuntimeError(f"BiFPN2 inputs must have same (C,H,W). Got {a.shape} and {b.shape}")
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        fused = weight[0] * a + weight[1] * b
        return fused


# ----------------------------
# BiFPN (three-input weighted fusion) -- add assertions and numeric stability
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
        # require same channel/spatial shapes
        if s0.shape[1:] != s1.shape[1:] or s0.shape[1:] != s2.shape[1:]:
            raise RuntimeError(f"BiFPN inputs must share shape (C,H,W): {s0.shape}, {s1.shape}, {s2.shape}")
        w = F.relu(self.w)
        weight = w / (w.sum() + self.eps)
        fused = weight[0] * s0 + weight[1] * s1 + weight[2] * s2
        return fused


# ----------------------------
# BiFPNConcat (concat along given dim) -- add simple checks
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
# DataSwitch (noop) kept as-is
# ----------------------------
class DataSwitch(nn.Module):
    def __init__(self, c1):
        super().__init__()

    def forward(self, x):
        return x


# ----------------------------
# DWT (pytorch_wavelets) wrapper: keep signature but made more robust
# ----------------------------
import torch
import torch.nn as nn
# from pytorch_wavelets import DWTForward  # commented to avoid runtime error if not installed

class DWT(nn.Module):
    def __init__(self, c1, c2):
        super(DWT, self).__init__()
        # Keep signature unchanged. If pytorch_wavelets is available, use it.
        try:
            from pytorch_wavelets import DWTForward
            self._use_pwavelets = True
            self.dwt = DWTForward(J=1, wave='sym4', mode='periodization')
        except Exception:
            # fallback flag; user environment may not have pytorch_wavelets
            self._use_pwavelets = False
            self.dwt = None
        # 1x1 conv mapping from 4*c1 -> c2
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, bias=False)
        self.eps = 1e-6

    def forward(self, x):
        # x: [B, C, H, W]
        if not self._use_pwavelets:
            # safe approximate fallback: use simple 2x2 average pooling + handcrafted bands
            Yl = F.avg_pool2d(x, kernel_size=2, stride=2)
            # create 3 high-band approximations by simple shifts (cheap, deterministic)
            LH = x[..., ::2, 1::2] if x.size(-1) > 1 else x[..., ::2, ::2]
            HL = x[..., 1::2, ::2] if x.size(-2) > 1 else x[..., ::2, ::2]
            HH = x[..., 1::2, 1::2] if x.size(-2) > 1 and x.size(-1) > 1 else x[..., ::2, ::2]
            # make sure all are same spatial size as Yl by adaptive pooling
            LH = F.adaptive_avg_pool2d(LH, Yl.shape[-2:])
            HL = F.adaptive_avg_pool2d(HL, Yl.shape[-2:])
            HH = F.adaptive_avg_pool2d(HH, Yl.shape[-2:])
        else:
            Yl, Yh = self.dwt(x)
            # Yh[0] might be a tensor shaped [B, C, 3, H/2, W/2] or a list/tuple of 3 tensors
            if isinstance(Yh[0], torch.Tensor) and Yh[0].dim() == 5 and Yh[0].shape[2] == 3:
                LH = Yh[0][:, :, 0]
                HL = Yh[0][:, :, 1]
                HH = Yh[0][:, :, 2]
            elif isinstance(Yh[0], (list, tuple)) and len(Yh[0]) == 3:
                LH, HL, HH = Yh[0]
            else:
                raise ValueError(f"DWT expected Yh[0] to contain 3 tensors, got {type(Yh[0])} with shape {getattr(Yh[0], 'shape', 'N/A')}")
        # concat along channel dim -> [B, 4*C, H/2, W/2]
        x_cat = torch.cat([Yl, LH, HL, HH], dim=1)
        return self.conv(x_cat)


# ----------------------------
# PConv (Pinwheel) but robust to c2 not divisible by 4
# ----------------------------
class PConv(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        # Use ultralytics Conv for asymmetric kernels; keep interfaces unchanged
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
        self.c2 = c2

    def forward(self, x):
        # produce four parts; if c2 not divisible by 4, the intermediate convs produce floor(c2/4)
        # we will pad last part if needed before concat to reach c2 channels
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        parts = [yw0, yw1, yh0, yh1]
        # After concat, ensure the total channel count equals c2
        out = torch.cat(parts, dim=1)
        if out.shape[1] != self.c2:
            # pad or trim channels to match expected c2
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
# HaarDWT (GPU-friendly) with dtype/device safety
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
        weight = torch.stack([ll, lh, hl, hh]).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer('weight', weight)

    def forward(self, x):
        B, C, H, W = x.shape
        # move to correct device/dtype
        weight = self.weight.to(x.dtype).to(x.device)
        weight = weight.repeat(C, 1, 1, 1)  # (4*C,1,2,2)
        out = F.conv2d(x, weight, stride=2, groups=C)
        return out


# ----------------------------
# DWTBackbone: keep signature, ensure channel matching commentary and robust forward
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
# grid_crop utility: clamp crop size, evenize, align_corners=False
# ----------------------------
def grid_crop(img, pos_xy, crop_size):
    B, C, H, W = img.shape
    sz = max(int(crop_size), 8)
    if sz % 2 == 1:
        sz += 1
    # normalized coordinates in [-1,1]
    x_norm = (pos_xy[:, 0] / (W - 1)) * 2 - 1
    y_norm = (pos_xy[:, 1] / (H - 1)) * 2 - 1
    lin = torch.linspace(-1, 1, sz, device=img.device, dtype=img.dtype)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    grid[..., 0] += x_norm.view(B, 1, 1)
    grid[..., 1] += y_norm.view(B, 1, 1)
    # ensure values stay in [-1,1]
    grid = torch.clamp(grid, -1.0, 1.0)
    return F.grid_sample(img, grid, mode="bilinear", align_corners=False)


# ----------------------------
# SEBlock (already defined above as SEBlock) - reuse that; ensure only one SEBlock defined in file
# ----------------------------

# ----------------------------
# CropBlock and CatBackbone (keep signatures unchanged) with coord_head stability and debug hooks
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


class CatBackbone(nn.Module):
    def __init__(self, in_channels_img, in_channels_feat):
        super().__init__()
        # improved coord_head but signature unchanged
        self.coord_head = nn.Sequential(
            nn.Conv2d(in_channels_feat, max(in_channels_feat // 2, 8), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(max(in_channels_feat // 2, 8), 2, 1)
        )
        self.block1 = CropBlock(in_channels_img)
        self.block2 = CropBlock(in_channels_img)
        self.block3 = CropBlock(in_channels_img)
        # debug counters (low-frequency logging, non-intrusive)
        self._debug_counter = 0

    def forward(self, x):
        image, feat = x
        B, C_img, H, W = image.shape
        coord = self.coord_head(feat).view(B, 2).sigmoid()
        pos_x = coord[:, 0] * (W - 1)
        pos_y = coord[:, 1] * (H - 1)
        pos_xy = torch.stack([pos_x, pos_y], dim=1)

        s1 = max(H // 4, 1)
        s2 = max(H // 8, 1)
        s3 = max(H // 16, 1)

        # clamp and ensure even sizes with grid_crop's internal clamp
        patch1 = grid_crop(image, pos_xy, s1)
        patch2 = grid_crop(image, pos_xy, s2)
        patch3 = grid_crop(image, pos_xy, s3)

        ret1 = self.block1(patch1)
        ret2 = self.block2(patch2)
        ret3 = self.block3(patch3)

        # low-frequency debug logging to monitor pos collapse and patch norms
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
# GetFeature: simple indexed extractor (kept signature, no extra mapping)
# ----------------------------
class GetFeature(nn.Module):
    def __init__(self, index: int, out_channels: int):
        super().__init__()
        self.index = index
        self.out_channels = out_channels

    def forward(self, features):
        ret = features[self.index]
        return ret


# ----------------------------
# Utility Detect placeholder (left untouched - assume Detect exists elsewhere)
# ----------------------------
# Keep rest of model-head related classes (Conv, C3k2, Concat, Detect, etc.) provided by ultralytics environment

# ----------------------------
# End of modified module file
# ----------------------------
