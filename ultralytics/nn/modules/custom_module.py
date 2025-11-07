from torch import nn

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
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False
        )

    def forward(self, x):
        # x: [B, c1, H, W]
        x = x.unsqueeze(2)              # → [B, c1, 1, H, W]
        x = self.conv(x)                # → [B, c2, 1, H/2, W/2]
        return x.squeeze(2)             # → [B, c2, H/2, W/2]
