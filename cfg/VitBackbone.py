import torch
import torch.nn as nn
from typing import List
import os
from torchinfo import summary


class VitBackbone(nn.Module):
    def __init__(self, repo_path: str, weight_path: str, size: str = 's', freeze: bool = False):
        super().__init__()

        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"DINOv3 repository not found at: {repo_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found at: {weight_path}")

        # 加载完整的DINOv3模型
        self.model = torch.hub.load(
            repo_or_dir=repo_path,
            model=f'dinov3_vit{size}16',
            source='local',
            weights=weight_path
        )

        dino_dim = self.model.embed_dim

        print(self.model.embed_dim)

        self._out_channels = dino_dim

        # 冻结参数（可选）
        if freeze:
            print(f"Freezing DINOv3 ViT-{size} backbone.")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # n=[1,2,3] 表示取第1、2、3个Transformer block的输出
        features_tuple = self.model.get_intermediate_layers(x, n=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], reshape = True)

        return list(features_tuple)
        return list(features_tuple)

    @property
    def out_channels(self):
        return self._out_channels

if __name__ == "__main__":
    DINOV3_REPO_PATH = "../dinov3"
    LOCAL_WEIGHT_PATH = "pth/dinov3_vits16.pth"
    backbone = VitBackbone(repo_path=DINOV3_REPO_PATH, weight_path=LOCAL_WEIGHT_PATH, size='s')
    backbone.eval()
    summary(backbone, (32, 3, 640, 640))