from typing import Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

'''
synthbuster: Towards detection of diffusion model generated images
'''


def cross_difference(image: torch.Tensor) -> torch.Tensor:
    """
    计算 cross-difference（交叉差分）高通滤波。
    输入: image, shape: [B, C, H, W]
    输出: residual, shape: [B, C, H-1, W-1]
    """
    # Get diagonals
    I = image
    I_tl = I[:, :, :-1, :-1]  # Top-left
    I_br = I[:, :, 1:, 1:]  # Bottom-right
    I_tr = I[:, :, :-1, 1:]  # Top-right
    I_bl = I[:, :, 1:, :-1]  # Bottom-left

    # Cross-difference: abs(I[x,y] + I[x+1,y+1] - I[x,y+1] - I[x+1,y])
    diff = torch.abs(I_tl + I_br - I_tr - I_bl)
    return diff


def fft_features(cross_diff: torch.Tensor, periods=[0, 2, 4, 8]) -> torch.Tensor:
    """
    对交叉差分图像做 FFT 并提取频率点幅值作为特征。
    输入: cross_diff, shape: [B, C, H, W]
    输出: features, shape: [B, C * len(periods) * 2]
    """
    B, C, H, W = cross_diff.shape

    # Compute FFT
    fft = torch.fft.fft2(cross_diff)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))  # center the zero freq

    # Compute magnitude spectrum
    magnitude = torch.abs(fft_shifted)

    # Center coordinates
    center_h, center_w = H // 2, W // 2

    features = []
    for p in periods:
        for axis in [0, 1]:  # vertical and horizontal
            if axis == 0:
                coords = (center_h + p, center_w)
            else:
                coords = (center_h, center_w + p)
            f = magnitude[:, :, coords[0], coords[1]]  # shape [B, C]
            features.append(f)

    # Concatenate features: [B, C * len(periods) * 2]
    return torch.cat(features, dim=1)


@register_model("synthbuster")
class Synthbuster(BaseModel):
    def __init__(self):
        super(Synthbuster, self).__init__()
        self.fc1 = nn.Linear(24, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, image, label, **kwargs) -> Dict[str, Any]:
        label = label.float()
        cd = cross_difference(image)
        features = fft_features(cd)
        features = torch.log1p(features)  # log 缩放
        # print(features)
        x = self.fc1(features)
        x = self.fc2(x)
        x = x.squeeze(dim=1) if x.ndim == 2 and x.shape[1] == 1 else x
        combined_loss = F.binary_cross_entropy_with_logits(x, label)
        pred_label = x.sigmoid()

        dict = {
            "backward_loss": combined_loss,
            "pred_label": pred_label,
            "visual_loss": {
                "combined_loss": combined_loss
            }
        }
        return dict
