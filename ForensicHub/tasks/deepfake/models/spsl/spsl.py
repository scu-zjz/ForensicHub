import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import logging

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel
from timm.models import xception


@register_model("Spsl")
class Spsl(BaseModel):
    def __init__(self, yaml_config_path=None):
        super(Spsl, self).__init__()
        with open(yaml_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # 构建模型和损失函数
        self.backbone = self.build_backbone()
        self.loss_func = self.build_loss()

    def build_backbone(self):
        return xception(num_classes=self.config['backbone_config']['num_classes'],
                        in_chans=self.config['backbone_config']['inc'], pretrained=True)

    def build_loss(self):
        # 固定为 CrossEntropyLoss
        return nn.CrossEntropyLoss()

    def phase_without_amplitude(self, img):
        # 计算没有幅度的相位信息
        gray_img = torch.mean(img, dim=1, keepdim=True)
        X = torch.fft.fftn(gray_img, dim=(-1, -2))
        phase_spectrum = torch.angle(X)
        reconstructed_X = torch.exp(1j * phase_spectrum)
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X, dim=(-1, -2)))
        return reconstructed_x

    def forward(self, image, label, **kwargs):
        label = label.long()  # 转换为 long 类型用于 cross entropy
        phase_fea = self.phase_without_amplitude(image)
        features_input = torch.cat((image, phase_fea), dim=1)
        features = self.backbone.forward_features(features_input)
        pred = self.backbone.forward_head(features)
        prob = torch.softmax(pred, dim=1)[:, 1]

        # 计算损失
        combined_loss = self.loss_func(pred, label)
        result = {
            "backward_loss": combined_loss,
            "pred_label": prob,
            "visual_loss": {
                "combined_loss": combined_loss
            },
        }
        return result
