from typing import Dict, Any

from .models.clip import clip
import torch.nn as nn
import torch
import torch.nn.functional as F

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

'''
Towards Universal Fake Image Detectors that Generalize Across Generative Models
'''

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768
}


@register_model("UnivFD")
class UnivFD(BaseModel):
    def __init__(self, name='ViT-L/14', num_classes=1):
        super(UnivFD, self).__init__()

        # self.preprecess will not be used during training, which is handled in Dataset class
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.model.requires_grad_(False)
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, image, label, **kwargs) -> Dict[str, Any]:
        x = image
        label = label.float()

        x = self.model.encode_image(x)
        x = self.fc(x)
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
