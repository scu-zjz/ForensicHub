from .models.seg_hrnet import get_seg_model
from .models.seg_hrnet_config import get_cfg_defaults
from .models.NLCDetection_api import NLCDetection
import torch
import torch.nn as nn
import torch.nn.functional as F

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel

'''
Hierarchical fine-grained image forgery detection and localization
'''


@register_model("HiFi_Net")
class HiFi_Net(BaseModel):
    def __init__(self, use_laplacian=False, feat_dim=1024, pretrained=True):
        super(HiFi_Net, self).__init__()
        self.use_laplacian = use_laplacian
        self.feat_dim = feat_dim

        # Feature extraction network (HRNet)
        self.FENet = get_seg_model(get_cfg_defaults())
        self.SegNet = NLCDetection()

    def forward(self, image, label, **kwargs):
        label = label.float()
        # Input: [B, 3, H, W]
        output = self.FENet(image)
        mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, image)
        pred_logits = out3[:, 1]
        combined_loss = F.binary_cross_entropy_with_logits(pred_logits, label)

        pred_label = torch.sigmoid(pred_logits)
        dict = {
            "backward_loss": combined_loss,
            "pred_label": pred_label,
            "visual_loss": {
                "combined_loss": combined_loss
            },
        }
        return dict
