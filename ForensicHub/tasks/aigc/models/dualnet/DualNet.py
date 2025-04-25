from .noise.noise_3down import *
from .rgb.rgb_3down import *
from .crosstrans2 import *
import torch.nn as nn

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

'''
Ai-generated image detection using a cross-attention enhanced dual-stream network
'''


@register_model("DualNet")
class DualNet(BaseModel):
    def __init__(self):
        super(DualNet, self).__init__()
        self.noise = Trans_Noise()
        self.rgb = Pre2()
        self.ct = crosstrans(depth=2, dim=256, hidden_dim=1024, heads=4, head_dim=64, dropout=0.1)
        self.n_elayers = nn.Sequential(
            residual(256, 256),
            residual(256, 256)
        )
        self.r_elayers = nn.Sequential(
            plain(256, 256),
            plain(256, 256)
        )
        self.fc1 = nn.Linear(512, 1)

    def forward(self, image, label, **kwargs):
        x = image
        label = label.float()
        b, _, h, w = x.size()
        n = self.noise(x)
        r = self.rgb(x)
        n, r = self.ct(n, r)
        n = self.n_elayers(n)
        r = self.r_elayers(r)
        x = torch.cat((n, r), dim=1)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size()[0], -1)
        x = self.fc1(x)
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
