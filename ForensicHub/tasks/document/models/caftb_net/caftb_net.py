import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel

'''
Cross-Attention Based Two-Branch Networks for Document Image Forgery Localization in the Metaverse
https://dl.acm.org/doi/abs/10.1145/3686158
'''


class CSA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x)


class RSA(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x.mean(1, keepdim=True) + x.max(1, keepdim=True).values))


class CAFM(nn.Module):
    def __init__(self, in_channels, resize=1):
        super().__init__()
        self.csa = CSA(in_channels)
        self.rsa = RSA()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.BatchNorm2d(in_channels), nn.ReLU())
        self.resize = resize

    def forward(self, xc, xt):
        if self.resize == 1:
            return self.conv(self.csa(xc) + self.rsa(xt))
        else:
            return F.interpolate(self.conv(self.csa(xc) + self.rsa(xt)), scale_factor=self.resize, mode='bilinear')


class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''

    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = F.softmax(logits, 1)[:, 1]
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


@register_model("CAFTB_Net")
class CAFTB_Net(BaseModel):
    def __init__(self,
                 vit_pretrained_path='/mnt/data0/public_datasets/Doc/Hub/caftb-net/segformer-b5-640x640-ade-160k'):
        super(CAFTB_Net, self).__init__()
        cnn_model_name='resnetv2_50d_gn.ah_in1k'
        self.cnn = timm.create_model(cnn_model_name, pretrained=False, features_only=True)
        self.cnn.stem_conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.cnn.stages_3.blocks[0].downsample.pool = nn.Identity()
        self.cnn.stages_3.blocks[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                      bias=False)
        self.vit = smp.from_pretrained(vit_pretrained_path).encoder.eval()
        channels = [64, 256, 512, 1024]
        self.convert0 = nn.Sequential(nn.Conv2d(2048, channels[3], 1, 1, 0), nn.ReLU())
        self.convert1 = nn.Sequential(nn.Conv2d(128, channels[1], 1, 1, 0), nn.ReLU())
        self.convert2 = nn.Sequential(nn.Conv2d(320, channels[2], 1, 1, 0), nn.ReLU())
        self.convert3 = nn.Sequential(nn.Conv2d(512, channels[3], 1, 1, 0), nn.ReLU())
        self.cafms = nn.ModuleList([CAFM(channels[i], 2 ** i) for i in range(4)])
        self.decoder = nn.Sequential(nn.Conv2d(sum(channels), 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 2, 1, 1, 0))
        self.dice_loss = SoftDiceLossV1()

    def vit_forward(self, x):
        rsts = []
        x = self.vit.patch_embed1(x)
        x = self.vit.block1(x)
        x = self.vit.norm1(x)
        x = x.contiguous()
        rsts.append(x)

        x = self.vit.patch_embed2(x)
        x = self.vit.block2(x)
        x = self.vit.norm2(x)
        x = x.contiguous()
        rsts.append(self.convert1(x))

        x = self.vit.patch_embed3(x)
        x = self.vit.block3(x)
        x = self.vit.norm3(x)
        x = x.contiguous()
        rsts.append(self.convert2(x))

        x = self.vit.patch_embed4(x)
        x = self.vit.block4(x)
        x = self.vit.norm4(x)
        x = x.contiguous()
        rsts.append(self.convert3(x))

        return rsts

    def cal_seg_loss(self, pred, gt):
        h, w = gt.shape[-2:]
        pred = F.interpolate(pred, size=(h, w), mode='bilinear')
        ce_loss = (F.cross_entropy(pred, gt) * 0.3 + self.dice_loss(pred, gt) * 0.7)
        return ce_loss, pred

    def forward(self, image, mask, **kwargs):
        x = image
        mask = mask.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        xc = self.cnn(x)
        xc = [xc[0], xc[1], xc[2], self.convert0(xc[4])]
        xt = self.vit_forward(x)
        xc = torch.cat([self.cafms[i](xc[i], xt[i]) for i in range(4)], 1)
        output = self.decoder(xc)
        seg_loss, output = self.cal_seg_loss(output, mask)
        pred_mask = F.softmax(output, dim=1)[:,1:]
        output_dict = {
            "backward_loss": seg_loss,

            "pred_mask": pred_mask,

            "visual_loss": {
                "seg_loss": seg_loss,
                "combined_loss": seg_loss
            },
            "visual_image": {
                "pred_mask": pred_mask,
            }
        }
        return output_dict


if __name__ == "__main__":
    img = torch.ones((1, 3, 512, 512))
    mask = torch.ones((1, 1, 512, 512), dtype=torch.int64)
    model = CAFTB_Net()
    pred = model(img, mask)
    print(pred)
