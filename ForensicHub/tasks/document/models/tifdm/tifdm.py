#!/usr/bin/python3
# coding=utf-8
import timm
import numpy as np
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel

'''

'''


def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Softmax):
            pass
        else:
            m.initialize()


class EfficientNetB3(nn.Module):
    def __init__(self):
        super(EfficientNetB3, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 4)
        )
        layers = [MBConvBlock(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(MBConvBlock(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = F.max_pool2d(F.relu(out1), kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('efficientnet-b3-5fb5a3c3.pth'), strict=False)


class MBConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(MBConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class LHSE(nn.Module):
    def __init__(self):
        super(LHSE, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse = out2h * out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True) + out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True) + out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)


class TFAM(nn.Module):
    def __init__(self, in_channels, out_channels, tfam_channels=64):
        super(TFAM, self).__init__()
        # Series of 1x1 conv to generate attention feature maps
        self.pab_channels = tfam_channels
        self.in_channels = in_channels
        self.top_conv = nn.Conv2d(in_channels, tfam_channels, kernel_size=1)
        self.center_conv = nn.Conv2d(in_channels, tfam_channels, kernel_size=1)
        self.bottom_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.map_softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x_top = self.top_conv(x)
        x_center = self.center_conv(x)
        x_bottom = self.bottom_conv(x)

        x_top = x_top.flatten(2)
        x_center = x_center.flatten(2).transpose(1, 2)
        x_bottom = x_bottom.flatten(2).transpose(1, 2)

        sp_map = torch.matmul(x_center, x_top)
        sp_map = self.map_softmax(sp_map.view(bsize, -1)).view(bsize, h * w, h * w)
        sp_map = torch.matmul(sp_map, x_bottom)
        sp_map = sp_map.reshape(bsize, self.in_channels, h, w)
        x = x + sp_map
        x = self.out_conv(x)
        return x

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lhse45 = LHSE()
        self.lhse34 = LHSE()
        self.lhse23 = LHSE()
        self.tfam = TFAM(64, 64, tfam_channels=16)  # 如果显存溢出可减小tfam_channels

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5 = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4 = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3 = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2 = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v = out5v + refine5
            out4h, out4v = self.lhse45(out4h + refine4, out5v)
            out3h, out3v = self.lhse34(out3h + refine3, out4v)
            out2h, pred = self.lhse23(out2h + refine2, out3v)
        else:
            out4h, out4v = self.lhse45(out4h, out5v)
            out3h, out3v = self.lhse34(out3h, out4v)
            out2h, pred = self.lhse23(out2h, out3v)

        pred = self.tfam(pred)
        return pred

    def initialize(self):
        weight_init(self)


@register_model("Tifdm")
class Tifdm(BaseModel):
    def __init__(self, backbone_model_name='resnet50'):
        super(Tifdm, self).__init__()
        self.backbone = timm.create_model(backbone_model_name, features_only=True, pretrained=True)
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder = Decoder()
        self.linearp = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def cal_seg_loss(self, pred, gt):
        h, w = gt.shape[-2:]
        pred = F.interpolate(pred, size=(h, w), mode='bilinear')
        ce_loss = F.cross_entropy(pred, gt)
        return ce_loss, pred

    def forward(self, image, mask, **kwargs):
        x = image
        mask = mask.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        out2h, out3h, out4h, out5v = self.backbone(x)[1:]
        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(
            out5v)
        pred1 = self.decoder(out2h, out3h, out4h, out5v)
        output = self.linearp(pred1)
        seg_loss, output = self.cal_seg_loss(output, mask)
        output = F.softmax(output,dim=1)[:,1:]
        output_dict = {
            "backward_loss": seg_loss,
            "pred_mask": output,
            "visual_loss": {
                "seg_loss": seg_loss,
                "combined_loss": seg_loss
            },
            "visual_image": {
                "pred_mask": output,
            }
        }
        return output_dict

    def initialize(self):
        weight_init(self.decoder)


if __name__ == "__main__":
    model = Tifdm()
    img = torch.ones((1, 3, 64, 64))
    mask = torch.ones((1, 1, 64, 64), dtype=torch.int64)
    pred = model(img, mask)
    print(pred)
