import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .extractors.fph import FPH

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model
from IMDLBenCo.modules.extractors.high_frequency_feature_extraction import (
    FFTExtractor,
    DCTExtractor
)
from IMDLBenCo.modules.extractors.sobel import SobelFilter
from IMDLBenCo.modules.extractors.bayar_conv import BayerConv


class Resnet(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',  # label or mask
                 backbone='resnet101',  # resnet50, reset101 or reset152
                 pretrained=True,  # if true, use imagenet-1k pretrained weight
                 image_size=256,  # to set mask size when output_type is mask
                 num_channels=3):
        super(Resnet, self).__init__()
        assert backbone in ['resnet50', 'resnet101', 'resnet152'], "Only resnet50, reset101 or reset152 are supported"
        self.model = timm.create_model(backbone, pretrained=pretrained)
        self.backbone = self.model.forward_features
        out_channels = self.model.num_features
        self.head = None
        self.output_type = output_type

        if output_type == 'label':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, 1)
            )
        elif output_type == 'mask':
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, kernel_size=1),
                nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
            )
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        original_first_layer = list(self.model.children())[0]
        if input_head is not None:
            self.input_head = input_head
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size,
                                        stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(
                    new_first_layer.weight[:, 3:, :, :])
            self.model.conv1 = new_first_layer
        else:
            self.input_head = None

    def forward(self, image, *args, **kwargs):
        if self.input_head is not None:
            feature = self.input_head(image)
            x = torch.cat([image, feature], dim=1)
        else:
            x = image
        out = self.head(self.backbone(x))

        if self.output_type == 'label':
            if len(out.shape) == 2:
                out = out.squeeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(out, kwargs['label'].float())
            pred = out.sigmoid()
        else:
            loss = F.binary_cross_entropy_with_logits(out, kwargs['mask'].float())
            pred = out.sigmoid()
        out_dict = {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {
                "combined_loss": loss
            }
        }

        return out_dict


@register_model("Resnet50")
class Resnet50(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, backbone='resnet50', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("Resnet101")
class Resnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("Resnet152")
class Resnet152(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, backbone='resnet152', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("SobelResnet101")
class SobelResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type, backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=1)


@register_model("BayerResnet101")
class BayerResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type, backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("FFTResnet101")
class FFTResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type, backbone='resnet101',
                         pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("DCTResnet101")
class DCTResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type, backbone='resnet101',
                         pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("QtResnet101")
class QtResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=0)
        self.fph = FPH()

        # 定位主干中间模块
        self.stem = nn.Sequential(
            self.model.conv1,  # conv1
            self.model.bn1,
            self.model.act1,
            self.model.maxpool
        )
        # 主干后续部分
        self.blocks = nn.Sequential(
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        )

        # 通道融合（拼接后通道不一致）
        in_channels = self.model.layer1[0].conv1.in_channels + 256  # 通常是 64 + 256
        self.fusion_conv = nn.Conv2d(in_channels, self.model.layer1[0].conv1.in_channels, kernel_size=1)

    def forward(self, image, dct, qt, *args, **kwargs):
        dct = dct.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        # FPH 特征（B, 256, H/8, W/8）
        x_aux = self.fph(dct, qt)

        # 主干前部分
        x = self.stem(image)

        # 若尺寸不一致，则插值对齐
        if x.shape[-2:] != x_aux.shape[-2:]:
            x_aux = F.interpolate(x_aux, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 拼接并降维融合
        x = torch.cat([x, x_aux], dim=1)
        x = self.fusion_conv(x)

        # 继续主干
        x = self.blocks(x)
        out = self.head(x)

        if self.output_type == 'label':
            if len(out.shape) == 2:
                out = out.squeeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(out, kwargs['label'].float())
            pred = out.sigmoid()
        else:
            loss = F.binary_cross_entropy_with_logits(out, kwargs['mask'].float())
            pred = out.sigmoid()

        return {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {"combined_loss": loss}
        }
