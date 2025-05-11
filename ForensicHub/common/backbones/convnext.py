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


class ConvNext(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',
                 backbone='convnext_base',  # 可以改为 tiny/small/large/xlarge
                 pretrained=True,
                 image_size=256,
                 num_channels=3):
        super(ConvNext, self).__init__()

        assert 'convnext' in backbone, "Backbone must be one of timm ConvNext variants"

        self.model = timm.create_model(backbone, pretrained=pretrained, features_only=False)
        self.backbone = self.model.forward_features
        out_channels = self.model.num_features
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

        self.input_head = input_head
        if input_head is not None:
            first_conv = self.model.stem[0]  # ConvNext 通常第一层在 stem[0]
            new_conv = nn.Conv2d(num_channels + 3,
                                 first_conv.out_channels,
                                 kernel_size=first_conv.kernel_size,
                                 stride=first_conv.stride,
                                 padding=first_conv.padding,
                                 bias=False)
            new_conv.weight.data[:, :3, :, :] = first_conv.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_conv.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(
                    new_conv.weight[:, 3:, :, :])
            self.model.stem[0] = new_conv

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


@register_model("ConvNextSmall")
class ConvNextSmall(ConvNext):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='convnext_small', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("SobelConvNextSmall")
class SobelConvNextSmall(ConvNext):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type,
                         backbone='convnext_small', pretrained=pretrained,
                         image_size=image_size, num_channels=1)


@register_model("BayerConvNextSmall")
class BayerConvNextSmall(ConvNext):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type,
                         backbone='convnext_small', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("FFTConvNextSmall")
class FFTConvNextSmall(ConvNext):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type,
                         backbone='convnext_small', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("DCTConvNextSmall")
class DCTConvNextSmall(ConvNext):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type,
                         backbone='convnext_small', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("QtConvNextSmall")
class QtConvNextSmall(ConvNext):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='convnext_small', pretrained=pretrained,
                         image_size=image_size, num_channels=0)

        self.fph = FPH()

        # ConvNeXt的前几个模块：stem, stages
        self.stem = self.model.stem
        self.stages = self.model.stages  # 分层结构

        # 拼接 FPH 后进行通道融合
        stem_out_channels = self.stem[0].out_channels  # ConvNeXt 的 stem 输出通道
        self.fusion_conv = nn.Conv2d(stem_out_channels + 256, stem_out_channels, kernel_size=1)

    def forward(self, image, dct, qt, *args, **kwargs):
        dct = dct.squeeze(1).long()  # [B, 1, H, W] -> [B, H, W]
        x_aux = self.fph(dct, qt)  # -> [B, 256, H/8, W/8]

        x = self.stem(image)  # ConvNeXt 的 stem，输出一般为 H/4
        if x.shape[-2:] != x_aux.shape[-2:]:
            x_aux = F.interpolate(x_aux, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 融合特征：concat + 1x1 conv 降维
        x = torch.cat([x, x_aux], dim=1)
        x = self.fusion_conv(x)

        # 继续 ConvNeXt 的主干
        for stage in self.stages:
            x = stage(x)

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
