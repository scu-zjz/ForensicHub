import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .extractors.fph import FPH
from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model
from IMDLBenCo.modules.extractors.high_frequency_feature_extraction import (
    FFTExtractor,
    DCTExtractor
)
from IMDLBenCo.modules.extractors.sobel import SobelFilter
from IMDLBenCo.modules.extractors.bayar_conv import BayerConv


class MobileNet(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',
                 backbone='mobilenetv3_large_100',
                 pretrained=True,
                 image_size=256,
                 num_channels=3):
        super(MobileNet, self).__init__()

        assert 'mobilenet' in backbone, "Backbone must be one of timm MobileNet variants"

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
            first_conv = self.model.conv_stem  # MobileNetV3 的第一层卷积
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
            self.model.conv_stem = new_conv

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

        return {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {"combined_loss": loss}
        }


# 注册各种 MobileNet 模型
@register_model("MobileNetV3")
class MobileNetV3(MobileNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='mobilenetv3_large_100',
                         pretrained=pretrained, image_size=image_size, num_channels=0)


@register_model("SobelMobileNetV3")
class SobelMobileNetV3(MobileNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type,
                         backbone='mobilenetv3_large_100',
                         pretrained=pretrained, image_size=image_size, num_channels=1)


@register_model("BayerMobileNetV3")
class BayerMobileNetV3(MobileNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type,
                         backbone='mobilenetv3_large_100',
                         pretrained=pretrained, image_size=image_size, num_channels=3)


@register_model("FFTMobilenetV3")
class FFTMobilenetV3(MobileNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type,
                         backbone='mobilenetv3_large_100',
                         pretrained=pretrained, image_size=image_size, num_channels=3)


@register_model("DCTMobilenetV3")
class DCTMobilenetV3(MobileNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type,
                         backbone='mobilenetv3_large_100',
                         pretrained=pretrained, image_size=image_size, num_channels=3)
