import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractors.fph import FPH

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model
from IMDLBenCo.modules.extractors.high_frequency_feature_extraction import (
    FFTExtractor,
    DCTExtractor
)
from IMDLBenCo.modules.extractors.sobel import SobelFilter
from IMDLBenCo.modules.extractors.bayar_conv import BayerConv


class SwinTransformer(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',  # 'label' or 'mask'
                 backbone='swin_base_patch4_window7_224',
                 pretrained=True,
                 image_size=256,
                 num_channels=3):
        super(SwinTransformer, self).__init__()

        self.model = timm.create_model(backbone, pretrained=pretrained, img_size=image_size)

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

        # Extend input channels if input_head is used
        if input_head is not None:
            self.input_head = input_head
            original_first_layer = self.model.patch_embed.proj
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size,
                                        stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(
                    new_first_layer.weight[:, 3:, :, :])
            self.model.patch_embed.proj = new_first_layer
        else:
            self.input_head = None

    def forward(self, image, *args, **kwargs):
        if self.input_head is not None:
            feature = self.input_head(image)
            x = torch.cat([image, feature], dim=1)
        else:
            x = image
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
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
            "visual_loss": {
                "combined_loss": loss
            }
        }


@register_model("SwinSmall")
class SwinSmall(SwinTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='swin_small_patch4_window7_224', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("SobelSwinSmall")
class SobelSwinSmall(SwinTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type,
                         backbone='swin_small_patch4_window7_224', pretrained=pretrained,
                         image_size=image_size, num_channels=1)


@register_model("BayerSwinSmall")
class BayerSwinSmall(SwinTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type,
                         backbone='swin_small_patch4_window7_224', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("FFTSwinSmall")
class FFTSwinSmall(SwinTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type,
                         backbone='swin_small_patch4_window7_224', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("DCTSwinSmall")
class DCTSwinSmall(SwinTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type,
                         backbone='swin_small_patch4_window7_224', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("QtSwinSmall")
class QtSwinSmall(SwinTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='swin_small_patch4_window7_224', pretrained=pretrained,
                         image_size=image_size, num_channels=0)
        self.fph = FPH()

        # 分解Swin Transformer结构
        self.patch_embed = self.model.patch_embed
        self.layers = nn.ModuleList([
            self.model.layers[0],
            self.model.layers[1],
            self.model.layers[2],
            self.model.layers[3]
        ])
        self.norm = self.model.norm

        # 融合层 (Swin Small第一层输出通道是96)
        self.fusion_conv = nn.Conv2d(96 + 256, 96, kernel_size=1)

        # 重建head确保维度匹配
        out_channels = self.model.num_features
        if output_type == 'label':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, kernel_size=1),
                nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
            )

    def forward(self, image, dct, qt, *args, **kwargs):
        dct = dct.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        # FPH特征提取 (B, 256, H/8, W/8)
        x_aux = self.fph(dct, qt)

        # 主干初始部分
        x = self.patch_embed(image)  # [B, H/4*W/4, C=96]

        # 调整形状用于融合 [B, H/4, W/4, C] -> [B, C, H/4, W/4]
        x = x.permute(0, 3, 1, 2)

        # 尺寸对齐
        if x.shape[-2:] != x_aux.shape[-2:]:
            x_aux = F.interpolate(x_aux, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 拼接并融合
        x = torch.cat([x, x_aux], dim=1)
        x = self.fusion_conv(x)

        x = x.permute(0, 2, 3, 1)

        # 继续Swin Transformer主体
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # 调整形状 [B, H*W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

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
