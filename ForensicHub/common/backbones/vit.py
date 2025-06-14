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


class VisionTransformer(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',
                 backbone='vit_base_patch16_224',
                 pretrained=True,
                 image_size=224,
                 num_channels=3):
        super(VisionTransformer, self).__init__()

        assert 'vit' in backbone, "Backbone must be a ViT model from timm"

        self.model = timm.create_model(backbone, pretrained=pretrained)
        self.output_type = output_type
        self.input_head = input_head

        # 修改第一层以适配自定义通道数
        if input_head is not None:
            patch_embed = self.model.patch_embed
            new_proj = nn.Conv2d(num_channels + 3,
                                 patch_embed.proj.out_channels,
                                 kernel_size=patch_embed.proj.kernel_size,
                                 stride=patch_embed.proj.stride,
                                 padding=patch_embed.proj.padding,
                                 bias=False)
            new_proj.weight.data[:, :3, :, :] = patch_embed.proj.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_proj.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(
                    new_proj.weight[:, 3:, :, :])
            patch_embed.proj = new_proj

        # 获取特征维度
        out_channels = self.model.head.in_features

        if output_type == 'label':
            self.head = nn.Linear(out_channels, 1)
        elif output_type == 'mask':
            # ViT 通常不适合 mask 回归，但为了统一格式，这里做基础实现
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 2, 1, 1),
                nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
            )
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

    def forward(self, image, *args, **kwargs):
        if self.input_head is not None:
            feat = self.input_head(image)
            x = torch.cat([image, feat], dim=1)
        else:
            x = image

        if self.output_type == 'label':
            features = self.model.forward_features(x)
            if isinstance(features, torch.Tensor) and features.ndim == 3:
                cls_token = features[:, 0]  # 取 CLS token
            else:
                raise ValueError("Unexpected ViT output shape")

            out = self.head(cls_token)
            if len(out.shape) == 2:
                out = out.squeeze(dim=1)

            loss = F.binary_cross_entropy_with_logits(out, kwargs['label'].float())
            pred = out.sigmoid()
        else:
            raise NotImplementedError("Mask output not supported for vanilla ViT")

        return {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {"combined_loss": loss}
        }


# 注册各种变种
@register_model("ViTBase")
class ViTBase(VisionTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='vit_base_patch16_224',
                         pretrained=pretrained, image_size=image_size, num_channels=0)


@register_model("SobelViTBase")
class SobelViTBase(VisionTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=SobelFilter(), output_type=output_type,
                         backbone='vit_base_patch16_224',
                         pretrained=pretrained, image_size=image_size, num_channels=1)


@register_model("BayerViTBase")
class BayerViTBase(VisionTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=BayerConv(), output_type=output_type,
                         backbone='vit_base_patch16_224',
                         pretrained=pretrained, image_size=image_size, num_channels=3)


@register_model("FFTViTBase")
class FFTViTBase(VisionTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=FFTExtractor(), output_type=output_type,
                         backbone='vit_base_patch16_224',
                         pretrained=pretrained, image_size=image_size, num_channels=3)


@register_model("DCTViTBase")
class DCTViTBase(VisionTransformer):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=DCTExtractor(), output_type=output_type,
                         backbone='vit_base_patch16_224',
                         pretrained=pretrained, image_size=image_size, num_channels=3)
