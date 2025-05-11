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


class EfficientNet(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',
                 backbone='efficientnet_b4',
                 pretrained=True,
                 image_size=256,
                 num_channels=3):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, features_only=False)
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

        original_first_layer = self.model.conv_stem
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
            self.model.conv_stem = new_first_layer
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


# 注册 EfficientNet-B4 及其变种
@register_model("EfficientB4")
class EfficientB4(EfficientNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, backbone='efficientnet_b4', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("SobelEfficientB4")
class SobelEfficientB4(EfficientNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type, backbone='efficientnet_b4',
                         pretrained=pretrained, image_size=image_size, num_channels=1)


@register_model("BayerEfficientB4")
class BayerEfficientB4(EfficientNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type, backbone='efficientnet_b4',
                         pretrained=pretrained, image_size=image_size, num_channels=3)


@register_model("FFTEfficientB4")
class FFTEfficientB4(EfficientNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type, backbone='efficientnet_b4',
                         pretrained=pretrained, image_size=image_size, num_channels=3)


@register_model("DCTEfficientB4")
class DCTEfficientB4(EfficientNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type, backbone='efficientnet_b4',
                         pretrained=pretrained, image_size=image_size, num_channels=3)


@register_model("QtEfficientB4")
class QtEfficientB4(EfficientNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        # 初始化基类（不需要 input_head，因为我们使用的是自定义 fph）
        super().__init__(input_head=None, output_type=output_type,
                         backbone='efficientnet_b4', pretrained=pretrained,
                         image_size=image_size, num_channels=0)

        self.fph = FPH()

        # 获取 conv_stem 的输出通道数，用于融合
        stem_out_channels = self.model.conv_stem.out_channels

        # 构建融合卷积层：接收 conv_stem + FPH 的特征
        self.fusion_conv = nn.Conv2d(stem_out_channels + 256, stem_out_channels, kernel_size=1)

    def forward(self, image, dct, qt, *args, **kwargs):
        # FPH 输入前处理
        dct = dct.squeeze(1).long()  # e.g. [B, 1, H, W] -> [B, H, W]
        fph_feat = self.fph(dct, qt)  # 输出 shape: [B, 256, H/8, W/8]

        # 提取图像基础特征
        x = self.model.conv_stem(image)
        x = self.model.bn1(x)
        # 如果 act_layer 存在，则调用它；否则使用 F.silu
        if hasattr(self.model, 'act_layer'):
            x = self.model.act_layer(x)
        else:
            x = F.silu(x)

        # 对 FPH 特征进行对齐插值（如果尺寸不一致）
        if fph_feat.shape[-2:] != x.shape[-2:]:
            fph_feat = F.interpolate(fph_feat, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 特征融合 + 卷积压缩
        x = torch.cat([x, fph_feat], dim=1)
        x = self.fusion_conv(x)

        # 手动走 EfficientNet 的后续 block
        x = self.model.blocks(x)
        x = self.model.conv_head(x)
        x = self.model.bn2(x)
        # 如果 act_layer 存在，则调用它；否则使用 F.silu
        if hasattr(self.model, 'act_layer'):
            x = self.model.act_layer(x)
        else:
            x = F.silu(x)

        # 预测输出
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
