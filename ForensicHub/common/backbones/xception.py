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


class XceptionNet(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',
                 pretrained=True,
                 image_size=256,
                 num_channels=3):
        super(XceptionNet, self).__init__()
        self.model = timm.create_model('legacy_xception', pretrained=pretrained)
        self.output_type = output_type
        self.input_head = input_head

        # 替换第一层以适配 input_head 融合
        if input_head is not None:
            original_first_layer = self.model.conv1
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size,
                                        stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(
                    new_first_layer.weight[:, 3:, :, :]
                )
            self.model.conv1 = new_first_layer

        # 提取分类头的输出维度
        out_channels = self.model.get_classifier().in_features

        # 重建 head
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

    def forward(self, image, *args, **kwargs):
        if self.input_head is not None:
            feature = self.input_head(image)
            x = torch.cat([image, feature], dim=1)
        else:
            x = image

        # 手动分离出 features，兼容 legacy xception
        features = self.model.forward_features(x)
        out = self.head(features)

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


@register_model("Xception")
class Xception(XceptionNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("SobelXception")
class SobelXception(XceptionNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type, pretrained=pretrained,
                         image_size=image_size, num_channels=1)


@register_model("BayerXception")
class BayerXception(XceptionNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type, pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("FFTXception")
class FFTXception(XceptionNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type, pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("DCTXception")
class DCTXception(XceptionNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type, pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("QtXception")
class QtXception(XceptionNet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         pretrained=pretrained, image_size=image_size, num_channels=0)
        self.fph = FPH()

        # Xception的结构分解
        # stem部分：conv1 + bn1 + act1 + conv2 + bn2 + act2
        self.stem = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.act1,
            self.model.conv2,
            self.model.bn2,
            self.model.act2
        )

        # 主体blocks部分：从block1到block12
        self.blocks = nn.Sequential(
            self.model.block1,
            self.model.block2,
            self.model.block3,
            self.model.block4,
            self.model.block5,
            self.model.block6,
            self.model.block7,
            self.model.block8,
            self.model.block9,
            self.model.block10,
            self.model.block11,
            self.model.block12
        )

        self.tail = nn.Sequential(
            self.model.conv3,
            self.model.bn3,
            self.model.act3,
            self.model.conv4,
            self.model.bn4,
            self.model.act4,
        )

        # 通道融合层（Xception的stem输出通道通常是64）
        in_channels = 64 + 256  # stem输出64通道 + FPH的256通道
        self.fusion_conv = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, image, dct, qt, *args, **kwargs):
        dct = dct.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        # FPH特征提取 (B, 256, H/8, W/8)
        x_aux = self.fph(dct, qt)

        # 主干前部分
        x = self.stem(image)

        # 尺寸对齐 (Xception的stem输出尺寸是输入的1/4)
        if x.shape[-2:] != x_aux.shape[-2:]:
            x_aux = F.interpolate(x_aux, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 拼接并融合
        x = torch.cat([x, x_aux], dim=1)
        x = self.fusion_conv(x)

        # 继续主干网络
        x = self.blocks(x)
        x = self.tail(x)
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
