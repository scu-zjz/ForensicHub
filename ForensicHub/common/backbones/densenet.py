import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model
from IMDLBenCo.modules.extractors.sobel import SobelFilter
from IMDLBenCo.modules.extractors.bayar_conv import BayerConv
from IMDLBenCo.modules.extractors.high_frequency_feature_extraction import FFTExtractor, DCTExtractor


class DenseNet(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',
                 backbone='densenet121',
                 pretrained=True,
                 image_size=224,
                 num_channels=3):
        super(DenseNet, self).__init__()

        assert 'densenet' in backbone, "Backbone must be one of timm DenseNet variants"

        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        out_channels = self.model.num_features
        self.output_type = output_type
        self.backbone = self.model.forward_features

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
            first_conv = self.model.conv0  # DenseNet 第一层是 conv0
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
            self.model.conv0 = new_conv

    def forward(self, image, *args, **kwargs):
        if self.input_head is not None:
            feat = self.input_head(image)
            x = torch.cat([image, feat], dim=1)
        else:
            x = image

        x = self.backbone(x)
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


@register_model("DenseNet121")
class DenseNet121(DenseNet):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=None,
                         output_type=output_type,
                         backbone='densenet121',
                         pretrained=pretrained,
                         image_size=image_size,
                         num_channels=0)


@register_model("SobelDenseNet121")
class SobelDenseNet121(DenseNet):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=SobelFilter(),
                         output_type=output_type,
                         backbone='densenet121',
                         pretrained=pretrained,
                         image_size=image_size,
                         num_channels=1)


@register_model("BayerDenseNet121")
class BayerDenseNet121(DenseNet):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=BayerConv(),
                         output_type=output_type,
                         backbone='densenet121',
                         pretrained=pretrained,
                         image_size=image_size,
                         num_channels=3)


@register_model("FFTDenseNet121")
class FFTDenseNet121(DenseNet):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=FFTExtractor(),
                         output_type=output_type,
                         backbone='densenet121',
                         pretrained=pretrained,
                         image_size=image_size,
                         num_channels=3)


@register_model("DCTDenseNet121")
class DCTDenseNet121(DenseNet):
    def __init__(self, output_type='label', pretrained=True, image_size=224):
        super().__init__(input_head=DCTExtractor(),
                         output_type=output_type,
                         backbone='densenet121',
                         pretrained=pretrained,
                         image_size=image_size,
                         num_channels=3)
