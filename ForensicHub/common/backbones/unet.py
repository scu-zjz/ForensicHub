import torch
import torch.nn as nn
import torch.nn.functional as F

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model
from IMDLBenCo.modules.extractors.sobel import SobelFilter
from IMDLBenCo.modules.extractors.bayar_conv import BayerConv
from IMDLBenCo.modules.extractors.high_frequency_feature_extraction import (
    FFTExtractor, DCTExtractor
)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 8, base_channels * 8))

        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 16, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 8, base_channels * 2)
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 4, base_channels)
        self.up4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels * 2, base_channels)

        self.out_channels = base_channels

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))

        return x


class UNet(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='mask',
                 pretrained=False,
                 image_size=256,
                 num_channels=3):
        super(UNet, self).__init__()

        self.output_type = output_type
        self.input_head = input_head

        self.model = UNetBackbone(in_channels=num_channels + 3 if input_head else 3)
        out_channels = self.model.out_channels

        if output_type == 'label':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, 1)
            )
        elif output_type == 'mask':
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1),
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

        features = self.model(x)
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


@register_model("UNetBase")
class UNetBase(UNet):
    def __init__(self, output_type='mask', image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         image_size=image_size, num_channels=0)


@register_model("SobelUNet")
class SobelUNet(UNet):
    def __init__(self, output_type='mask', image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type,
                         image_size=image_size, num_channels=1)


@register_model("BayerUNet")
class BayerUNet(UNet):
    def __init__(self, output_type='mask', image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type,
                         image_size=image_size, num_channels=3)


@register_model("FFTUNet")
class FFTUNet(UNet):
    def __init__(self, output_type='mask', image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type,
                         image_size=image_size, num_channels=3)


@register_model("DCTUNet")
class DCTUNet(UNet):
    def __init__(self, output_type='mask', image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type,
                         image_size=image_size, num_channels=3)
