import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    def __init__(self, block, layers, num_classes=1000, n_input=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def forward(self, image, label, **kwargs):

        x = image
        label = label.float()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = x.squeeze(dim=1) if x.ndim == 2 and x.shape[1] == 1 else x
        combined_loss = F.binary_cross_entropy_with_logits(x, label)
        pred_label = x.sigmoid()

        out_dict = {
            "backward_loss": combined_loss,
            "pred_label": pred_label,
            "visual_loss": {
                "combined_loss": combined_loss
            }
        }

        return out_dict


def resnet(pretrained=False, backbone='resnet50', n_input=3, **kwargs):
    if backbone == 'resnet50':
        layers = [3, 4, 6, 3]
    elif backbone == 'resnet101':
        layers = [3, 4, 23, 3]
    else:
        ValueError("not support")
    model = ResNet(Bottleneck, layers, n_input=n_input, **kwargs)

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls[backbone])
        try:
            model.load_state_dict(pretrain_dict, strict=False)
        except:
            print("loss conv1")
            model_dict = {}
            for k, v in pretrain_dict.items():
                if k in pretrain_dict and 'conv1' not in k:
                    model_dict[k] = v
            model.load_state_dict(model_dict, strict=False)
        print(f"load {backbone} pretrain success, url: {model_urls[backbone]}")
    return model


@register_model("Resnet18")
class Resnet18(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(Bottleneck, layers=[2, 2, 2, 2], **kwargs)
        if pretrained:
            state_dict = model_zoo.load_url(model_urls['resnet18'])
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.load_state_dict(state_dict, strict=False)
            print("Load Resnet pretrained weight.")


@register_model("Resnet34")
class Resnet34(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(Bottleneck, layers=[3, 4, 6, 3], **kwargs)
        if pretrained:
            state_dict = model_zoo.load_url(model_urls['resnet34'])
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.load_state_dict(state_dict, strict=False)
            print("Load Resnet pretrained weight.")


@register_model("Resnet50")
class Resnet50(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(Bottleneck, layers=[3, 4, 6, 3], **kwargs)
        if pretrained:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.load_state_dict(state_dict, strict=False)
            print("Load Resnet pretrained weight.")


@register_model("Resnet101")
class Resnet101(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(Bottleneck, layers=[3, 4, 23, 3], **kwargs)
        if pretrained:
            state_dict = model_zoo.load_url(model_urls['resnet101'])
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.load_state_dict(state_dict, strict=False)
            print("Load Resnet pretrained weight.")


@register_model("Resnet152")
class Resnet152(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(Bottleneck, layers=[3, 8, 36, 3], **kwargs)
        if pretrained:
            state_dict = model_zoo.load_url(model_urls['resnet152'])
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            self.load_state_dict(state_dict, strict=False)
            print("Load Resnet pretrained weight.")
