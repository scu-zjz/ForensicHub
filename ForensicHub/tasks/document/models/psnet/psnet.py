#!/usr/bin/python3
# coding=utf-8
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.01

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel

'''
Progressive Supervision for Tampering Localization in Document Images
https://link.springer.com/chapter/10.1007/978-981-99-8184-7_11
'''


class NonLocalMask(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMask, self).__init__()

        self.r = reduce_scale

        # input channel number
        self.ic = in_channels * self.r * self.r

        # middle channel number
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)

        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.W_c = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))

        self.gamma_c = nn.Parameter(torch.ones(1))

        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            value :
                f: B X (HxW) X (HxW)
                ic: intermediate channels
                z: feature maps( B X C X H X W)
            output:
                mask: feature maps( B X 1 X H X W)
        """

        b, c, h, w = x.shape

        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

        # g x
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta
        theta_x = self.theta(x1).view(b, self.mc, -1)

        theta_x_s = theta_x.permute(0, 2, 1)
        theta_x_c = theta_x

        # phi x
        phi_x = self.phi(x1).view(b, self.mc, -1)

        phi_x_s = phi_x
        phi_x_c = phi_x.permute(0, 2, 1)

        # non-local attention
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)

        f_c = torch.matmul(theta_x_c, phi_x_c)
        f_c_div = F.softmax(f_c, dim=-1)

        # get y_s
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous()
        y_s = y_s.view(b, c, h, w)

        # get y_c
        y_c = torch.matmul(g_x, f_c_div)
        y_c = y_c.view(b, c, h, w)

        # get z
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)

        # get mask
        mask = torch.sigmoid(self.getmask(z.clone()))

        return mask, z


class NLCDetection(nn.Module):
    def __init__(self, ):
        super(NLCDetection, self).__init__()

        feat1_num, feat2_num, feat3_num, feat4_num = 64, 128, 256, 512

        self.getmask4 = NonLocalMask(feat4_num, 1)
        self.getmask3 = NonLocalMask(feat3_num, 2)
        self.getmask2 = NonLocalMask(feat2_num, 2)
        self.getmask1 = NonLocalMask(feat1_num, 4)

    def forward(self, feat):
        """
            inputs :
                feat : a list contains features from s1, s2, s3, s4
            output:
                mask1: output mask ( B X 1 X H X W)
                pred_cls: output cls (B X 4)
        """
        s1, s2, s3, s4 = feat

        mask4, z4 = self.getmask4(s4)
        mask4U = F.interpolate(mask4, size=s3.size()[2:], mode='bilinear', align_corners=True)

        s3 = s3 * mask4U
        mask3, z3 = self.getmask3(s3)
        mask3U = F.interpolate(mask3, size=s2.size()[2:], mode='bilinear', align_corners=True)

        s2 = s2 * mask3U
        mask2, z2 = self.getmask2(s2)
        mask2U = F.interpolate(mask2, size=s1.size()[2:], mode='bilinear', align_corners=True)

        s1 = s1 * mask2U
        mask1, z1 = self.getmask1(s1)

        return [mask1, mask2, mask3, mask4]


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
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


class DetectionHead(nn.Module):
    def __init__(self, ):
        super(DetectionHead, self).__init__()

        pre_stage_channels = (64, 128, 256, 512)

        # classification head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        self.classifier = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = pre_stage_channels

        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, y_list):

        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + \
                self.downsamp_modules[i](y)

        y = self.final_layer(y)

        # average and flatten
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        logit = self.classifier(y).squeeze(1)

        return logit


class BayerConv(nn.Module):
    def __init__(self, in_channel=3):
        super(BayerConv, self).__init__()
        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=2, bias=False)

        # Convert tensors to Parameters
        bayar_mask = np.ones(shape=(5, 5))
        bayar_mask[2, 2] = 0
        self.bayar_mask = nn.Parameter(torch.tensor(bayar_mask, dtype=torch.float32), requires_grad=False)

        bayar_final = np.zeros((5, 5))
        bayar_final[2, 2] = -1
        self.bayar_final = nn.Parameter(torch.tensor(bayar_final, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        return self.BayarConv2D(x)


def lbp_torch(x):
    # pad image for 3x3 mask size
    x = F.pad(input=x, pad=[1, 1, 1, 1], mode='constant')
    b = x.shape
    M = b[1]
    N = b[2]

    y = x
    # select elements within 3x3 mask
    y00 = y[:, 0:M - 2, 0:N - 2]
    y01 = y[:, 0:M - 2, 1:N - 1]
    y02 = y[:, 0:M - 2, 2:N]
    #
    y10 = y[:, 1:M - 1, 0:N - 2]
    y11 = y[:, 1:M - 1, 1:N - 1]
    y12 = y[:, 1:M - 1, 2:N]
    #
    y20 = y[:, 2:M, 0:N - 2]
    y21 = y[:, 2:M, 1:N - 1]
    y22 = y[:, 2:M, 2:N]

    # Apply comparisons and multiplications
    bit = torch.ge(y01, y11)
    tmp = torch.mul(bit, torch.tensor(1))

    bit = torch.ge(y02, y11)
    val = torch.mul(bit, torch.tensor(2))
    val = torch.add(val, tmp)

    bit = torch.ge(y12, y11)
    tmp = torch.mul(bit, torch.tensor(4))
    val = torch.add(val, tmp)

    bit = torch.ge(y22, y11)
    tmp = torch.mul(bit, torch.tensor(8))
    val = torch.add(val, tmp)

    bit = torch.ge(y21, y11)
    tmp = torch.mul(bit, torch.tensor(16))
    val = torch.add(val, tmp)

    bit = torch.ge(y20, y11)
    tmp = torch.mul(bit, torch.tensor(32))
    val = torch.add(val, tmp)

    bit = torch.ge(y10, y11)
    tmp = torch.mul(bit, torch.tensor(64))
    val = torch.add(val, tmp)

    bit = torch.ge(y00, y11)
    tmp = torch.mul(bit, torch.tensor(128))
    val = torch.add(val, tmp)
    return val[..., 1:-1]


@register_model("PSNet")
class PSNet(BaseModel):
    def __init__(self, ):
        super(PSNet, self).__init__()
        self.bayar = BayerConv()
        self.backbone = timm.create_model('hrnet_w18', features_only=True, pretrained=True)
        self.backbone.conv1 = nn.Conv2d(7, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv = nn.Conv2d(1024, 512, 1, 1, 0)
        self.cls = DetectionHead()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.decoder = NLCDetection()
        self.kl_loss = torch.nn.KLDivLoss()
        self.maxp = nn.AdaptiveMaxPool2d(1)

    def cal_seg_loss(self, pred, gt):
        h, w = gt.shape[-2:]
        pred = F.interpolate(pred, size=(h, w), mode='bilinear')
        ce_loss = F.binary_cross_entropy_with_logits(pred, gt)
        return ce_loss, pred

    def forward(self, image, mask, **kwargs):
        x = image
        lbp = lbp_torch(x)
        bayar = (x - self.bayar(x))
        bayer = torch.where(bayar > 0, x, 1 - x)
        x = torch.cat((x, bayar, lbp), 1)
        feats = self.backbone(x)
        feats[-2] = (feats[-2] + self.upsample(self.conv(feats[-1])))
        del feats[-1]
        output = self.decoder(feats)
        cls_output = self.cls(feats)
        seg_loss, seg_pred = self.cal_seg_loss(output[0], mask)
        output[0] = seg_pred
        for i in range(3):
            seg_loss = (seg_loss + self.cal_seg_loss(output[i + 1], mask)[0])
        kl_loss = F.kl_div(output[0].flatten(1).max(1).values, cls_output)
        cls_loss = F.binary_cross_entropy_with_logits(cls_output, mask.flatten(1).max(1).values)
        combined_loss = (seg_loss + cls_loss + kl_loss)
        output_dict = {
            "backward_loss": combined_loss,
            "pred_mask": seg_pred,
            "pred_label": cls_output,
            "visual_loss": {
                "seg_loss": seg_loss,
                "cls_loss": cls_loss,
                "kl_loss": kl_loss,
                "combined_loss": combined_loss
            },
            "visual_image": {
                "pred_mask": seg_pred,
            }
        }
        return output_dict


if __name__ == "__main__":
    model = PSNet()
    img = torch.ones((1, 3, 64, 64))
    mask = torch.ones((1, 1, 64, 64), dtype=torch.float32)
    pred = model(img, mask)  # pred[0] for final segmentation prediction
    print(pred)
