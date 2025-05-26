import torch
import jpegio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable
from .fph import FPH
from .dwt import DWTFPN
from .backbone_convnext import ConvNeXt
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from typing import Optional, Union, List, Dict, Any

from ForensicHub.registry import register_model
from ForensicHub.core.base_model import BaseModel

'''

'''


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_norm_layer(
        use_norm: Union[bool, str, Dict[str, Any]], out_channels: int
) -> nn.Module:
    supported_norms = ("inplace", "batchnorm", "identity", "layernorm", "instancenorm")

    # Step 1. Convert tot dict representation

    ## Check boolean
    if use_norm is True:
        norm_params = {"type": "batchnorm"}
    elif use_norm is False:
        norm_params = {"type": "identity"}

    ## Check string
    elif isinstance(use_norm, str):
        norm_str = use_norm.lower()
        if norm_str == "inplace":
            norm_params = {
                "type": "inplace",
                "activation": "leaky_relu",
                "activation_param": 0.0,
            }
        elif norm_str in supported_norms:
            norm_params = {"type": norm_str}
        else:
            raise ValueError(
                f"Unrecognized normalization type string provided: {use_norm}. Should be in "
                f"{supported_norms}"
            )

    ## Check dict
    elif isinstance(use_norm, dict):
        norm_params = use_norm

    else:
        raise ValueError(
            f"Invalid type for use_norm should either be a bool (batchnorm/identity), "
            f"a string in {supported_norms}, or a dict like {{'type': 'batchnorm', **kwargs}}"
        )

    # Step 2. Check if the dict is valid
    if "type" not in norm_params:
        raise ValueError(
            f"Malformed dictionary given in use_norm: {use_norm}. Should contain key 'type'."
        )
    if norm_params["type"] not in supported_norms:
        raise ValueError(
            f"Unrecognized normalization type string provided: {use_norm}. Should be in {supported_norms}"
        )
    if norm_params["type"] == "inplace" and InPlaceABN is None:
        raise RuntimeError(
            "In order to use `use_norm='inplace'` the inplace_abn package must be installed. Use:\n"
            "  $ pip install -U wheel setuptools\n"
            "  $ pip install inplace_abn --no-build-isolation\n"
            "Also see: https://github.com/mapillary/inplace_abn"
        )

    # Step 3. Initialize the norm layer
    norm_type = norm_params["type"]
    norm_kwargs = {k: v for k, v in norm_params.items() if k != "type"}

    if norm_type == "inplace":
        norm = InPlaceABN(out_channels, **norm_kwargs)
    elif norm_type == "batchnorm":
        norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
    elif norm_type == "identity":
        norm = nn.Identity()
    elif norm_type == "layernorm":
        norm = nn.LayerNorm(out_channels, **norm_kwargs)
    elif norm_type == "instancenorm":
        norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        raise ValueError(f"Unrecognized normalization type: {norm_type}")

    return norm


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int = 0,
            stride: int = 1,
            use_batchnorm: bool = True,
    ):
        if use_batchnorm:
            norm = get_norm_layer(use_batchnorm, out_channels)

        is_identity = isinstance(norm, nn.Identity)
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=is_identity,
        )

        activation = nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(conv, norm, activation)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        ipt = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = ipt + self.drop_path(x)
        return x


class AddCoords(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim, dtype=input_tensor.dtype),
                                    torch.arange(y_dim, dtype=input_tensor.dtype))
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim)
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class VPH(nn.Module):
    def __init__(self, dims=[96, 192], drop_path_rate=0.4, layer_scale_init_value=1e-6):
        super().__init__()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(6, dims[0], kernel_size=4, stride=4),
                                                              LayerNorm(dims[0], eps=1e-6,
                                                                        data_format="channels_first")), nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))])
        self.stages = nn.ModuleList([nn.Sequential(
            *[ConvBlock(dim=dims[0], drop_path=dp_rates[j], layer_scale_init_value=layer_scale_init_value) for j in
              range(3)]), nn.Sequential(
            *[ConvBlock(dim=dims[1], drop_path=dp_rates[3 + j], layer_scale_init_value=layer_scale_init_value) for j in
              range(3)])])
        self.apply(self._init_weights)

    def initnorm(self):
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(self.dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        outs = []
        x = self.stages[0](self.downsample_layers[0](x))
        outs = [self.norm0(x)]
        x = self.stages[1](self.downsample_layers[1](x))
        outs.append(self.norm1(x))
        return outs


class DecoderBlock(nn.Module):
    def __init__(self, cin, cadd, cout, ):
        super().__init__()
        self.cin = (cin + cadd)
        self.cout = cout
        self.conv1 = Conv2dReLU(self.cin, self.cout, kernel_size=3, padding=1, use_batchnorm=True)
        self.conv2 = Conv2dReLU(self.cout, self.cout, kernel_size=3, padding=1, use_batchnorm=True)

    def forward(self, x1, x2=None):
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:, :self.cin])
        x1 = self.conv2(x1)
        return x1


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, ks, stride=1, norm=True, res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False)
        self.res = res

    def forward(self, x):
        if self.res:
            return (x + self.conv(x))
        else:
            return self.conv(x)


class FUSE1(nn.Module):
    def __init__(self, in_channels_list=(96, 192, 384, 768)):
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 1)
        self.c32 = ConvBNReLU(in_channels_list[3], in_channels_list[2], 1)
        self.c33 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 3)

        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1, x2, x3 = x
        h, w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3), size=(h, w)) + self.c31(x2))
        h, w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2), size=(h, w)) + self.c21(x1))
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w)) + self.c11(x))
        return x, x1, x2, x3


class FUSE2(nn.Module):
    def __init__(self, in_channels_list=(96, 192, 384)):
        super(FUSE2, self).__init__()

        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1, x2 = x
        h, w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2), size=(h, w), mode='bilinear', align_corners=True) + self.c21(x1))
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w), mode='bilinear', align_corners=True) + self.c11(x))
        return x, x1, x2


class FUSE3(nn.Module):
    def __init__(self, in_channels_list=(96, 192)):
        super(FUSE3, self).__init__()

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1 = x
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w), mode='bilinear', align_corners=True) + self.c11(x))
        return x, x1


class MID(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [96]
        self.out_channels = decoder_channels
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(self.in_channels[-1], 0,
                                                                           self.out_channels[-1])
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, *features):
        decoder_features = {}
        features = self.fuse1(features)[::-1]
        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0], features[1])
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1], features[2])
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2], features[3])
        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2(
            (decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]))
        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](decoder_features["x_0_0"],
                                                                torch.cat((decoder_features["x_1_1"], features[2]), 1))
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](decoder_features["x_1_1"],
                                                                torch.cat((decoder_features["x_2_2"], features[3]), 1))
        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3(
            (decoder_features["x_1_2"], decoder_features["x_0_1"]))
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](decoder_features["x_0_1"], torch.cat(
            (decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]), 1))
        return self.decoder_convs["x_0_3"](
            torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"]), 1))


@register_model("FFDN")
class FFDN(BaseModel):
    def __init__(self, decoder_channels=(384, 192, 96, 64), classes=2,
                 weight_path='/mnt/data1/public_datasets/Doc/Hub/ffdn/convnext_small.pth'):
        super().__init__()
        self.vph = ConvNeXt()  # Backbone1, using stage 0, 1
        self.fph = FPH()
        self.decoder = MID(encoder_channels=(96, 192, 384, 768), decoder_channels=decoder_channels)
        self.head = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.Conv2d(256, 2, 1, 1, 0))
        self.addcoords = AddCoords()
        self.FU = nn.Sequential(SCSEModule(448), nn.Conv2d(448, 192, 3, 1, 1), nn.BatchNorm2d(192), nn.ReLU(True),
                                nn.Conv2d(192, 192, 1, 1, 0))
        self.FU[-1].weight.data.zero_()
        self.decoder = DWTFPN([96, 192, 384, 768], 256)
        self.init_vph(weight_path)

    def init_vph(self, weight_path):
        weights = torch.load(weight_path)['state_dict']
        dels = [k for k in weights.keys() if not k.startswith('backbone.')]
        for k in dels:
            del weights[k]
        new_weights = {}
        for k, v in weights.items():
            new_weights['.'.join(k.split('.')[1:])] = v
        self.vph.load_state_dict(new_weights)

    def cal_seg_loss(self, pred, gt):
        h, w = gt.shape[-2:]
        pred = F.interpolate(pred, size=(h, w), mode='bilinear')
        ce_loss = F.cross_entropy(pred, gt)
        return ce_loss, pred

    def forward(self, image, DCT_coef, qtables, mask, **kwargs):
        x = image
        mask = mask.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        DCT_coef = DCT_coef.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        if len(qtables.shape) == 3:
            qtables = qtables.unsqueeze(1)
        features = self.vph.forward_features(x, end_index=2)
        features[1] = self.FU(torch.cat((features[1], self.fph(DCT_coef, qtables)), 1)) + features[1]
        features.extend(self.vph.forward_features(features[1], start_index=2, end_index=4))
        decoder_output = self.decoder(features)
        output = self.head(decoder_output[0])
        seg_loss, output = self.cal_seg_loss(output, mask)
        output = F.softmax(output, dim=1)   
        output = output[:, 1:]
        output_dict = {
            "backward_loss": seg_loss,
            "pred_mask": output,
            "visual_loss": {
                "seg_loss": seg_loss,
            },
            "visual_image": {
                "pred_mask": output,
            }
        }
        return output_dict


if __name__ == "__main__":
    img = torch.ones((1, 3, 512, 512))
    mask = torch.ones((1, 1, 512, 512), dtype=torch.int64)
    dct = torch.ones((1, 1, 512, 512), dtype=torch.int64)  # From JPEGIO
    qt = torch.ones((1, 1, 8, 8), dtype=torch.int64)  # From JPEGIO
    model = FFDN()
    pred = model(img, dct, qt, mask)
    print(pred)
