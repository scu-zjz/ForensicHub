from efficientnet_pytorch.utils import *
import os
import logging
import functools
import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import collections

BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio','input_filters', 'output_filters', 'se_ratio', 'id_skip'])
GlobalParams = collections.namedtuple('GlobalParams', ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate','num_classes', 'batch_norm_momentum', 'batch_norm_epsilon','drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])
global_params = GlobalParams(width_coefficient=1.8, depth_coefficient=2.6, image_size=528, dropout_rate=0.0, num_classes=1000, batch_norm_momentum=0.99, batch_norm_epsilon=0.001, drop_connect_rate=0.0, depth_divisor=8, min_depth=None, include_top=True)

def get_width_and_height_from_size(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()

def calculate_output_image_size(input_image_size, stride):
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]

class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, image_size=25):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class AddCoords(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r
    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim,dtype=input_tensor.dtype), torch.arange(y_dim,dtype=input_tensor.dtype))
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size,1,x_dim,y_dim)
        yy_c = yy_c.expand(batch_size,1,x_dim,y_dim)
        ret = torch.cat((input_tensor,xx_c,yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret

class FPH(nn.Module):

    def __init__(self):
        super(FPH, self).__init__()
        self.obembed = nn.Embedding(21,21).from_pretrained(torch.eye(21))
        self.qtembed = nn.Embedding(64,16)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=21,out_channels=64,kernel_size=3,stride=1,dilation=8,padding=8),nn.BatchNorm2d(64, momentum=0.01),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),nn.BatchNorm2d(16, momentum=0.01),nn.ReLU(inplace=True))
        self.addcoords = AddCoords()
        repeats = (1,1,1)
        in_channles = (256,256,256)
        out_channles = (256,256,512)
        self.conv0 = nn.Sequential(nn.Conv2d(in_channels=35, out_channels=256, kernel_size=8, stride=8, padding=0, bias=False),nn.BatchNorm2d(256, momentum=0.01),nn.ReLU(inplace=True),MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6, input_filters=in_channles[0], output_filters=in_channles[1], se_ratio=0.25, id_skip=True), global_params),MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6, input_filters=in_channles[1], output_filters=in_channles[1], se_ratio=0.25, id_skip=True), global_params),MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6, input_filters=in_channles[1], output_filters=in_channles[1], se_ratio=0.25, id_skip=True), global_params),)

    def forward(self, x, qtable):
        x = self.conv2(self.conv1(self.obembed(x).permute(0,3,1,2).contiguous()))
        B, C, H, W = x.shape
        return self.conv0(self.addcoords(torch.cat(((x.reshape(B,C,H//8,8,W//8,8).permute(0,1,3,5,2,4)*self.qtembed(qtable.unsqueeze(-1).unsqueeze(-1).long()).transpose(1,6).squeeze(6).contiguous()).permute(0,1,4,2,5,3).reshape(B,C,H,W),x), dim=1)))

if __name__=="__main__":
    fph = FPH()
    dct = torch.ones((1,64,64),dtype=torch.int64)
    qt = torch.ones((1,1,8,8),dtype=torch.int64)
    pred = fph(dct, qt)
    print(pred.shape)
