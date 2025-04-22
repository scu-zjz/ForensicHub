import torch
import torch.nn as nn
from ..utils.srm_filter_kernel import *
import torch.nn.functional as F


class Block_(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(Block_, self).__init__()
        self.process = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)])
        for i in range(depth-1):
            self.process.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False))

    def forward(self, x):
        out = x
        for conv in self.process:
            out = conv(x)
        return out

class Pre4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.blk2 = Block_(6, 6, depth=1)
        self.blk3 = Block_(6, 6, depth=1)

    def forward(self, x):
        f1 = self.conv1(x)
        f1 = torch.cat((f1, x), dim=1)
        f2 = self.blk2(f1)
        f = f1-f2
        f3 = self.blk3(f)
        f = torch.cat((f, f3), dim=1)
        return f

class Block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block2, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )

    def forward(self, x):
        out = self.process(x)
        return out

class plain(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(plain, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
    def forward(self, x):
        x = self.blk(x)
        return x


class Pre2(nn.Module):
    def __init__(self):
        super(Pre2, self).__init__()
        self.pre = Pre4()
        self.layers = nn.Sequential(
            Block2(12, 64),
            plain(64, 128),
            plain(128, 256),
        )


    def forward(self, inp):
        b, _, h, w = inp.size()
        x = self.pre(inp)
        x = self.layers(x)
        return x
