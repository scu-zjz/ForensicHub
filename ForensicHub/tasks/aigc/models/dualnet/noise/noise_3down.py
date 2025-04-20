import torch
import torch.nn as nn
from ..utils.srm_filter_kernel import *
import torch.nn.functional as F


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


class residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.brunch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.blk(x)
        x2 = self.brunch(x)
        x = self.relu(x1 + x2)
        return x


class Pre(nn.Module):
    def __init__(self):
        super(Pre, self).__init__()
        hpf_list = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            hpf_list.append(hpf_item)
        # hpf_weight = nn.Parameter(torch.Tensor(hpf_list).view(30, 1, 5, 5), requires_grad=True)
        hpf_array = np.array(hpf_list)  # Efficient conversion
        hpf_weight = nn.Parameter(torch.Tensor(hpf_array).view(30, 1, 5, 5), requires_grad=True)
        self.hpf1 = nn.Conv2d(1, 30, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=False)
        self.hpf2 = nn.Conv2d(1, 30, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=False)
        self.hpf3 = nn.Conv2d(1, 30, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=False)
        self.hpf1.weight = hpf_weight
        self.hpf2.weight = hpf_weight
        self.hpf3.weight = hpf_weight

    def forward(self, x):
        r = self.hpf1(x[:, 0, :, :].unsqueeze(dim=1))
        g = self.hpf2(x[:, 1, :, :].unsqueeze(dim=1))
        b = self.hpf3(x[:, 2, :, :].unsqueeze(dim=1))
        pre = torch.cat((r, g, b), dim=1)
        return pre

class Trans_Noise(nn.Module):
    def __init__(self):
        super(Trans_Noise, self).__init__()
        self.pre = Pre()
        self.layers = nn.Sequential(
            Block2(90, 128),
            residual(128, 128),
            residual(128, 256)
        )

    def forward(self, inp):
        b, _, h, w = inp.size()
        x = self.pre(inp)
        x = self.layers(x)
        return x



