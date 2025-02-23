import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        layers = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            layers.append(act)
        super(BasicBlock, self).__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        layers = []
        for i in range(2):
            layers.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                layers.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                layers.append(act)
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res + x

class EDSR(nn.Module):
    def __init__(self, conv=default_conv):
        super(EDSR, self).__init__()
        n_resblocks = 32
        n_feats = 64
        kernel_size = 3
        n_colors = 3
        out_channels = 31
        act = nn.ReLU(True)
        
        self.head = nn.Sequential(conv(n_colors, n_feats, kernel_size))
        self.body = nn.Sequential(*[ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)],
                                  conv(n_feats, n_feats, kernel_size))
        self.tail = nn.Sequential(conv(n_feats, out_channels, kernel_size))

    def forward(self, x):
        x = self.head(x)
        res = self.body(x) + x
        return self.tail(res)
