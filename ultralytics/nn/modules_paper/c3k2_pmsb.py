"""C3k2_PMSB: C3k2 block enhanced with Partial Multi-Scale Bottleneck (PMSB).

Neck enhancement module. The standard bottleneck inside C3k2 is replaced
by PMSB, which applies Partial Diluted/Group Convolution (PDw/GConv) across
multiple kernel sizes to build a progressive multi-scale feature pyramid
while retaining the split-and-fuse C3k2 structure and the Boolean c3k
interface.
"""

import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_tf_

from ultralytics.nn.modules.block import C2f, C3k

__all__ = ["C3k2_PMSB"]


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=""):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == "normal":
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "trunc_normal":
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "relu6":
        layer = nn.ReLU6(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(n_prelu, inplace)
    elif act == "gelu":
        layer = nn.GELU()
    elif act == "hswish":
        layer = nn.Hardswish(inplace)
    elif act == "hsigmoid":
        layer = nn.Hardsigmoid()
    elif act == "swish":
        layer = nn.SiLU()
    elif act == "sigmoid":
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class Partial_conv3(nn.Module):
    """Partial Diluted/Group Convolution (PDw/GConv).

    Applies convolution only to a 1/n_div subset of channels and
    transmits the rest unchanged.
    """

    def __init__(self, dim, n_div, k, forward="split_cat"):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, k, 1, k // 2, groups=self.dim_conv3, bias=False
        )

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class MSDC(nn.Module):
    """Multi-scale depth-wise convolution (MSDC) built on PDw/GConv."""

    def __init__(self, in_channels, kernel_sizes, stride, activation="relu6", dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    Partial_conv3(self.in_channels, 4, kernel_size),
                    nn.BatchNorm2d(self.in_channels),
                    act_layer(self.activation, inplace=True),
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.init_weights("normal")

    def init_weights(self, scheme=""):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel is False:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    """Multi-scale convolution block (MSCB), the core of PMSB.

    Pointwise expansion -> multi-scale PDw/GConv -> channel shuffle ->
    pointwise fusion, with optional skip connection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=False,
        stride=1,
        kernel_sizes=(1, 3, 5),
        expansion_factor=2,
        dw_parallel=True,
        activation="relu6",
    ):
        super().__init__()
        add = shortcut
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True),
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add is True:
            self.combined_channels = self.ex_channels
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights("normal")

    def init_weights(self, scheme=""):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add is True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        return out


class C3k2_PMSB(C2f):
    """C3k2 with Partial Multi-Scale Bottleneck (C3k2-PMSB).

    When c3k is True the internal blocks are standard C3k; otherwise the
    PMSB (MSCB) bottleneck is used.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else MSCB(self.c, self.c, shortcut) for _ in range(n)
        )
