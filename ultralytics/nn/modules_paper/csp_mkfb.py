"""CSP_MKFB: Cross Stage Partial Bottleneck with Multi-Kernel Fusion Block (CSP-MKFB).

Backbone enhancement module. A CSP split feeds one branch through the
Multi-Kernel Fusion Block (MKFB): stacked multi-kernel units (5/7/9) whose
fundamental unit is either a Star Universal Inverted Bottleneck (SUIB, deep
stages) or a standard 3x3 convolution (shallow P2 stage), giving a large
effective receptive field at low computational cost.
"""

import torch
import torch.nn as nn

__all__ = ["CSP_MKFB"]


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self._autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    @staticmethod
    def _autopad(k, p=None, d=1):
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SUIB(nn.Module):
    """Star Universal Inverted Bottleneck (SUIB) block.

    UIB-style inverted bottleneck (depthwise conv -> dual 1x1 star gating ->
    expand 1x1 -> depthwise project conv) with an identity shortcut.
    The star operation fuses features via element-wise multiplication of two
    1x1 projections, expanding the feature space non-linearly.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, d=1, star=True):
        super().__init__()
        self.star = star

        self.conv1 = Conv(
            c1=in_channels, c2=in_channels, k=kernel_size, d=d,
            p=d * (kernel_size - 1) // 2, g=in_channels, act=False,
        )
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv2 = Conv(c1=out_channels, c2=in_channels, k=1, act=False)
        self.conv3 = Conv(
            c1=out_channels, c2=out_channels, k=kernel_size, d=d,
            p=d * (kernel_size - 1) // 2, g=in_channels, act=True,
        )
        self.act = nn.ReLU6()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        if self.star:
            x1, x2 = self.conv11(x), self.conv12(x)
            x = self.act(x1) * x2
        else:
            x = self.conv11(x)
        x = self.conv2(self.conv3(x))
        return x + identity


class MKFB(nn.Module):
    """Multi-Kernel Fusion Block (MKFB).

    Stacks multi-kernel fundamental units (SUIB or 3x3 Conv, selected by
    light_block) with kernel sizes cycling through 5/7/9, and aggregates
    multi-scale features via squeeze/excitation 1x1 convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, d=1, layer_num=6, identity=False, light_block=True):
        super().__init__()
        self.identity = identity

        total_channels = in_channels + layer_num * in_channels
        if light_block:
            kernel_cycle = [5, 7, 9, 5, 7, 9]
            self.layers = nn.ModuleList(
                SUIB(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    stride=1,
                    kernel_size=kernel_cycle[i % len(kernel_cycle)],
                    star=True,
                    d=1,
                )
                for i in range(layer_num)
            )
        else:
            self.layers = nn.ModuleList(
                Conv(c1=in_channels, c2=in_channels, s=1, k=kernel_size, d=d) for _ in range(layer_num)
            )
        self.aggregation_squeeze_conv = Conv(c1=total_channels, c2=out_channels // 2, k=1, s=1)
        self.aggregation_excitation_conv = Conv(c1=out_channels // 2, c2=out_channels, k=1, s=1)

    def forward(self, x):
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x = x + identity
        return x


class CSP_MKFB(nn.Module):
    """Cross Stage Partial Bottleneck with Multi-Kernel Fusion Block (CSP-MKFB).

    CSP-style split: one half passes through MKFB (multi-kernel fusion of SUIB
    or Conv units), the other half is skipped; features are concatenated and
    fused by a final 1x1 conv.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int): Number of stacked fundamental units (layer_num) in MKFB.
        c3k (bool): If False, use 3x3 Conv units (shallow P2); if True, use SUIB units with d=2.
        e (float): Channel expansion ratio for the internal width.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.c = int(c2 * e)
        self.c3k = c3k

        self.cbs = Conv(c1, self.c, k=1, s=1)

        if c3k is False:
            self.block = MKFB(
                self.c // 2, self.c // 2, kernel_size=3, layer_num=n, d=1, identity=True, light_block=False
            )
        else:
            self.block = MKFB(
                self.c // 2, self.c // 2, kernel_size=3, layer_num=n, d=2, identity=True, light_block=True
            )

        self.final_cbs = Conv(self.c + self.c // 2, c2, k=1, s=1, act=True)

    def forward(self, x):
        x = self.cbs(x)
        out1, out2 = torch.chunk(x, 2, dim=1)
        x = self.block(out1)
        x = torch.cat([out1, out2, x], dim=1)
        return self.final_cbs(x)
