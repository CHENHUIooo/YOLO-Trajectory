"""DMSAB: Dilated Multi-branch Shared Attention Block.

Replaces the traditional downsampling stem in the YOLO11 backbone/neck.
A depthwise-separable downsampling stem is followed by a multi-branch
kernel set (3x3 / 5x5, optional dilation) governed by a shared spatial
attention mechanism.
"""

import torch
import torch.nn as nn

__all__ = ["DMSAB"]


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, dilation=dilation,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SpatialAttention(nn.Module):
    """Shared spatial attention (weights shared across branches)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DMSAB(nn.Module):
    """Dilated Multi-branch Shared Attention Block (DMSAB).

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        att (bool): Whether to apply shared spatial attention on each branch.
        dil (int): Dilation rate for the multi-branch kernels (1 or 2).
    """

    def __init__(self, c1, c2, att, dil):
        super().__init__()
        self.att = att
        self.dil = dil
        self.spatial_attention = SpatialAttention() if att else None
        self.groups = c2 // 2

        # Downsampling stem
        self.conv1 = DepthwiseSeparableConv(c1, c2 // 2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c2 // 2)
        self.relu1 = nn.ReLU()

        if self.dil == 1:
            self.conv2 = DepthwiseSeparableConv(c2 // 2, c2, kernel_size=3, padding=1, dilation=1)
            self.bn2 = nn.BatchNorm2d(c2)
            self.relu2 = nn.ReLU()

            self.conv3 = DepthwiseSeparableConv(c2 // 2, c2, kernel_size=5, padding=2, dilation=1)
            self.bn3 = nn.BatchNorm2d(c2)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv2d(c2 // 2, c2, kernel_size=1, groups=c2 // 2, dilation=1)
            self.bn4 = nn.BatchNorm2d(c2)
            self.relu4 = nn.ReLU(inplace=True)

        if self.dil == 2:
            self.conv2 = DepthwiseSeparableConv(c2 // 2, c2, kernel_size=3, padding=2, dilation=2)
            self.bn2 = nn.BatchNorm2d(c2)
            self.relu2 = nn.ReLU()

            self.conv3 = DepthwiseSeparableConv(c2 // 2, c2, kernel_size=5, padding=4, dilation=2)
            self.bn3 = nn.BatchNorm2d(c2)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv2d(c2 // 2, c2, kernel_size=3, padding=1, groups=c2 // 2, dilation=1)
            self.bn4 = nn.BatchNorm2d(c2)
            self.relu4 = nn.ReLU(inplace=True)

    def channel_shuffle(self, x):
        batch_size, channels, height, width = x.data.size()
        assert channels % self.groups == 0
        group_channels = channels // self.groups
        out = x.reshape(batch_size, group_channels, self.groups, height, width)
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape(batch_size, channels, height, width)
        return out

    def forward(self, x):
        out1 = self.relu1(self.bn1(self.conv1(x)))

        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out1)))

        if self.att:
            out2 = out2 * self.spatial_attention(out2)
            out3 = out3 * self.spatial_attention(out3)

        out4 = self.relu4(self.bn4(self.channel_shuffle(self.conv4(out1))))

        if self.att:
            out4 = out4 * self.spatial_attention(out4)

        return out2 + out3 + out4
