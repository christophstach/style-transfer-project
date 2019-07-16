from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckType(Enum):
    RESIDUAL_BLOCK = 1
    MOBILE_VERSION_ONE_BLOCK = 2
    MOBILE_VERSION_TWO_BLOCK = 3


def get_activation_fn(name, channels):
    activation_fn_map = {
        'ELU': lambda: nn.ELU(),
        'ReLU': lambda: nn.ReLU(),
        'RReLU': lambda: nn.RReLU(),
        'PReLU': lambda: nn.PReLU(channels),
        'SELU': lambda: nn.SELU(),
        'CELU': lambda: nn.CELU(),
        'ReLU6': lambda: nn.ReLU6(),
        'Hardtanh': lambda: nn.Hardtanh(min_val=0.0, max_val=1.0),
        'Sigmoid': lambda: nn.Sigmoid(),
        'None': lambda: None
    }

    return activation_fn_map[name]()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation_fn='None'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation_fn = get_activation_fn(activation_fn, out_channels)

    def forward(self, x):
        x = self.norm(self.conv(x))
        x = self.activation_fn(x) if self.activation_fn else x

        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scale_factor=None,
                 activation_fn='None'):
        super(UpSampleBlock, self).__init__()

        self.scale_factor = scale_factor

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation_fn=activation_fn
        )

    def forward(self, x):
        if self.scale_factor:
            x = F.interpolate(x, mode='nearest', scale_factor=self.scale_factor)

        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=None, kernel_size=3, activation_fn='None'):
        super(ResidualBlock, self).__init__()

        inner_channels = inner_channels if inner_channels else in_channels

        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size)
        self.conv2 = nn.Conv2d(inner_channels, out_channels, kernel_size)

        self.pad1 = nn.ReflectionPad2d(padding=kernel_size // 2)
        self.pad2 = nn.ReflectionPad2d(padding=kernel_size // 2)

        self.norm1 = nn.InstanceNorm2d(inner_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        self.activation_fn = get_activation_fn(activation_fn, inner_channels)

    def forward(self, x):
        identity = x

        x = self.norm1(self.conv1(self.pad1(x)))
        x = self.activation_fn(x) if self.activation_fn else x
        x = self.norm2(self.conv2(self.pad2(x)))

        return x + identity


class MobileVersionTwoBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6,
                 use_skip_connection=True, activation_fn1='None', activation_fn2='None'):
        super(MobileVersionTwoBlock, self).__init__()

        inner_channels = in_channels * expansion_factor
        self.use_skip_connection = use_skip_connection

        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=kernel_size, stride=stride,
                               groups=inner_channels)
        self.conv3 = nn.Conv2d(inner_channels, out_channels, kernel_size=1)

        self.pad = nn.ReflectionPad2d(padding=kernel_size // 2)

        self.norm1 = nn.InstanceNorm2d(inner_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(inner_channels, affine=True)
        self.norm3 = nn.InstanceNorm2d(out_channels, affine=True)

        self.activation_fn1 = get_activation_fn(activation_fn1)
        self.activation_fn2 = get_activation_fn(activation_fn2)

    def forward(self, x):
        identity = x
        x = self.pad(x)

        x = self.norm1(self.conv1(x))
        x = self.activation_fn1(x) if self.activation_fn1 else x
        x = self.norm2(self.conv2(x))
        x = self.activation_fn2() if self.activation_fn2 else x
        x = self.norm3(self.conv3(x))

        if self.use_skip_connection:
            return x + identity
        else:
            return x


class ZeroPadding(torch.nn.Module):
    def __init__(self, padding, channels, x_h, x_w):
        super(ZeroPadding, self).__init__()

        self.padding = padding
        self.x_h = x_h
        self.x_w = x_w

        self.channels = channels
        self.pad_width = torch.zeros([1, self.channels, x_h, padding], dtype=torch.float)
        self.pad_height = torch.zeros([1, self.channels, padding, x_w + (padding * 2)], dtype=torch.float)

    def forward(self, x):
        n = torch.cat([self.pad_width, x, self.pad_width], 3)
        n = torch.cat([self.pad_height, n, self.pad_height], 2)

        return n


class ConcatUpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scale_factor=None,
                 activation_fn='None'):
        super(ConcatUpSampleBlock, self).__init__()

        self.scale_factor = scale_factor

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation_fn=activation_fn
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(-1, 1)
        x = torch.cat([x, x], 1)
        x = x.view(-1, w * 2)
        x = torch.cat([x, x], 1)
        x = x.view(-1, c, h * 2, w * 2)

        return self.conv(x)


class MobileVersionOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_skip_connection=True,
                 activation_fn1='None', activation_fn2='None'):
        super(MobileVersionOneBlock, self).__init__()

        self.use_skip_connection = use_skip_connection

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.pad = nn.ReflectionPad2d(padding=kernel_size // 2)

        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        self.activation_fn1 = get_activation_fn(activation_fn1)
        self.activation_fn2 = get_activation_fn(activation_fn2)

    def forward(self, x):
        identity = x
        x = self.pad(x)

        x = self.norm1(self.conv1(x))
        x = self.activation_fn1(x) if self.activation_fn1 else x

        x = self.norm2(self.conv2(x))
        x = self.activation_fn2(x) if self.activation_fn2 else x

        if self.use_skip_connection:
            return x + identity
        else:
            return x


class TransformerNet(nn.Module):
    def __init__(self, channel_multiplier=32, bottleneck_size=5, bottleneck_type=BottleneckType.RESIDUAL_BLOCK,
                 expansion_factor=6, final_activation_fn='None', intermediate_activation_fn='None'):
        super(TransformerNet, self).__init__()

        self.pad = nn.ReflectionPad2d(padding=20)

        self.down1 = ConvBlock(3, channel_multiplier, kernel_size=9, stride=1, activation_fn=intermediate_activation_fn)
        self.down2 = ConvBlock(channel_multiplier, channel_multiplier * 2, kernel_size=5, stride=2,
                               activation_fn=intermediate_activation_fn)
        self.down3 = ConvBlock(channel_multiplier * 2, channel_multiplier * 4, kernel_size=5, stride=2,
                               activation_fn=intermediate_activation_fn)

        if bottleneck_type == BottleneckType.RESIDUAL_BLOCK:
            self.bottleneck = nn.Sequential(*[
                ResidualBlock(channel_multiplier * 4, channel_multiplier * 4, activation_fn=intermediate_activation_fn)
                for _ in range(bottleneck_size)
            ])
        else:
            raise ValueError('Wrong value for bottleneck_type')

        self.up1 = UpSampleBlock(channel_multiplier * 4, channel_multiplier * 2, kernel_size=5, scale_factor=2,
                                 activation_fn=intermediate_activation_fn)
        self.up2 = UpSampleBlock(channel_multiplier * 2, channel_multiplier, kernel_size=5, scale_factor=2,
                                 activation_fn=intermediate_activation_fn)
        self.up3 = ConvBlock(channel_multiplier, 3, kernel_size=9, activation_fn=final_activation_fn)

    def forward(self, x):
        x = self.pad(x)
        x = self.down3(self.down2(self.down1(x)))
        x = self.bottleneck(x)
        return self.up3(self.up2(self.up1(x)))
