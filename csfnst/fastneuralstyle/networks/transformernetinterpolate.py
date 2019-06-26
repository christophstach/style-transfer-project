import torch.nn as nn
import torch.nn.functional as F


class ConvBLock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, final_relu=True):
        super(ConvBLock, self).__init__()

        self.final_relu = final_relu

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.final_relu else x

        return F.relu(x) if self.final_relu else x


class ResidualBlock(nn.Module):
    def __init__(self, channels, inner_channels=None, kernel_size=3):
        super(ResidualBlock, self).__init__()

        inner_channels = inner_channels if inner_channels else channels

        self.conv1 = nn.Conv2d(channels, inner_channels, kernel_size)
        self.conv2 = nn.Conv2d(inner_channels, channels, kernel_size)

        self.pad1 = nn.ReflectionPad2d(padding=kernel_size // 2)
        self.pad2 = nn.ReflectionPad2d(padding=kernel_size // 2)

        self.norm1 = nn.InstanceNorm2d(inner_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        identity = x

        x = self.pad1(x)
        x = F.relu(self.norm1(self.conv1(x)))

        x = self.pad1(x)
        x = self.norm2(self.conv2(x))

        return x + identity


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scale_factor=None, final_relu=True):
        super(UpBlock, self).__init__()

        self.scale_factor = scale_factor
        self.final_relu = final_relu

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        if self.scale_factor:
            x = F.interpolate(x, mode='nearest', scale_factor=self.scale_factor)

        x = self.conv(x)
        x = self.norm(x) if self.final_relu else x

        return F.relu(x) if self.final_relu else x


class TransformerNetInterpolate(nn.Module):
    def __init__(self):
        super(TransformerNetInterpolate, self).__init__()

        self.down1 = ConvBLock(3, 16, kernel_size=9, stride=1)
        self.down2 = ConvBLock(16, 32, kernel_size=4, stride=2)
        self.down3 = ConvBLock(32, 64, kernel_size=4, stride=2)

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)

        self.up1 = UpBlock(64, 32, kernel_size=4, scale_factor=2)
        self.up2 = UpBlock(32, 16, kernel_size=4, scale_factor=2)
        self.up3 = UpBlock(16, 3, kernel_size=9, final_relu=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return self.sigmoid(x)


class TransformerNetInterpolateLarge(nn.Module):
    def __init__(self):
        super(TransformerNetInterpolateLarge, self).__init__()

        self.down1 = ConvBLock(3, 16, kernel_size=9, stride=1)
        self.down2 = ConvBLock(16, 32, kernel_size=4, stride=2)
        self.down3 = ConvBLock(32, 64, kernel_size=4, stride=2)

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)

        self.up1 = UpBlock(64, 32, kernel_size=4, scale_factor=2)
        self.up2 = UpBlock(32, 16, kernel_size=4, scale_factor=2)
        self.up3 = UpBlock(16, 3, kernel_size=9, final_relu=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return self.sigmoid(x)


class CustomTransformerNetInterpolate(nn.Module):
    def __init__(self, channel_multiplier=32):
        super(CustomTransformerNetInterpolate, self).__init__()

        self.down1 = ConvBLock(3, channel_multiplier, kernel_size=9, stride=1)
        self.down2 = ConvBLock(channel_multiplier, channel_multiplier * 2, kernel_size=5, stride=2)
        self.down3 = ConvBLock(channel_multiplier * 2, channel_multiplier * 4, kernel_size=5, stride=2)

        self.res1 = ResidualBlock(channel_multiplier * 4)
        self.res2 = ResidualBlock(channel_multiplier * 4)
        self.res3 = ResidualBlock(channel_multiplier * 4)
        self.res4 = ResidualBlock(channel_multiplier * 4)
        self.res5 = ResidualBlock(channel_multiplier * 4)

        self.pad1 = nn.ReflectionPad2d(padding=2)
        self.up1 = UpBlock(channel_multiplier * 4, channel_multiplier * 2, kernel_size=5, scale_factor=2)

        self.pad2 = nn.ReflectionPad2d(padding=2)
        self.up2 = UpBlock(channel_multiplier * 2, channel_multiplier, kernel_size=5, scale_factor=2)

        self.pad3 = nn.ReflectionPad2d(padding=4)
        self.up3 = UpBlock(channel_multiplier, 3, kernel_size=9, final_relu=False)

        self.pad4 = nn.ReflectionPad2d(padding=4)
        self.conv = ConvBLock(3, 3, kernel_size=1, stride=1, final_relu=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.pad1(x)
        x = self.up1(x)

        x = self.pad2(x)
        x = self.up2(x)

        x = self.pad3(x)
        x = self.up3(x)

        x = self.pad4(x)
        x = self.conv(x)

        return self.sigmoid(x)
