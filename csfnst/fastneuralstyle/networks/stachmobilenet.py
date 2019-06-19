import torch.nn as nn
import torch.nn.functional as F


class MobileNetResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, expand_ratio):
        super(MobileNetResidualBlock, self).__init__()

        inner_channels = expand_ratio * in_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            in_channels=inner_channels,
            out_channels=inner_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=inner_channels
        )
        self.conv3 = nn.Conv2d(in_channels=inner_channels, out_channels=in_channels, kernel_size=1, stride=1)

        self.norm1 = nn.InstanceNorm2d(num_features=inner_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(num_features=inner_channels, affine=True)
        self.norm3 = nn.InstanceNorm2d(num_features=in_channels, affine=True)

        self.relu1 = nn.ReLU6()
        self.relu2 = nn.ReLU6()

        self.pad = nn.ReflectionPad2d(padding=kernel_size // 2)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.pad(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        return x + identity


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MobileNetBlock, self).__init__()

        inner_channels = expand_ratio * in_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            in_channels=inner_channels,
            out_channels=inner_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=inner_channels
        )
        self.conv3 = nn.Conv2d(in_channels=inner_channels, out_channels=out_channels, kernel_size=1, stride=1)

        self.norm1 = nn.InstanceNorm2d(num_features=inner_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(num_features=inner_channels, affine=True)
        self.norm3 = nn.InstanceNorm2d(num_features=out_channels, affine=True)

        self.relu1 = nn.ReLU6()
        self.relu2 = nn.ReLU6()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        return x


class UpResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, expand_ratio, scale_factor):
        super(UpResidualBlock, self).__init__()

        self.scale_factor = scale_factor

        self.mobile = MobileNetResidualBlock(
            in_channels=in_channels,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio
        )

    def forward(self, x):
        x = F.interpolate(x, mode='nearest', scale_factor=self.scale_factor)
        x = self.mobile(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, scale_factor):
        super(UpBlock, self).__init__()

        self.scale_factor = scale_factor

        self.mobile = MobileNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio
        )

    def forward(self, x):
        x = F.interpolate(x, mode='nearest', scale_factor=self.scale_factor)
        x = self.mobile(x)

        return x


class LastBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, expand_ratio):
        super(LastBlock, self).__init__()

        self.mobile = MobileNetBlock(
            in_channels=in_channels,
            out_channels=3,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mobile(x)
        x = self.sigmoid(x)

        return x


class LastResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, expand_ratio):
        super(LastResidualBlock, self).__init__()

        self.mobile = MobileNetResidualBlock(
            in_channels=in_channels,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mobile(x)
        x = self.sigmoid(x)

        return x


class StachMobileNet(nn.Module):
    def __init__(self):
        super(StachMobileNet, self).__init__()
        er = 1

        self.down1 = MobileNetBlock(
            in_channels=3,
            out_channels=128,
            kernel_size=9,
            stride=2,
            expand_ratio=er
        )

        self.bottleneck1 = MobileNetResidualBlock(in_channels=128, kernel_size=3, expand_ratio=er)
        self.bottleneck2 = MobileNetResidualBlock(in_channels=128, kernel_size=3, expand_ratio=er)
        self.bottleneck3 = MobileNetResidualBlock(in_channels=128, kernel_size=3, expand_ratio=er)

        self.up1 = UpBlock(
            in_channels=128,
            out_channels=3,
            kernel_size=9,
            stride=1,
            expand_ratio=er,
            scale_factor=2
        )

        
        self.pad1 = nn.ReflectionPad2d(padding=8)

        self.last = LastResidualBlock(in_channels=3, kernel_size=3, expand_ratio=er)

    def forward(self, x):
        x = self.down1(x)

        x = self.bottleneck1(x)
        # x = self.bottleneck2(x)
        # x = self.bottleneck3(x)

        x = self.up1(x)
        x = self.pad1(x)

        x = self.last(x)

        return x


class StachMobileFullResidualNet(nn.Module):
    def __init__(self):
        super(StachMobileFullResidualNet, self).__init__()

        er = 2
        channels = 3

        self.mobile1 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)
        self.mobile2 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)
        self.mobile3 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)

        self.mobile4 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)
        self.mobile5 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)
        self.mobile6 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)

        self.mobile7 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)
        self.mobile8 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)
        self.mobile9 = MobileNetResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)

        self.last = LastResidualBlock(in_channels=channels, kernel_size=3, expand_ratio=er)

    def forward(self, x):
        x = self.mobile1(x)
        x = self.mobile2(x)
        x = self.mobile3(x)

        x = self.mobile4(x)
        x = self.mobile5(x)
        x = self.mobile6(x)

        x = self.mobile7(x)
        x = self.mobile8(x)
        x = self.mobile9(x)

        x = self.last(x)

        return x
