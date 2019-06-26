import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, final_relu=True):
        super(ConvBlock, self).__init__()

        self.final_relu = final_relu

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.final_relu else x

        return F.relu(x) if self.final_relu else x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, final_relu=True):
        super(UpBlock, self).__init__()

        self.final_relu = final_relu

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.convTranspose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.convTranspose(x)
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

        x = self.pad2(x)
        x = self.norm2(self.conv2(x))

        return x + identity


class TransformerNetConvTranspose(nn.Module):
    def __init__(self):
        super(TransformerNetConvTranspose, self).__init__()

        self.down1 = ConvBlock(3, 16, kernel_size=9, stride=1)
        self.down2 = ConvBlock(16, 32, kernel_size=4, stride=2)
        self.down3 = ConvBlock(32, 64, kernel_size=4, stride=2)

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)

        self.up1 = UpBlock(64, 32, kernel_size=4, stride=2)
        self.up2 = UpBlock(32, 16, kernel_size=4, stride=2)
        self.up3 = UpBlock(16, 3, kernel_size=9, stride=1)

        self.pad = nn.ReflectionPad2d(padding=1)
        self.conv4 = ConvBlock(3, 3, kernel_size=1, stride=1)

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

        x = self.pad(x)
        x = self.conv4(x)

        return self.sigmoid(x)


class TransformerNetConvTransposeLarge(nn.Module):
    def __init__(self):
        super(TransformerNetConvTransposeLarge, self).__init__()

        self.down1 = ConvBlock(3, 32, kernel_size=9, stride=1)
        self.down2 = ConvBlock(32, 64, kernel_size=4, stride=2)
        self.down3 = ConvBlock(64, 128, kernel_size=4, stride=2)

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        self.up1 = UpBlock(128, 64, kernel_size=4, stride=2)
        self.up2 = UpBlock(64, 32, kernel_size=4, stride=2)
        self.up3 = UpBlock(32, 3, kernel_size=9, stride=1)

        self.pad = nn.ReflectionPad2d(padding=1)
        self.conv4 = ConvBlock(3, 3, kernel_size=1, stride=1)

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

        x = self.pad(x)
        x = self.conv4(x)

        return self.sigmoid(x)
