import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()

        self.pad = nn.ReflectionPad2d(padding=kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)

        return F.leaky_relu(x)


class JoinBlock(nn.Module):
    def __init__(self, out_channels1, out_channels2):
        super(JoinBlock, self).__init__()

        self.norm1 = nn.InstanceNorm2d(out_channels1, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_channels2, affine=True)

    def forward(self, x1, x2):
        height = x1.shape[2]
        width = x1.shape[3]

        x1 = self.norm1(x1)

        x2 = F.interpolate(x2, mode='nearest', size=(height, width))
        x2 = self.norm2(x2)

        return torch.cat((x1, x2), 1)


class TextureNet(nn.Module):
    def __init__(self):
        super(TextureNet, self).__init__()

        # Z1 Row
        self.block_1_1_1 = ConvBlock(3, 8, 3)
        self.block_1_1_2 = ConvBlock(8, 8, 3)
        self.block_1_1_3 = ConvBlock(8, 8, 1)

        self.join1 = JoinBlock(8, 32)

        self.block_1_2_1 = ConvBlock(40, 40, 3)
        self.block_1_2_2 = ConvBlock(40, 40, 3)
        self.block_1_2_3 = ConvBlock(40, 40, 1)
        self.block_1_2_4 = ConvBlock(40, 3, 1)

        # Z2 Row
        self.block_2_1_1 = ConvBlock(3, 8, 3)
        self.block_2_1_2 = ConvBlock(8, 8, 3)
        self.block_2_1_3 = ConvBlock(8, 8, 1)

        self.join2 = JoinBlock(8, 24)

        self.block_2_2_1 = ConvBlock(32, 32, 3)
        self.block_2_2_2 = ConvBlock(32, 32, 3)
        self.block_2_2_3 = ConvBlock(32, 32, 1)

        # Z3 Row
        self.block_3_1_1 = ConvBlock(3, 8, 3)
        self.block_3_1_2 = ConvBlock(8, 8, 3)
        self.block_3_1_3 = ConvBlock(8, 8, 1)

        self.join3 = JoinBlock(8, 16)

        self.block_3_2_1 = ConvBlock(24, 24, 3)
        self.block_3_2_2 = ConvBlock(24, 24, 3)
        self.block_3_2_3 = ConvBlock(24, 24, 1)

        # Z4 Row
        self.block_4_1_1 = ConvBlock(3, 8, 3)
        self.block_4_1_2 = ConvBlock(8, 8, 3)
        self.block_4_1_3 = ConvBlock(8, 8, 1)

        self.join4 = JoinBlock(8, 8)

        self.block_4_2_1 = ConvBlock(16, 16, 3)
        self.block_4_2_2 = ConvBlock(16, 16, 3)
        self.block_4_2_3 = ConvBlock(16, 16, 1)

        # Z5 Row
        self.block_5_1_1 = ConvBlock(3, 8, 3)
        self.block_5_1_2 = ConvBlock(8, 8, 3)
        self.block_5_1_3 = ConvBlock(8, 8, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x, scale_factor=1 / 2)
        x3 = F.interpolate(x, scale_factor=1 / 4)
        x4 = F.interpolate(x, scale_factor=1 / 8)
        x5 = F.interpolate(x, scale_factor=1 / 16)

        # Z5 Row
        x5 = self.block_5_1_1(x5)
        x5 = self.block_5_1_2(x5)
        x5 = self.block_5_1_3(x5)

        # Z4 Row
        x4 = self.block_4_1_1(x4)
        x4 = self.block_4_1_2(x4)
        x4 = self.block_4_1_3(x4)

        x4 = self.join4(x4, x5)

        x4 = self.block_4_2_1(x4)
        x4 = self.block_4_2_1(x4)
        x4 = self.block_4_2_1(x4)

        # Z3 Row
        x3 = self.block_3_1_1(x3)
        x3 = self.block_3_1_2(x3)
        x3 = self.block_3_1_3(x3)

        x3 = self.join3(x3, x4)

        x3 = self.block_3_2_1(x3)
        x3 = self.block_3_2_2(x3)
        x3 = self.block_3_2_3(x3)

        # Z2 Row
        x2 = self.block_2_1_1(x2)
        x2 = self.block_2_1_2(x2)
        x2 = self.block_2_1_3(x2)

        x2 = self.join2(x2, x3)

        x2 = self.block_2_2_1(x2)
        x2 = self.block_2_2_2(x2)
        x2 = self.block_2_2_3(x2)

        # Z1 Row
        x1 = self.block_1_1_1(x1)
        x1 = self.block_1_1_2(x1)
        x1 = self.block_1_1_3(x1)

        x1 = self.join1(x1, x2)

        x1 = self.block_1_2_1(x1)
        x1 = self.block_1_2_2(x1)
        x1 = self.block_1_2_3(x1)
        x1 = self.block_1_2_4(x1)

        return self.sigmoid(x1)
