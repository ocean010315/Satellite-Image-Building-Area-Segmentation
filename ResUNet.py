import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transform
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class ResUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResUNet, self).__init__()
        self.down1 = ConvBlock(in_channels, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)

        self.center = ConvBlock(512, 1024)

        self.up4 = ConvBlock(1024 + 512, 512)
        self.up3 = ConvBlock(512 + 256, 256)
        self.up2 = ConvBlock(256 + 128, 128)
        self.up1 = ConvBlock(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.down1(x)
        x = F.max_pool2d(conv1, kernel_size=2, stride=2)

        conv2 = self.down2(x)
        x = F.max_pool2d(conv2, kernel_size=2, stride=2)

        conv3 = self.down3(x)
        x = F.max_pool2d(conv3, kernel_size=2, stride=2)

        conv4 = self.down4(x)
        x = F.max_pool2d(conv4, kernel_size=2, stride=2)

        x = self.center(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv4], dim=1)
        x = self.up4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv3], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv2], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv1], dim=1)
        x = self.up1(x)

        x = self.final_conv(x)
        x = F.softmax(x, dim=1)

        return x