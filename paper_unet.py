"""
Implementation of U-Net architecture from paper: 'U-Net: Convolutional Networks for Biomedical Image Segmentation'.
Ro be modified for diffusion network
"""

from typing import List
import torch
from torch import nn
from torch.nn import functional as funcnn


class DownBlock(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()

        # template of contraction block
        self.downsample_layer = nn.Sequential(
            nn.Conv2d(channels[0], channels[1],
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1],
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        x_contracted = self.downsample_layer(x_img)

        x = self.maxpool(x_contracted)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Conv2d(channels[0], channels[1],
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1],
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        x_img = funcnn.interpolate(x_img, scale_factor=2, mode="nearest")
        x = self.upsample_layer(x_img)

        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # downsampling blovks for contraction
        self.down1 = DownBlock([3, 64])
        self.down2 = DownBlock([64, 128])
        self.down3 = DownBlock([128, 256])
        self.down4 = DownBlock([256, 512])
        self.down5 = DownBlock([512, 1024])

        # 'expansion'. upsampling blocks
        self.upstep1 = UpsampleBlock([1024, 512])
        self.upstep2 = UpsampleBlock([512, 256])
        self.upstep3 = UpsampleBlock([256, 128])
        self.upstep4 = UpsampleBlock([128, 64])
        # self.upstep5 = UpsampleBlock([64, 3])

        self.final_conv = nn.Conv2d(
            64, 3, kernel_size=1, stride=1
        )  # final 1x1 convolution

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(x_img)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        # upsampling stages with concatenation
        up1 = down4 + self.upstep1(down5)
        up2 = down3 + self.upstep2(up1)
        up3 = down2 + self.upstep3(up2)
        up4 = down1 + self.upstep4(up3)

        x = self.final_conv(up4)

        return x
