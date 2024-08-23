"""

Implementation of the MobileNet convolutional architecture from the paper:
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
by Google.

"""

import torch
from torch import nn
from torch.nn import functional as func_nn
from huggingface_hub import login, PytorchModelHubMixin
from .utils_config import config


login()


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x = self.depthwise_conv(x)
        x = self.pointwise(x)

        return x


# mobilenet class, with depthwise(cross-channel) and pointwise(1x1) convolutions.
# 90% of the parameters are from the pointwise convs.


class MobileNet(nn.Module, PytorchModelHubMixin):
    def __init__(
        self,
        out_ch=32,
        num_classes=15,
        out_size=1024,
        channels=[32, 64, 128, 256, 512, 1024],
    ):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.mobilenet_layers = nn.Sequential(
            MobileBlock(32, 64),
            MobileBlock(64, 128, stride=2),
            MobileBlock(128, 256),
            MobileBlock(256, 512, stride=2),
            # 5x (512, 512) block
            MobileBlock(512, 512),
            MobileBlock(512, 512),
            MobileBlock(512, 512),
            MobileBlock(512, 512),
            MobileBlock(512, 512),
            MobileBlock(512, 1024, stride=2),
            MobileBlock(1024, 1024, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear_fc = nn.Linear(out_size, num_classes)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.mobilenet_layers(x)

        x = x.view(-1, 1024)
        x = self.linear_fc(x)

        x = func_nn.softmax(x, dim=1)

        return x


mobilenet = MobileNet().to(config.dtype).to(config.device)

mobilenet.save_pretrained(config.har_model_id)
# push to the hub
mobilenet.push_to_hub(config.har_model_id)
