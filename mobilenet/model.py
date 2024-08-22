import torch
from torch import nn
from huggingface_hub import login, PytorchModelHubMixin

login()


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x = self.depthwise_conv(x)
        x = self.pointwise(x)

        return x


class MobileNet(nn.Module, PytorchModelHubMixin):
    def __init__(self, out_ch=32, num_classes=15, out_size=None):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_ch),
        )

        self.linear_fc = nn.Linear(out_size, num_classes)

    def forward(self, x):
        x = self.input_conv(x)

        x = self.linear_fc(x)

        return x
