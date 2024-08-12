import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


# patchify module
class Patchify(nn.Module):
    def __init__(self, patches=16):
        super().__init__()
        self.patch_size = patches
        self.patch_module = nn.Unfold(kernel_size=patches, stride=patches)

    def forward(self, x: torch.Tensor):
        batch_s, ch, h, w = x.shape

        x = self.patch_module(x)
        x = x.view(batch_s, ch, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3)

        return x


class DepthwiseConvBlock(nn.Module):
    def __init__(self):
        self.depthconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, groups=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.depthconv(x)

        return x


# convnext block
class ConvNextBlock(nn.Module):
    def __init__(self, dim, depth):
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),  # pointwise convolution with MLp
            nn.GELU(),  # replace ReLU
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        input = x
        x = self.conv_block(x)  # b, h, w, c

        x = rearrange(x, "b h w c -> b c h w")

        return input + x


class ConvNext(nn.Module):
    def __init__(self, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3]):
