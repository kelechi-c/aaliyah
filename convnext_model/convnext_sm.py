from torch import nn
from einops import rearrange
from timm.layers import trunc_normal_


# convnext block
class ConvNextBlock(nn.Module):
    def __init__(self, dim):
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),  # pointwise convolution with MLp
            nn.GELU(),  # replace ReLU
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        input = x
        x = self.conv_block(x)  # b, h, w, c

        x = rearrange(x, "b h w c -> b c h w")

        return input + x


class ConvNext(nn.Module):
    def __init__(
        self, in_ch=3, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], classes=44
    ):
        self.layers = nn.ModuleList()  # init downsampling layers list
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 96, kernel_size=4, padding=4), nn.LayerNorm(96)
        )

        self.layers.append(self.stem)
        for x in range(3):
            down_layers = nn.Sequential(
                nn.LayerNorm(dims[x]),
                nn.Conv2d(dims[x], dims[x + 1], kernel_size=2, padding=2),
            )
            self.layers.append(down_layers)

        self.conv_stages = nn.ModuleList()  # for convolutions
        for x in range(3):
            stage = nn.Sequential(  # convnext blocks/heads of different depths
                *[ConvNextBlock(dim=dims[x]) for _ in range(depths[x])]
            )

            self.conv_stages.append(stage)

        self.mlp_head = nn.Linear(dims[-1], classes)
        self.layer_norm = nn.LayerNorm(dims[-1])

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            trunc_normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0)

    def forward_convolution(self, x):
        for k in range(4):
            x = self.layers[k](x)
            x = self.conv_stages[k](x)
            x = self.layer_norm(x.mean([-2, -1]))

        return x

    def forward(self, x):
        x = self.layers(x)
        x = self.mlp_head(x)

        return x
