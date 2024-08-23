import torch
from torch import nn
import cv2
import numpy as np

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


def read_image(img, img_size):
    img = np.array(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img
