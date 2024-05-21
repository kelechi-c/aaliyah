import torch.nn as nn
import torch.nn.functional as func
import torch

from util_functions import generate_sine_wave, generate_spectrograms

class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super(ConvNet, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        self.bn1 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        x = self.convnet(x)
        x = self.bn1(x)
        return x
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, epsilon: float = 1e-06):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.epsilon = epsilon