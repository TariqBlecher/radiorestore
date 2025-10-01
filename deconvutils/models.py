import numpy as np
import torch
import torch.nn as nn
from .utils import *
from .trainer import *
from .datasets import *
from .unet import UNet


class nn_convolve_straight(nn.Module):
    def __init__(self, input_model=UNet(), dist_shift=True):
        super().__init__()
        self.Umodel = input_model
        self.dist_shift = dist_shift

    def UNet_dist_shift(self, x):
        x_std = x.std((1, 2, 3), keepdim=True)
        x = x / x_std

        x = self.Umodel(x)
        return x

    def generate_model_image(self, x):
        x_max = x.amax((1, 2, 3), keepdim=True)
        x = x / x_max

        if self.dist_shift:
            x = self.UNet_dist_shift(x)
        else:
            x = self.Umodel(x)

        x = nn.functional.relu(x)
        x = x * x_max
        return x

    def forward(self, x):
        x = self.generate_model_image(x)
        if torch.isnan(x).sum() > 0:
            print('NAN output by model')
        return x
    