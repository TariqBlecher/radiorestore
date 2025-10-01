import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, batchnorm=True, padding='same'):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, padding=padding)
        if batchnorm:
            self.batch_norm = nn.BatchNorm2d(out_chan)
        else:
            self.batch_norm = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, batchnorm=True, padding='same'):
        super().__init__()
        self.convblock1 = ConvBlock(in_chan, out_chan, kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)
        self.convblock2 = ConvBlock(out_chan, out_chan, kernel_size=kernel_size, batchnorm=batchnorm, padding=padding)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_chan=1, depth=3, starting_chan=64, batchnorm=True, padding='same'):
        super().__init__()
        self.blocks = nn.ModuleList()
        for layer_ind in range(depth):

            out_chan = starting_chan * 2**layer_ind

            self.blocks.append(DoubleConvBlock(in_chan, out_chan, batchnorm=batchnorm, padding=padding))

            in_chan = out_chan

    def forward(self, x):
        features = []
        for ind, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            x = F.max_pool2d(x, 2)
        return features[::-1]


class Decoder(nn.Module):
    def __init__(self, depth=3, chan_factor=64, batchnorm=True, padding='same', upsampling='bilinear'):
        super().__init__()
        self.channels = [chan_factor * 2**(depth - layer_ind - 1) for layer_ind in range(depth)]
        self.upsample_interpolation = nn.ModuleList([nn.Upsample(scale_factor=2, mode=upsampling) for i in range(depth - 1)])
        self.upsample_convblocks = nn.ModuleList([nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=3, padding=padding) for i in range(depth - 1)])

        self.convblocks = nn.ModuleList([DoubleConvBlock(self.channels[i], self.channels[i + 1], batchnorm=batchnorm, padding=padding) for i in range(depth - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.upsample_interpolation[i](x)
            x = self.upsample_convblocks[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.convblocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, in_chan=1, depth=3, chan_factor=64, out_chan=1, batchnorm=True, padding='same', upsampling='bilinear'):
        super().__init__()
        self.encoder = Encoder(in_chan=in_chan, depth=depth, starting_chan=chan_factor, batchnorm=batchnorm, padding=padding)
        self.decoder = Decoder(depth=depth, chan_factor=chan_factor, batchnorm=batchnorm, padding=padding, upsampling=upsampling)
        self.head = nn.Conv2d(self.decoder.channels[-1], out_chan, 1)

    def forward(self, x):
        features = self.encoder(x)
        dec_out = self.decoder(features[0], features[1:])
        out = self.head(dec_out)
        return out
