from torch.utils.data import DataLoader
import torch
import numpy as np
from .utils import backproject


class XYDataset(object):
    def __init__(self, sky_images_noise, sky_images, transform=None):
        self.sky_images_noise = sky_images_noise
        self.sky_images = sky_images
        self.transform = transform

    def __getitem__(self, idx):
        if self.transform:
            dirty_image = self.transform(self.sky_images_noise[idx])
        else:
            dirty_image = self.sky_images_noise[idx]
        return dirty_image, self.sky_images[idx]

    def __len__(self):
        return self.sky_images.shape[0]


def generate_dataloaders(X, Y, batch_size=8, validation_split=0.2, num_workers=4, transform=None):
    nimgs = X.shape[0]
    split_index = int(nimgs * validation_split)

    X_test = X[:split_index]
    X_train = X[split_index:]
    Y_test = Y[:split_index]
    Y_train = Y[split_index:]
    train_dataset = XYDataset(X_train, Y_train, transform=transform)
    test_dataset = XYDataset(X_test, Y_test, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader



class gaussnoise(torch.nn.Module):
    def __init__(self, std_low, std_upper):
        super().__init__()
        self.std_low = np.log10(std_low)
        self.std_high = np.log10(std_upper)

    def __call__(self, x):
        stddev = self.std_low + torch.rand(1)*(self.std_high - self.std_low)
        stddev = 10**stddev
        noise = torch.normal(0., float(stddev), x.shape, device=x.device)        
        return x + noise
    
class PSF_convolve(torch.nn.Module):
    def __init__(self, psf, device='cpu'):
        self.psf = psf[0,:,:,:].to(device)
        self.device=device

    def __call__(self, x):
        x = backproject(x, self.psf, device=self.device)
        return x 

    