import numpy as np
import torch
import logging
import scipy
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def make_noise_map(restored_image, boxsize, dtype=np.float64, device='cuda'):
    # Torch adaption of Cyril's magic minimum filter
    n = boxsize**2.0
    k = np.linspace(-10, 10, 1000)
    f = 0.5 * (1.0 + scipy.special.erf(k / np.sqrt(2.0)))
    Fparam = 1.0 - (1.0 - f)**n
    ratio = np.abs(np.interp(0.5, Fparam, k))

    ratio = torch.tensor(ratio.astype(dtype=dtype), device=device)

    noise = torch.nn.functional.max_pool2d(-1 * restored_image, boxsize, stride=4, padding=boxsize//2-1) / ratio
    Upsampler = torch.nn.Upsample(scale_factor=4, mode='nearest')
    noise = Upsampler(noise)
    negative_mask = noise < 0.0

    noise[negative_mask] = 1.0e-10
    median_noise = torch.median(noise)
    median_mask = noise < median_noise
    noise[median_mask] = median_noise
    return noise

def binary_closure(mask_image, dilation_kernel = 1, device='cpu'):
    """Note that large kernels will take exponentially longer to run"""
    if dilation_kernel%2!=1:
        print('Rather use odd dilation kernel, even kernels can result in artefacts where border regions are lost to the mask')
    mask_image = mask_image.double()
    R = dilation_kernel
    r = torch.arange(-R, R+1)
    struct = torch.sqrt(r[:, None]**2 + r[None,:]**2) <= R
    kernel = struct.double()[None, None, :, :]
    kernel = kernel.to(device)
    mask_dilated = F.conv2d(mask_image, kernel, padding='same')
    mask_dilated[mask_dilated>0]=1
    mask_eroded = F.conv2d(mask_dilated, kernel, padding='same')
    mask_eroded = mask_eroded>=kernel.sum()
    return mask_eroded


def create_mask(input_image, boxsize=100, threshold=6.5, dtype=np.float64, invert=False, device='cuda', dilate=True, lower_cut=0.2, initial_mask=None, dilation_kernel=3):

    noise_image = make_noise_map(input_image, boxsize, dtype=dtype, device=device)
    mask_image = (input_image > threshold * noise_image)

    mask_image[0, 0, :, -1]=0
    mask_image[0, 0, :, 0]=0
    mask_image[0, 0, 0, :]=0
    mask_image[0, 0, -1, :]=0

    if lower_cut:
        mask_tmp = input_image > (input_image.max() * lower_cut)
        mask_image = torch.logical_and(mask_tmp, mask_image) 
    
    if initial_mask is not None:
        mask_image = torch.logical_or(mask_image, initial_mask)

    if dilate:
        mask_image = binary_closure(mask_image, dilation_kernel=dilation_kernel, device=device)
    
    if invert:
        mask_image = mask_image==0
    
    return mask_image

