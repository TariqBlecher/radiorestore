import numpy as np
import torch
import os
from pathlib import Path
from astropy.io import fits
from torch import nn
import glob
import torch.nn.functional as F
import logging
import sys

logger = logging.getLogger(__name__)


def get_psf_kernel(Npix, psf_file='meerkat_psfs/meerkat_0_small-psf.fits'):
    psf = fits.getdata(psf_file)
    psf = get_central_pix(psf, Npix)
    return psf


def get_central_pix(data, Npix):
    npix_original = data.shape[-1]
    if npix_original > Npix:
        if len(data.shape)==2:
            data = data[(npix_original - Npix)//2:(npix_original + Npix)//2, (npix_original - Npix)//2:(npix_original + Npix)//2]
        elif len(data.shape)==3:
            data = data[:, (npix_original - Npix)//2:(npix_original + Npix)//2, (npix_original - Npix)//2:(npix_original + Npix)//2]
        elif len(data.shape)==4:
            data = data[:, :, (npix_original - Npix)//2:(npix_original + Npix)//2, (npix_original - Npix)//2:(npix_original + Npix)//2]
    return data


def backproject(sky_model, psf, device='cuda'):
    original_device = sky_model.device.type
    sky_model, psf = sky_model.to(device), psf.to(device)
    
    original_npix = sky_model.shape[-1]
    pad_needed = (psf.shape[-1] - sky_model.shape[-1]) // 2
    sky_model = F.pad(sky_model, (pad_needed, pad_needed, pad_needed, pad_needed))
    sky_model_fft = torch.fft.fft2(sky_model)
    psf_fft = torch.fft.fft2(psf)
    dirty_vis = sky_model_fft * psf_fft
    backprojected = torch.fft.ifft2(dirty_vis)
    backprojected = torch.fft.fftshift(backprojected)
    backprojected = get_central_pix(backprojected, original_npix)
    backprojected = torch.real(backprojected)
    return backprojected.to(original_device)


def create_tiles(img, window_size, margin, device='cuda'):
    """
    Split an array or tensor into square tiles with overlapping margins
    returns tensor of shape (N_tiles, 1,nchan, Npix, Npix)
    """
    sh = list(img.shape)
    sh[-1], sh[-2] = sh[-1] + margin * 2, sh[-2] + margin * 2
    img_ = torch.zeros(sh, device=device)
    if margin:
        img_[:, :, margin:-margin, margin:-margin] = img
    else:
        img_[:, :, :, :] = img

    stride = window_size
    step = window_size + 2 * margin

    nrows, ncols = img.shape[-2] // window_size, img.shape[-1] // window_size
    splitted = []
    start_inds = []
    for i in range(nrows):
        for j in range(ncols):
            h_start = j * stride
            v_start = i * stride
            cropped = img_[:, :, v_start:v_start + step, h_start:h_start + step]
            cropped = cropped[None, :, :, :, :]
            splitted.append(cropped)
            start_inds.append([h_start, v_start])
    splitted = torch.cat(splitted)
    return splitted, start_inds


def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_read=None, checkpoint_read_model_only=False, device='cuda'):
    if os.path.exists(checkpoint_read):
        checkpoint = torch.load(checkpoint_read)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not checkpoint_read_model_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                None
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        logger.info('loading model checkpoint with loss %s', checkpoint['loss'])
    else:
        logger.info('Model checkpoint does not exist')
    return model, optimizer, scheduler


def copy_scripts(output_dir):
    import deconvutils
    python_file_dir = Path(output_dir, 'py_files')
    if not os.path.exists(python_file_dir):
        os.mkdir(python_file_dir)

    os.system(f'cp {deconvutils.__path__[0]}/*.py {python_file_dir}')
    os.system(f'cp {sys.argv[0]} {python_file_dir}')


def wsclean_predict(image_name_stem, Npix, ms):
    """Converts a model image to the Model column of a Measurement Set"""

    wsclean_predict_model_call = f'wsclean -name {image_name_stem} -predict -weight briggs -2.0 -padding 1.5 -size {Npix} {Npix} -channels-out 1 -scale 1.6asec {ms}'
    return wsclean_predict_model_call


def wsclean_image_call(name, Npix, ms, subtract_model_col=False, make_psf=False, niter=1):
    """Creates Dirty Image, Residuals and PSF"""

    wsclean_image_call = f'wsclean -name {name} -data-column DATA -no-update-model-required -channels-out 1 -weight briggs -2.0 -niter {niter} -padding 1.2 -size {Npix} {Npix} -scale 1.6asec -log-time '
    if subtract_model_col:
        wsclean_image_call += ' -subtract-model '
    if make_psf:
        wsclean_image_call += ' -make-psf '
    wsclean_image_call += f' {ms}'

    return wsclean_image_call


def set_defaults(dtype_keyword):
    torch.manual_seed(0)
    np.random.seed(0)
    if dtype_keyword == 'float32':
        dtype_np = np.float32
        torch.set_default_dtype(torch.float32)
    else:
        dtype_np = np.float64
        torch.set_default_dtype(torch.float64) 
    return dtype_np
