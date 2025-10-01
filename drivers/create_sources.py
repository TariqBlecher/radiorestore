import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from astropy.io import fits
from deconvutils.utils import get_central_pix, backproject, set_defaults
from deconvutils.sources import make_points, make_gaussians, make_points_off_center_positions


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    np_dtype  = set_defaults(args.dtype)
    
    TNG_files = list(Path(args.tng_dir).glob('*.npy'))
    test_input = np.load(TNG_files[0])
    Npix = test_input.shape[-1]

    psf = fits.getdata(args.psf_file)
    psf_trim = get_central_pix(psf, 2 * Npix)
    psf_trim = torch.from_numpy(psf_trim.astype(np_dtype)).to(device)

    TNG_files_subsample = np.random.choice(TNG_files, size=args.n_samples, replace=False, p=None)

    true_sky_array = torch.zeros((TNG_files_subsample.shape[0], 1, Npix + 2 * args.margin, Npix + 2 * args.margin)).to(device)
    dirty_sky_array = torch.zeros((TNG_files_subsample.shape[0], 1, Npix + 2 * args.margin, Npix + 2 * args.margin)).to(device)

    for ind, tng_file in enumerate(TNG_files_subsample):
        with torch.no_grad():
            true_sky = np.load(tng_file)[np.newaxis, np.newaxis, :, :].astype(np_dtype)
            true_sky = torch.from_numpy(true_sky).to(device)
            if args.add_points:
                points = make_points_off_center_positions(Npix, margin=1, npoints=80).to(device)
                small_gauss = make_gaussians(1, Npix, 1, args.margin, std_dev_scaling=2, ngauss=30)
                small_gauss = torch.from_numpy(small_gauss.astype(args.dtype)).to(device)
                small_gauss = small_gauss / small_gauss.mean() * true_sky.mean() * 0.15
                points = points / points.mean() * true_sky.mean() * 0.15
                true_sky += points + small_gauss
                true_sky = true_sky / true_sky.max()

            true_sky = F.pad(true_sky, (args.margin, args.margin, args.margin, args.margin))
            if args.add_noise:
                noise = torch.normal(0, 1e-4, (1, 1, Npix + 2 * args.margin, Npix + 2 * args.margin)).to(device)
                true_sky_noise = true_sky + noise

            dirty_im = backproject(sky_model=true_sky_noise, psf=psf_trim, device=device)
            dirty_im = torchvision.transforms.functional.center_crop(dirty_im, Npix + 2 * args.margin)
            dirty_im_max = dirty_im.max()
            dirty_im = dirty_im / dirty_im_max
            true_sky = true_sky / dirty_im_max

            true_sky_array[ind] = true_sky
            dirty_sky_array[ind] = dirty_im

    np.save(Path(args.output_dir, f'{args.out_stem}true_sky.npy'), true_sky_array.cpu().numpy())
    np.save(Path(args.output_dir, f'{args.out_stem}dirty_sky.npy'), dirty_sky_array.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation for Deconvolution")
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for prepared data')
    parser.add_argument('--out_stem', type=str, required=True, help='Output Stem for prepared data')
    parser.add_argument('--tng_dir', type=str, default='sources/tng_data', help='Path to TNG data directory')
    parser.add_argument('--psf_file', type=str, default='sources/eso/eso137_psf_centered.fits', help='Path to PSF file')
    parser.add_argument('--margin', type=int, default=0, help='Margin size for padding')
    parser.add_argument('--add_noise', type=int, default=1, choices=[0, 1], help='Whether to add noise to the data')
    parser.add_argument('--add_points', type=int, default=1, choices=[0, 1], help='Whether to add points to the data')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to use')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], help='Data type for tensors')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()
    main(args)
