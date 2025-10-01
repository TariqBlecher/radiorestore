import numpy as np
import torch
from deconvutils.utils import wsclean_image_call, wsclean_predict, get_central_pix, backproject
from deconvutils.masker import create_mask
from deconvutils.models import nn_convolve_straight
from deconvutils.unet import UNet
from deconvutils.utils import load_checkpoint, get_psf_kernel, set_defaults
from deconvutils.tiling_class import TileMem

import time
import os
from astropy.io import fits
from pathlib import Path
import logging
import argparse


def main(args):
    start_time = time.time()

    np_dtype = set_defaults(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename= output_dir / 'deconvmajor.log', format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Follow WSCLEAN naming convention
    image_name_stem = output_dir / args.image_name_stem
    dirty_sky_file = image_name_stem.parent / (image_name_stem.name + '-dirty.fits') 
    model_file =  image_name_stem.parent / (image_name_stem.name + '-model.fits')

    # Create Initial Dirty Image and PSF
    if not dirty_sky_file.exists():
        os.system(wsclean_image_call(name= image_name_stem.as_posix(), Npix=args.Npix, ms=args.ms, subtract_model_col=False, make_psf=False))

    # Model 
    model = nn_convolve_straight(dist_shift=False)
    model,_,_ = load_checkpoint(model,None,None, checkpoint_read=args.model_checkpoint, checkpoint_read_model_only=True)
    model = model.to(args.device)

    # Read Data
    psf_trim = torch.tensor(get_psf_kernel(2*args.Npix, psf_file=args.psf_file).astype(np_dtype))
    psf_trim = psf_trim.to(args.device)
    fitshdr = fits.getheader(dirty_sky_file)
    x = torch.from_numpy(fits.getdata(dirty_sky_file).astype(np_dtype)).to(args.device)  
    x = get_central_pix(x,args.Npix)
    x = x.to(args.device)

    # Initialise clean image and mask
    clean_im_final = torch.zeros_like(x, device=args.device)
    mask = torch.zeros_like(x, device=args.device,dtype=torch.bool)

    # If continuing from previous iteration
    if args.continue_from_iteration:
        logger.info('Continuing from previous iteration')
        starting_ind=args.continue_from_iteration
        clean_im_final = torch.from_numpy(fits.getdata(model_file.as_posix().replace( '-model.fits', str(args.continue_from_iteration-1)+'-model.fits')).astype(np_dtype)).to(args.device)
        mask =  torch.from_numpy(fits.getdata(model_file.as_posix().replace( '-model.fits', str(args.continue_from_iteration-1)+'-mask.fits')).astype(np_dtype)).to(args.device)
    else:
        starting_ind=0


    for major_loop_index in range(starting_ind, args.num_major_loops):
        if major_loop_index!=0:
            residual_sky_file = image_name_stem.parent / (image_name_stem.name + str(major_loop_index) + '-dirty.fits')
            x = torch.tensor(fits.getdata(residual_sky_file).astype(np_dtype)).to(args.device)
            x = get_central_pix(x,args.Npix)

        logger.info(f'dirty image stats:    std: {x.std().item() :.4}, min: {x.min().item():.4}, max: {x.max().item():.4}')
        
        # # Minor Loop Cycle
        initial_max = x.max().item()
        for loop_ind in range(args.num_minor_loops):
            
            with torch.no_grad():
                tilemem = TileMem(x, args.window_size, args.margin, model, psf_trim, device=args.device)
                clean_im = tilemem.deconv_mem()

            
            mask =  create_mask(x, threshold=args.threshold, lower_cut=args.lower_cut, initial_mask=mask) 
            clean_im[torch.logical_not(mask)] = 0

            backprojected = backproject(sky_model=clean_im, psf=psf_trim, device='cpu')
               
            clean_im = clean_im * args.mgain
            backprojected = backprojected * args.mgain
            residual = x - backprojected

            logger.info(f'iter: {loop_ind}, residual image stats: std: {residual.std().item():.4}, min : {residual.min().item():.4}, max :{residual.max().item():.4}')
            x= residual.clone()
            clean_im_final+=clean_im

            if args.debug:
                fits.writeto(output_dir / f'{major_loop_index}_{loop_ind}_model.fits', clean_im_final.cpu().detach().numpy().astype(np.float32), header = fitshdr,overwrite=True)
                fits.writeto(output_dir / f'{major_loop_index}_{loop_ind}_residual.fits', residual.cpu().detach().numpy().astype(np.float32), header =fitshdr, overwrite=True)
                fits.writeto(output_dir / f'{major_loop_index}_{loop_ind}_mask.fits', mask.cpu().detach().numpy().astype(np.float32), header = fitshdr, overwrite=True)
            
            res_max = x.max().item()
            if res_max <= args.minor_cycle_stopping_threshold * initial_max:
                print(f'initial_max was {initial_max}, current max {res_max}.stopping minor cycle at {loop_ind}')
                break

        if not args.minor_cycle_only:
            # Write out Model
            fits.writeto(model_file.as_posix().replace('-model.fits', str(major_loop_index)+'-model.fits'), clean_im_final.cpu().numpy(), header=fitshdr, overwrite=True)

            # Write out Mask
            fits.writeto(model_file.as_posix().replace('-model.fits', str(major_loop_index)+'-mask.fits'), mask.cpu().numpy().astype(np.float32), header=fitshdr, overwrite=True)

            # Predict new model column
            os.system(wsclean_predict(image_name_stem=image_name_stem.parent / (image_name_stem.name + str(major_loop_index)), Npix=args.Npix, ms=args.ms))

            # Create new residual image
            os.system(wsclean_image_call(name=image_name_stem.parent / (image_name_stem.name +str(major_loop_index+1)), Npix=args.Npix, ms=args.ms, subtract_model_col=True, niter=0))
            logger.info(f'major cycle {major_loop_index}, time taken for last major loop: {time.time()-start_time}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Script")
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='model checkpoint')
    parser.add_argument('--psf_file', type=str, help='PSF')
    parser.add_argument('--num_minor_loops', type=int, default=6, help='Number of iterations')
    parser.add_argument('--num_major_loops', type=int, default=4, help='Number of iterations')
    parser.add_argument('--mgain', type=float, default=0.6)
    parser.add_argument('--Npix', type=int, default=5120)
    parser.add_argument('--lower_cut', type=float, default=0.1, help='Lower cut value')
    parser.add_argument('--window_size', type=int, default=512, help='Window size')
    parser.add_argument('--margin', type=int, default=8, help='Margin size')
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'], help='Data type for tensors')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--ms',  type=str)
    parser.add_argument('--image_name_stem', default='imagetest', type=str)
    parser.add_argument('--threshold', type=float, default=5, help='Masking Threshold')
    parser.add_argument('--continue_from_iteration', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--minor_cycle_stopping_threshold', type=float, default=0.2)
    parser.add_argument('--minor_cycle_only', action='store_true')

    args_cli = parser.parse_args()
    main(args_cli)
  

