import json
import numpy as np
from pathlib import Path
import argparse
import torch
from torch.optim import AdamW
from deconvutils.models import nn_convolve_straight
from deconvutils.unet import UNet
from deconvutils.utils import load_checkpoint, get_psf_kernel, copy_scripts, set_defaults
from deconvutils.trainer import trainer, loss_fn_l1
from deconvutils.plots import plot_training_results, plot_train_test_losses
from deconvutils.datasets import gaussnoise,  PSF_convolve, generate_dataloaders
import torchvision
import os
import sys
from torch.optim import lr_scheduler
import logging
import time


def main(args):
    start_time = time.time()

    device = torch.device(args.device)
    np_dtype = set_defaults(args.dtype)

    if args.model == 'unet':
        model = UNet(batchnorm=args.use_bn)

    model = nn_convolve_straight(input_model=model, dist_shift=False)
    
    model_checkpoint_read = Path(args.model_checkpoint_dir, 'deconv_model.pt')        
    checkpoint_write = Path(args.output_dir, 'deconv_model.pt')

    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = vars(args)
    with open(Path(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    logging.basicConfig(filename=Path(args.output_dir, 'training.log'), format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Data
    sky = torch.from_numpy(np.load(args.true_sky_file).astype(np_dtype))
    sky_convolved_noise = torch.from_numpy(np.load(args.dirty_sky_file).astype(np_dtype))

    _, _, h, w = sky.shape
    assert h == w, 'image not square'
    psf_trim = torch.tensor(get_psf_kernel(2 * h, psf_file=args.psf_file).astype(np_dtype)).to(device)
    gauss_noise = gaussnoise(1e-4, 1e-2)

    psf_convolve = PSF_convolve(psf_trim, device='cuda')
    transforms = torchvision.transforms.Compose([gauss_noise, psf_convolve])
    
    train_loader, test_loader = generate_dataloaders(sky_convolved_noise, sky, batch_size=args.batch_size, validation_split=args.val_split, transform=gauss_noise)

    # Model & Optimisation
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, threshold=1e-4)

    if model_checkpoint_read.exists():
        print('loading Model Checkpoint')
        model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, model_checkpoint_read,
                                                      checkpoint_read_model_only=args.checkpoint_read_model_only, device=device)

    if not args.no_train:
        # Train
        print('training')
        model, metrics = trainer(train_loader, test_loader, model, loss_fn_l1, optimizer, scheduler,
                                 checkpoint_write, args.epochs, crop_target=True, device=device)
        plot_train_test_losses(metrics, outfile=Path(args.output_dir, 'losses.png'))

    # Plot results
    for index in range(5):
        print('plot', index)
        print(Path(args.output_dir, f'deconv_results_{index}.png'))
        plot_training_results(test_loader, model, psf_trim, outfile=Path(args.output_dir, f'deconv_results_{index}.png'), index=index, device=device)

    copy_scripts(output_dir=args.output_dir)

    logger.info(f'Training pipeline took {time.time() - start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Deconvolution Training Script")
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--true_sky_file', type=str, default='sources/eso_1k/true_sky.npy', help='Path to true sky file')
    parser.add_argument('--dirty_sky_file', type=str, default='sources/eso_1k/dirty_sky.npy', help='Path to dirty sky file')
    parser.add_argument('--model_checkpoint_dir', type=str, default='', help='model checkpoint directory')
    parser.add_argument('--psf_file', type=str, default='sources/eso/eso137_psf_centered.fits', help='Path to PSF file')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=60, help='Max number of training epochs')
    parser.add_argument('--model', type=str, default='unet', choices=['unet'], help='Model to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--use_bn', type=int, default=1, help='Whether to use batch normalization')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], help='Data type for tensors')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--checkpoint_read_model_only', action='store_true')
    parser.add_argument('--margin', type=int, default=8, help='Margin size for padding')

    args = parser.parse_args()
    main(args)
