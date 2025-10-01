import matplotlib.pyplot as plt
import torch
from pathlib import Path
from .utils import backproject
import numpy as np

def plot_train_test_losses(metric_dict, outfile):
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    plt.plot(metric_dict['train_losses'], label='train')
    plt.plot(metric_dict['test_losses'], label='test')
    plt.legend()
    plt.yscale('log')
    plt.savefig(outfile)


def plot_training_results(test_loader, model, psf_trim, outfile, index=0, device='cuda'):
    plt.rcParams.update({'font.size': 15})

    tester = iter(test_loader)
    model = model.to(device)
    x, y = next(tester)
    x = x[index:index + 1]
    y = y[index:index + 1]
    with torch.no_grad():
        model.eval()
        x = x.to(device)
        x_pred = model(x)
    psf_trim = psf_trim.to(device)
    backprojected = backproject(sky_model=x_pred, psf=psf_trim)
    backprojected = backprojected.cpu().detach().numpy()[0, 0, :, :]
    x = x.cpu().detach().numpy()[0, 0, :, :]
    x_pred = x_pred.cpu().detach().numpy()[0, 0, :, :]
    gt = y.detach().numpy()[0, 0, :, :]

    # backprojected = backprojected / backprojected.max() * x.max()
    residual = x - backprojected
    if x_pred.max()<=0:
        print(' x_pred.max()<=0')
    fig, axes = plt.subplots(figsize=(12, 12), facecolor='w', dpi=200)

    plt.subplot(2, 2, 1)
    plt.title('Dirty Image\n stddev:%.4f' % x.std())
    plt.imshow(x)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Ground Truth')
    plt.imshow(gt + 1e-8, norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=gt.max()))
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('Residual \n stddev:%.4f' % residual.std())
    plt.imshow(residual)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('Model Image')
    plt.imshow(x_pred + 1e-8, norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=x_pred.max()))
    plt.colorbar()
    plt.savefig(outfile)

    plt.figure(figsize=(20, 20), facecolor='w', dpi=200)
    plt.title('Model Image Zoom')
    plt.imshow(x_pred[10:150, 10:150] + 1e-8, norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=x_pred.max()))
    plt.colorbar()
    outfile = Path(outfile)
    plt.savefig(Path(outfile.parent, outfile.stem + '_model_zoom.png'))
