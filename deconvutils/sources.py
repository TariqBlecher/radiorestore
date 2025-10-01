import numpy as np
import logging
from photutils.datasets import make_gaussian_sources_image
from astropy.table import QTable
import torch

logger = logging.getLogger(__name__)

def make_points(out_shape, margin, cutoff=0.995, scaling=10):
    point_sources = (np.random.random(out_shape) > cutoff).astype(int)
    point_sources = point_sources * scaling * np.random.exponential(2, point_sources.shape)
    point_sources = create_margin(point_sources, margin)
    if point_sources.max() == 0:
        print('NO POINT SOURCES!')
    logger.info('average flux value for point sources %s', point_sources.mean())
    logger.info('peak flux value for point sources %s', point_sources.max())
    return point_sources

def attenuate(d):
    return (torch.sqrt(torch.tensor(2.)) - d) / (torch.sqrt(torch.tensor(2.))+d)

def dist_attenuate(x,y,xref,yref):
    return attenuate(torch.sqrt((x-xref)**2+(y-yref)**2))

def make_points_off_center_positions(npix, margin=2, npoints = 50):
    field = torch.zeros((npix,npix))
    x_pos= margin + torch.rand(npoints)*(npix - 2*margin)
    y_pos= margin + torch.rand(npoints)*(npix - 2*margin)
    amplitudes = torch.exp(torch.rand(npoints)) 

    x_pos_lower = torch.floor(x_pos).int()
    x_pos_upper = torch.ceil(x_pos).int()
    y_pos_lower = torch.floor(y_pos).int()
    y_pos_upper = torch.ceil(y_pos).int()

    distances_ll = dist_attenuate(x_pos,y_pos,x_pos_lower,y_pos_lower)
    distances_lu = dist_attenuate(x_pos,y_pos,x_pos_lower,y_pos_upper)
    distances_ul = dist_attenuate(x_pos,y_pos,x_pos_upper,y_pos_lower)
    distances_uu = dist_attenuate(x_pos,y_pos,x_pos_upper,y_pos_upper)

    field[y_pos_lower, x_pos_lower] = amplitudes * distances_ll
    field[y_pos_lower, x_pos_upper] = amplitudes * distances_lu
    field[y_pos_upper, x_pos_lower] = amplitudes * distances_ul
    field[y_pos_upper, x_pos_upper] = amplitudes * distances_uu

    return field[None, :, :]

def create_margin(arr, margin):
    if margin == 0:
        pass
    else:
        arr[:, :, :margin, :] = 0
        arr[:, :, -1 * margin:, :] = 0
        arr[:, :, :, :margin] = 0
        arr[:, :, :, -1 * margin:] = 0
    return arr

def make_gaussians(nimgs, Npix, nchan, margin, std_dev_scaling=3, ngauss=40, std_dev_min=0):
    sky =  np.zeros((nimgs, nchan, Npix, Npix))
    for i in range(nimgs):
        # make a table of Gaussian sources
        table = QTable()
        table['amplitude'] = np.random.rand(ngauss)
        table['x_mean'] = np.random.randint(2*margin, Npix-2*margin, ngauss)
        table['y_mean'] = np.random.randint(2*margin, Npix-2*margin, ngauss)
        table['x_stddev'] = std_dev_min + std_dev_scaling * np.random.rand(ngauss)
        table['y_stddev'] = std_dev_min + std_dev_scaling * np.random.rand(ngauss)
        table['theta'] = np.pi*np.random.rand(ngauss)
        shape = (Npix, Npix)
        image1 = make_gaussian_sources_image(shape, table)
        sky[i,0,:,:] = image1
    return sky