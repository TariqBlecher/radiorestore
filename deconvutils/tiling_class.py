import numpy as np
import torch
from .utils import *
torch.backends.cudnn.benchmark = True


class TileMem(object):
    def __init__(self, dirty_im, window_size, margin, model, psf, device, factor=8):
        self.device = device

        self.dirty_im = dirty_im.to(device)
        self.model = model.to(device)
        self.psf = psf.to(device)

        assert (window_size + 2 * margin) % factor == 0, f'window_size + 2*margin should be divisible by {factor}'

        self.window_size = window_size
        self.margin = margin

        self.pad_needed = self.determine_pad_needed_to_ensure_whole_final_tile()
        self.pad_for_tiling()
        self.dirty_im_max = dirty_im.max() 
        self.dirty_im_processed /= self.dirty_im_max

    def determine_pad_needed_to_ensure_whole_final_tile(self):
        npix = self.dirty_im.shape[-1]
        # Total size of a tile including margin
        total_tile_size = self.window_size + 2 * self.margin
        # Number of tiles needed to cover the image without padding
        n_tiles = np.ceil(npix / self.window_size).astype(int)
        # Total size required to fit all tiles without cutting any tile
        total_size_needed = n_tiles * total_tile_size
        # Padding needed to make the image fit into tiles exactly
        pad_needed = total_size_needed - npix
        return pad_needed

    def pad_for_tiling(self):
        self.dirty_im_processed = torch.nn.functional.pad(self.dirty_im, (0, self.pad_needed, 0, self.pad_needed), mode='reflect')

    def recombine_tiles(self, start_inds, clean_tiles):
        clean_im = torch.zeros_like(self.dirty_im_processed)
        for ind, (hstart, vstart) in enumerate(start_inds):
            clean_im[0, 0, vstart:vstart + self.window_size, hstart:hstart + self.window_size] = clean_tiles[ind]

        return clean_im

    def deconv_mem(self, force_float=True):
        tiles, start_inds = create_tiles(self.dirty_im_processed, window_size=self.window_size, margin=self.margin, device=self.device)

        clean_tiles = torch.zeros_like(tiles)
        ntiles = tiles.shape[0]

        self.model.eval()
        if force_float:
            self.model = self.model.float()
            tiles = tiles.float()
            clean_tiles = clean_tiles.float()
        with torch.no_grad():
            for tile_index in range(ntiles):
                if torch.max(tiles[tile_index]) <= 0:
                    continue
                else:
                    temp_model = self.model.Umodel(tiles[tile_index])
                    temp_model = nn.functional.relu(temp_model)
                    if torch.isnan(temp_model).sum() > 0:
                        continue
                    else:
                        clean_tiles[tile_index] = temp_model
        if force_float:
            tiles = tiles.double()
            clean_tiles = clean_tiles.double()


        if self.margin:
            clean_tiles = clean_tiles[:, :, :, self.margin:-1 * self.margin, self.margin:-1 * self.margin]
        clean_im = self.recombine_tiles(clean_tiles=clean_tiles, start_inds=start_inds)

        original_height, original_width = self.dirty_im.shape[-2], self.dirty_im.shape[-1]
        clean_im = clean_im[:, :, :original_height, :original_width]
        
        clean_im = clean_im * self.dirty_im_max

        return clean_im
