import os
from operator import itemgetter
import skimage
import numpy as np

# Defined by images used for training the weights
IMAGE_SIZE = (256,256)

class Dataset():
    def __init__(self):
        self.orig_size = {}
        self.tile_coords = {}

    def load_image(self, image_id, img_path, tile_overlap = 0):
        image = skimage.io.imread(img_path)
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        self.orig_size[image_id] = image.shape[:2]
        # Make image at least min size
        if image.shape[0] < IMAGE_SIZE[0] or image.shape[1] < IMAGE_SIZE[1]:
            maxy = max(image.shape[0],IMAGE_SIZE[0])
            maxx = max(image.shape[1],IMAGE_SIZE[1])
            tmp = np.zeros((maxy,maxx,3), dtype=image.dtype)
            tmp[:image.shape[0],:image.shape[1],:] = image
            image = tmp
        tiles = self.crop_tiles(image_id, image, tile_overlap)
        return tiles

    def crop_tiles(self, image_id, image, tile_overlap = 0):
        # All tiles need to be IMAGE_SIZE for prediction
        if tile_overlap > max(IMAGE_SIZE) / 2:
            tile_overlap = max(IMAGE_SIZE) / 2

        tile_coords = []
        tiles = []
        if image.shape[0] > IMAGE_SIZE[0] or image.shape[1] > IMAGE_SIZE[1]:
            ycoords = list(range(0, image.shape[0], IMAGE_SIZE[0]-tile_overlap))
            xcoords = list(range(0, image.shape[1], IMAGE_SIZE[1]-tile_overlap))
            if xcoords[-1] + IMAGE_SIZE[1] >= image.shape[1]:
                xcoords[-1] = image.shape[1] - IMAGE_SIZE[1]
            if ycoords[-1] + IMAGE_SIZE[0] >= image.shape[0]:
                ycoords[-1] = image.shape[0] - IMAGE_SIZE[0]
            xcoords = list(filter(lambda x: x+IMAGE_SIZE[1] <= image.shape[1], xcoords))
            ycoords = list(filter(lambda x: x+IMAGE_SIZE[0] <= image.shape[0], ycoords))
            tile_coords = [(y,x) for x in xcoords for y in ycoords]
        else:
            tile_coords.append((0,0))

        # Remove duplicates
        tile_coords = list(dict.fromkeys(tile_coords))

        for t in tile_coords:
            tile = image[t[0]:t[0]+IMAGE_SIZE[0], t[1]:t[1]+IMAGE_SIZE[1], :]
            tiles.append(tile)
        
        self.tile_coords[image_id] = tile_coords
        return tiles

    def merge_tiles(self, image_id, tile_masks):
        orig_size = self.get_orig_size(image_id)
        mask_img = np.zeros(orig_size, dtype=np.bool)
        tile_coords = self.tile_coords[image_id]
        objects = []

        for tile,coords in zip(tile_masks, tile_coords):
            mask = tile
            # Cut added regions from small images
            if mask.shape[0] > orig_size[0]:
                mask = mask[:orig_size[0],:]
            if mask.shape[1] > orig_size[1]:
                mask = mask[:,:orig_size[1]]
            mask[mask < 2] = 0
            mask[mask == 2] = 1
            mrows = np.any(mask, axis=1)
            mcols = np.any(mask, axis=0)
            try:
                rmin,rmax = np.where(mrows)[0][[0, -1]]
                cmin,cmax = np.where(mcols)[0][[0, -1]]
            except:
                continue
                
            mask = mask[rmin:rmax+1, cmin:cmax+1]
            m_coords = [0,0,0,0]
            m_coords[0] = coords[0] + rmin
            m_coords[2] = coords[0] + rmax
            m_coords[1] = coords[1] + cmin
            m_coords[3] = coords[1] + cmax
            objects.append({'roi': m_coords,
                            'mask': mask})

        for obj in objects:
            roi = obj['roi']
            mcrop = mask_img[roi[0]:roi[2]+1, roi[1]:roi[3]+1]
            if ~(mcrop & obj['mask']).any():
                mcrop = mcrop | obj['mask']
                mask_img[roi[0]:roi[2]+1, roi[1]:roi[3]+1] = mcrop
        
        return mask_img.astype(np.uint8)

    def get_orig_size(self, image_id):
        return self.orig_size.get(image_id, (0,0))
