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

    def merge_tiles(self, image_id, tile_masks, tile_overlap = 0):
        orig_size = self.get_orig_size(image_id)
        mimg = np.zeros((*orig_size,3))
        tile_coords = self.tile_coords[image_id]
        objects = []
        ycoords,xcoords = zip(*tile_coords)
        ycoords = list(dict.fromkeys(ycoords))
        xcoords = list(dict.fromkeys(xcoords))
        hoverlap = int(tile_overlap / 2)
        for x,xcoord in enumerate(xcoords):
            for y,ycoord in enumerate(ycoords):
                tile = tile_masks[x*len(ycoords)+y]
                tsy,tsx = hoverlap,hoverlap
                if x==0: tsx = 0
                if y==0: tsy = 0
                tey,tex = tile.shape[:2]
                if ycoord+tey > mimg.shape[0]: tey = mimg.shape[0]-ycoord
                if xcoord+tex > mimg.shape[1]: tex = mimg.shape[1]-xcoord

                mimg[ycoord+tsy:ycoord+tey,xcoord+tsx:xcoord+tex,:] = tile[tsy:tey,tsx:tex,:]

        return mimg

    def get_orig_size(self, image_id):
        return self.orig_size.get(image_id, (0,0))
