import sys
import os
import math
import numpy as np
import skimage.io
from scipy import ndimage
import skimage.morphology
import skimage.feature
import skimage.filters
import skimage.segmentation
from cytomine.models import Job
from neubiaswg5 import CLASS_OBJSEG
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics
import utils.model_builder
import utils.metrics
import unet_utils
from unet_utils import Dataset

TILE_OVERLAP = 8

def label_image(probmap, boundary_weight, min_size):
    predmask = utils.metrics.probmap_to_pred(probmap, boundary_weight).astype(np.bool)
    predmask = skimage.morphology.remove_small_holes(predmask, area_threshold=min_size)
    distance = ndimage.distance_transform_edt(predmask)
    distance = skimage.filters.gaussian(distance, sigma=3)
    lmax = skimage.feature.peak_local_max(distance, indices=False, footprint=np.ones((3,3)), labels=predmask)
    markers = skimage.morphology.label(lmax)
    labelimg = skimage.morphology.watershed(-distance, markers, mask=predmask)
    labelimg = labelimg.astype(np.uint16)
    labelimg = skimage.morphology.remove_small_objects(labelimg, min_size=min_size)
    labelimg = skimage.segmentation.relabel_sequential(labelimg)[0].astype(np.uint16)
    return labelimg

def main(argv):
    base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
    problem_cls = CLASS_OBJSEG

    with NeubiasJob.from_cli(argv) as nj:
        nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        
        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, is_2d=True, **nj.flags)

        # 2. Run image analysis workflow
        nj.job.update(progress=25, statusComment="Launching workflow...")

        model = utils.model_builder.get_model_3_class(unet_utils.IMAGE_SIZE[0], unet_utils.IMAGE_SIZE[1], channels=3) # Create the model
        model.load_weights('/app/weights.h5')

        dataset = Dataset()
        for img in in_imgs:
            image_id = img.filename_no_extension
            tiles = dataset.load_image(image_id, img.filepath, TILE_OVERLAP)
            orig_size = dataset.get_orig_size(image_id)
            mask_img = np.zeros(orig_size, dtype=np.uint8)

            tile_stack = np.zeros((len(tiles),unet_utils.IMAGE_SIZE[0],unet_utils.IMAGE_SIZE[1],3))
            for i,tile in enumerate(tiles):
                tile_stack[i,:,:,:] = tile
            tile_stack = tile_stack / 255

            tile_masks = model.predict(tile_stack, batch_size=1)

            probmap = dataset.merge_tiles(image_id, tile_masks, tile_overlap = TILE_OVERLAP)
            labelimg = label_image(probmap, nj.parameters.boundary_weight, nj.parameters.nuclei_min_size)
            skimage.io.imsave(os.path.join(out_path,img.filename), labelimg)

        # 3. Upload data to BIAFLOWS
        upload_data(problem_cls, nj, in_imgs, out_path, **nj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        nj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, nj, in_imgs, gt_path, out_path, tmp_path, **nj.flags)

        # 5. Pipeline finished
        nj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
