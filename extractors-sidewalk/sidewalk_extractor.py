#!/usr/bin/env python

import logging
import os
import re
import subprocess
import tempfile

from pyclowder.extractors import Extractor
import pyclowder.files
import pyclowder.utils

import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

import skimage.draw

class SidewalkConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sidewalk"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + sidewalk

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(SidewalkConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def sidewalk(inputfile, outputfile):
    config = InferenceConfig()

    # Directory to save logs and trained model
    MODEL_DIR = 'logs'

    # Local path to trained weights file
    SIDEWALK_MODEL_PATH = 'mask_rcnn_sidewalk.h5'

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on sidewalk
    model.load_weights(SIDEWALK_MODEL_PATH, by_name=True)

    # Run model detection and generate the color splash effect
    # Read image
    image = skimage.io.imread(inputfile)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    skimage.io.imsave(outputfile, splash)

class SidewalkExtractor(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)


    def process_message(self, connector, host, secret_key, resource, parameters):
        # Process the file and upload the results

        inputfile = resource["local_paths"][0]
        file_id = resource['id']
        image_type = 'png'

        self.execute_command(connector, host, secret_key, inputfile, file_id, resource, image_type)

    @staticmethod
    def execute_command(connector, host, key, inputfile, fileid, resource, ext):
        logger = logging.getLogger(__name__)

        (fd, tmpfile) = tempfile.mkstemp(suffix='.' + ext)
        try:
            # close tempfile
            os.close(fd)

            sidewalk(inputfile, tmpfile)

            if os.path.getsize(tmpfile) != 0:
                # upload result
                pyclowder.files.upload_preview(connector, host, key, fileid, tmpfile, None)
                connector.status_update(pyclowder.utils.StatusMessage.processing, resource,
                                            "Uploaded preview of type %s" % ext)
            else:
                logger.warning("Extraction resulted in 0 byte file, nothing uploaded.")

        except subprocess.CalledProcessError as e:
            logger.error(binary + " : " + str(e.output))
            raise
        finally:
            try:
                os.remove(tmpfile)
            except OSError:
                pass

if __name__ == "__main__":
    extractor = SidewalkExtractor()
    extractor.start()

