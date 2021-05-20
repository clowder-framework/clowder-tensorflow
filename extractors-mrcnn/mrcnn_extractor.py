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
#from pycocotools import coco
from mrcnn.config import Config

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 81

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def mrcnn(inputfile, outputfile):
    """
    This function generates bounding boxes and segmentation masks for each instance of an object in the image was uploaded.

    :param input_file_path: Full path to the input file
    :return: Result dictionary containing metadata about lines, words, and characters in the input file
    """

    config = InferenceConfig()

    # Directory to save logs and trained model
    MODEL_DIR = 'logs'

    # Local path to trained weights file
    COCO_MODEL_PATH = 'mask_rcnn_coco.h5'

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # Load a image
    image = skimage.io.imread(inputfile)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    plt = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], inputfile)
    plt.savefig(outputfile, bbox_inches='tight', pad_inches=0.0)

class MrcnnExtractor(Extractor):
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

            mrcnn(inputfile, tmpfile)

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
    extractor = MrcnnExtractor()
    extractor.start()

