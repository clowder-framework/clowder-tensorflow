#!/usr/bin/env python

"""Example extractor based on the clowder code."""

import logging
import os
from tensorflow import keras
import time
import subprocess

from pyclowder.extractors import Extractor
import pyclowder.files


class TensorFlowExtractor(Extractor):
    """Count the number of characters, words and lines in a text file."""
    def __init__(self):
        Extractor.__init__(self)

        # add any additional arguments to parser
        # self.parser.add_argument('--max', '-m', type=int, nargs='?', default=-1,
        #                          help='maximum number (default=-1)')

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

    def process_message(self, connector, host, secret_key, resource, parameters):
        # Process the file and upload the results

        logger = logging.getLogger(__name__)
        input_file = resource["local_paths"][0]
        file_id = resource['id']

        # call actual program
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

        # Recreate the exact same model, including weights and optimizer.
        logger.debug("where am I?")
        logger.debug(os.getcwd())
        logger.debug('what is here?')
        logger.debug(os.listdir(os.getcwd()))
        saved_model_location = "/home/tensorflow_model_mnist.h5"
        model = keras.models.load_model(saved_model_location)
        model.summary()

        img = keras.preprocessing.image.load_img(path=input_file, color_mode="grayscale", target_size=(28, 28, 1))
        img = keras.preprocessing.image.img_to_array(img)
        test_img = img.reshape((1, 784))
        img_class = model.predict_classes(test_img)
        prediction = img_class[0]
        classname = img_class[0]
        print("Class: ", classname)

        # img = img.reshape((28, 28))
        # plt.imshow(img)
        # plt.title(classname)
        # plt.show()

        metadata = {
            'label': classes[int(prediction)]
        }
        result = {
            'metadata': metadata
        }

        metadata = self.get_metadata(result, 'file', file_id, host)
        logger.debug(metadata)

        # upload metadata
        pyclowder.files.upload_metadata(connector, host, secret_key, file_id, metadata)


if __name__ == "__main__":
    print('starting the extrarctor')
    extractor = TensorFlowExtractor()

    extractor.start()