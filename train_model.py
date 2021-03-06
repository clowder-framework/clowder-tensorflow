#!/usr/bin/python3

import argparse
import requests
import json
import os

import utils

import tensorflow as tf
from tensorflow import keras

import load_data


# Simple model creation from Tensorflow documentation
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

# Upload model to Clowder as HD5 file and metadata about author and tenforflow version
def upload_model(savefile, author, tfversion, clowderurl, clowderkey, datasetid):
    # upload file
    url = "{}/api/uploadToDataset/{}?key={}".format(clowderurl, datasetid, clowderkey)

    if os.path.exists(savefile):
        result = requests.post(url, files={"File": open(savefile, 'rb')})

        uploadedfileid = result.json()['id']
        print("Uploaded model {}/files/{}?key={}".format(clowderurl, uploadedfileid, clowderkey))

        # add metadata
        metadata = {
            '@context': ['https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld',
                         {'Author': 'http://purl.org/dc/terms/Author',
                          'Tensorflow Version': 'https://clowder.ncsa.illinois.edu/terms/tensorflow/version'}],
            # 'attachedTo': {
            #     'resourceType': 'cat:extractor',
            #     'id': resource_id
            # },
            'agent': {
                '@type': 'cat:extractor',
                'extractor_id': 'http://localhost:9000/extractors/ncsa.tensorflow/2.0'
            },
            'content': {"Author": author, "Tensorflow Version": tf.__version__}
        }
        headers = {'Content-Type': 'application/json'}
        r = requests.post("{}/api/files/{}/metadata.jsonld?key={}".format(clowderurl, uploadedfileid, clowderkey),
                          headers=headers, data=json.dumps(metadata))
        response = r.json()
        print("Metadata added to file %s", uploadedfileid)

    else:
        print("unable to upload file %s (not found)", savefile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clowderurl", help="Clowder URL")
    parser.add_argument("--clowderkey", help="Clower API key")
    parser.add_argument("--datasetid", help="Clowder dataset id")
    args = parser.parse_args()

    training_label = "Training Label" # 'basic_caltech101_score'
    (file_paths, labels) = load_data.download_data(args.clowderurl, args.clowderkey, args.datasetid, training_label)
    dataset = load_data.load_data(file_paths, labels)

    # Create a basic model instance
    model = create_model()
    model.summary()

    # You need to use a keras.optimizer to restore the optimizer state from an HDF5 file.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(dataset, epochs=1, steps_per_epoch=2)

    # Save entire model to a HDF5 file
    saved_model_location = 'temp/tensorflow_model_mnist.h5'
    model.save(saved_model_location)

    upload_model(saved_model_location, 'Luigi Marini', tf.__version__, args.clowderurl, args.clowderkey, args.datasetid)

    # Recreate the exact same model, including weights and optimizer.
    new_model = keras.models.load_model(saved_model_location)
    new_model.summary()

    image_string = tf.read_file("/Users/lmarini/data/clowder-demo-files/IMG_0997.jpg")
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])

    print(image_resized.shape)

    predictions = model.predict([image_resized], steps=2)

    print(predictions)

    # delete model on disk
    # utils.delete_temp_files([saved_model_location])
