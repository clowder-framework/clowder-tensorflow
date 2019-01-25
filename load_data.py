#!/usr/bin/python3

import argparse
import tensorflow as tf
import requests
import os

# Temporary directory to which files from Clowder are downloaded to. They are deleted after being used.
temp_directory = "./temp"

# Must be called at program startup.
tf.enable_eager_execution()


# Reads an image from a file, decodes it into a dense tensor, and resizes it to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


def download_file(url, filename):
    file_path = os.path.join(temp_directory, filename)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_path, 'wb') as fd:
            for chunk in r.iter_content(2000):
                fd.write(chunk)
    return file_path


def download_data(clowderurl, clowderkey, datasetid):
    r = requests.get("{}/api/datasets/{}/files?key={}".format(clowderurl, datasetid, clowderkey))
    files = r.json()
    print(files)
    file_paths = []
    labels = []
    for f in files:
        if f.get('contentType').startswith('image'):
            # dowload metadata
            metadata_url = "{}/api/files/{}/metadata.jsonld?key={}".format(clowderurl, f['id'], clowderkey)
            print(metadata_url)
            r = requests.get(metadata_url)
            try:
                metadata = r.json()
                for m in metadata:
                    score = m.get('content').get('basic_caltech101_score')
                    if score is not None:
                        print("score: " + score)
                        labels.append(int(float(score) * 100))
                        # Download file to temp directory.
                        # The file metadata includes path on disk. If file system is shared this could be replaced by
                        # reading the file directly from the file system.
                        download_url = "{}/api/files/{}/blob?key={}".format(clowderurl, f['id'], clowderkey)
                        print("download_url " + download_url)
                        path = download_file(download_url, f['filename'])
                        file_paths.append(path)
            except ValueError:
                print(r)

    print(file_paths)
    print(labels)
    return (file_paths, labels)


def delete_temp_files(file_paths):
    for file_path in file_paths:
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def load_data(file_paths, labels):
    filenames = tf.constant(file_paths)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)

    # delete files downloaded from Clowder
    delete_temp_files(file_paths)

    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("clowderurl", help="Clowder URL")
    parser.add_argument("clowderkey", help="Clower API key")
    parser.add_argument("datasetid", help="Clowder dataset id")
    args = parser.parse_args()
    (file_paths, labels) = download_data(args.clowderurl, args.clowderkey, args.datasetid)
    dataset = load_data(file_paths, labels)

