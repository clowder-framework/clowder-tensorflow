# Clowder Tensorflow (Mask RCNN COCO) Example

Examples of how to use Clowder and Tensorflow together:

- Loading data from Clowder and creating Tensorflow datasets

Download the model weights to a file with the name ‘mask_rcnn_coco.h5‘ in the working directory.
* [Download Mask RCNN COCO H5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)

## Extractor

Build and run using docker:

```docker build -t clowder/ncsa.tensorflow.mrcnn .```

```docker run --rm -ti --name=tensorflow -e 'RABBITMQ_URI=amqp://guest:guest@141.142.60.207/%2f' -e 'REGISTRATION_ENDPOINTS=http://141.142.60.207:9000/api/extractors?key=r1ek3rs' clowder/ncsa.tensorflow.mnist```
