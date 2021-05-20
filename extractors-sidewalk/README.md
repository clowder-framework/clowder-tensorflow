# Clowder Tensorflow (Sidewalk) Example

Examples of how to use Clowder and Tensorflow together:

- Loading data from Clowder and creating Tensorflow datasets

Download the model weights to a file with the name ‘mask_rcnn_sidewalk.h5‘ in the working directory.
* [Download Mask RCNN SIDEWALK H5](https://opensource.ncsa.illinois.edu/bitbucket/projects/BD/repos/smu-dl/browse/sidewalk/asset/mask_rcnn_sidewalk.h5)

## Extractor

Build and run using docker:

```docker build -t clowder/ncsa.tensorflow.sidewalk .```


```docker run -ti --rm --network clowder_clowder -e RABBITMQ_URI=amqp://guest:guest@rabbitmq/%2F clowder/ncsa.tensorflow.sidewalk```

