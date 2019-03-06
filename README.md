# Clowder Tensorflow Examples

Examples of how to use Clowder and Tensorflow together:

- Loading data from Clowder and creating Tensorflow datasets


## Extractor

Build and run using docker:

```docker build -t clowder/ncsa.tensorflow.mnist .```

```docker run --rm -ti --name=tensorflow -e 'RABBITMQ_URI=amqp://guest:guest@141.142.60.207/%2f' -e 'REGISTRATION_ENDPOINTS=http://141.142.60.207:9000/api/extractors?key=r1ek3rs' clowder/ncsa.tensorflow.mnist```