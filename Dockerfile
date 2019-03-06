FROM clowder/extractors-simple-extractor:onbuild

ENV EXTRACTION_MODULE="tensorflow_extractor"
ENV EXTRACTION_FUNC="extract"

COPY temp/tensorflow_model_mnist.h5 tensorflow_model_mnist.h5