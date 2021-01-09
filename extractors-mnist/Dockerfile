FROM tensorflow/tensorflow

RUN pip install pyclowder

COPY requirements.txt /home/requirements.txt

RUN pip install -r /home/requirements.txt

#RUN pip install --upgrade pip

COPY tensorflow_model_mnist.h5 /home/tensorflow_model_mnist.h5

COPY test_image.png /home/test_image.png

COPY new_tensorflow_extractor.py /home/new_tensorflow_extractor.py

COPY * /home/

CMD ["python", "/home/new_tensorflow_extractor.py"]


