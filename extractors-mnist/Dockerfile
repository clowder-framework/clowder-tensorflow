FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

ARG conda_env=mnist

RUN conda clean -a
#RUN pip install --upgrade pip

COPY tensorflow_model_mnist.h5 /home/tensorflow_model_mnist.h5

COPY test_image.png /home/test_image.png

COPY new_tensorflow_extractor.py .

COPY new_tensorflow_extractor.py /home/new_tensorflow_extractor.py

COPY extractor_info.json .

COPY extractor_info.json /home/extractor_info.json

COPY environment.yml /home/environment.yml

RUN conda env create -f /home/environment.yml

COPY test.py .

COPY test.py /home/test.py

COPY run.py /home/run.py

ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env

ENV PATH /opt/conda/envs/mro_env/bin:$PATH
RUN /bin/bash -c "source activate mnist"

RUN python -m pip install --ignore-installed pyclowder

# The code to run when container is started:

# CMD ["conda", "run", "-n", "myenv", "python","-u", "/home/new_tensorflow_extractor.py"]

CMD ["python", "/home/run.py"]


