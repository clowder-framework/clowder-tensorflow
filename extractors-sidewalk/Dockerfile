FROM continuumio/miniconda3
  
WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "sidewalk", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure tensorflow is installed:"
RUN python -c "import tensorflow"

# Install pyclowder
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy Mask RCNN code
COPY mrcnn/ mrcnn

# The code to run when container is started:
COPY mask_rcnn_sidewalk.h5 .
COPY sidewalk_extractor.py extractor_info.json policy.xml ./

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sidewalk", "python", "sidewalk_extractor.py"]
