FROM tensorflow/tensorflow:2.9.2-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    python3-pip

RUN python -m pip install -U pip
RUN pip install tensorflow

## Install Flask
RUN pip install Flask

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"
RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

## Copy Flask Application into image
COPY flask_application.py /home/tensorflow/flask_application.py

## Copy aux files into images
ADD exported-models /tensorflow/models/research/exported-models
ADD annotations /tensorflow/models/research/annotations
ADD images /tensorflow/models/research/images

## Set image run command to Flask application
CMD ["python", "/home/tensorflow/flask_application.py"]
