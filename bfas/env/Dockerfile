FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

LABEL maintainer="muhammed.pektas@syntonym.com"

WORKDIR /app
ADD . /app/
ENV HOME=/app/


RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    build-essential \
    byobu \
    ca-certificates \
    git-core git \
    htop \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libxext6 \
    libsm6 \
    libxrender1 \
    libcupti-dev \
    libgl1 \
    openssh-server \
    software-properties-common \
    vim \
    -y \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6 \
    python3-pip \
    python3-setuptools \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip install -r ${HOME}bfas/env/requirements.txt
RUN pip install .

# Open Ports for TensorBoard, Jupyter, and SSH
EXPOSE 6006 8080



