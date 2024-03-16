FROM continuumio/miniconda3

RUN mkdir -p S2_Coursework

COPY . /S2_Coursework

WORKDIR /S2_Coursework

RUN conda env update -f environment.yml --name S2CW

RUN apt-get update && apt-get install -y \
    git

RUN echo "conda activate S2CW" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN git init
RUN pre-commit install
