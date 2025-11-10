FROM ubuntu:22.04
ENV TZ=Europe/Rome \
    DEBIAN_FRONTEND=noninteractive
LABEL author="Andrea Di Marco, Fabio Polese, Ph.D."
LABEL version="0.0.1"


# UBUNTU SETUP
RUN apt update -y \
    && apt install -y python3 \
                      python3-pip \
    && apt update -y


# MP4 FILES
RUN apt-get install -y --no-install-recommends ffmpeg


# PYTHON LIBRARIES
WORKDIR /
COPY install/python-requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


