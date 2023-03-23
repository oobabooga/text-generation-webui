FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

WORKDIR /app

RUN apt update -y
RUN apt upgrade -y

RUN apt install -y wget libxml2 g++ wget git pip

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b

COPY entrypoint.sh /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
