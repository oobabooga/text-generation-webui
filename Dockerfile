FROM python:3.10.6-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git software-properties-common gnupg

COPY . /app

WORKDIR /app

RUN dpkg -i /app/installers/cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb
RUN cp /var/cuda-repo-debian11-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN add-apt-repository contrib
RUN apt-get update
RUN apt-get -y install cuda \
 && apt -y remove nvidia-* \
 && rm -rf /var/cuda-repo-debian11-11-6-local

RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/extensions/google_translate/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/extensions/silero_tts/requirements.txt

CMD python server.py --auto-devices --cai-chat --load-in-8bit --bf16 --listen --listen-port=8888
