# DEV
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder

RUN apt-get update && \
    apt-get install --no-install-recommends -y git ninja-build build-essential python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

ARG QUANT_MODEL_URL QUANT_BRANCH QUANT_SHA
RUN git clone ${QUANT_MODEL_URL} /build

WORKDIR /build

RUN git checkout ${QUANT_BRANCH}
RUN git reset --hard ${QUANT_SHA}
RUN --mount=type=cache,target=/root/.cache/pip pip install torch
RUN pip install -r requirements.txt


ARG TORCH_CUDA_ARCH_LIST
RUN python3 setup_cuda.py bdist_wheel -d .

# RUNTIME
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y libportaudio2 libasound-dev git python3-dev python3-pip make g++ ninja-build cuda-compiler-11-8 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio


# WebUI
ARG WEBUI_BRANCH WEBUI_SHA
RUN --mount=type=cache,target=/root/.cache/pip <<EOF
git config --global http.postBuffer 1048576000
git clone https://github.com/oobabooga/text-generation-webui.git /app
git checkout ${WEBUI_BRANCH}
git reset --hard ${WEBUI_SHA}
cd /app/extensions/api && pip install -r requirements.txt
cd /app/extensions/elevenlabs_tts && pip install -r requirements.txt
cd /app/extensions/google_translate && pip install -r requirements.txt
cd /app/extensions/silero_tts && pip install -r requirements.txt
cd /app/extensions/whisper_stt && pip install -r requirements.txt
cd /app && pip install -r requirements.txt
EOF

COPY --from=builder /build /app/repositories/GPTQ-for-LLaMa
RUN --mount=type=cache,target=/root/.cache/pip pip install /app/repositories/GPTQ-for-LLaMa/*.whl

# Get Default Model from ./models
COPY ./models /app/models

COPY ./docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# RUN cp /app/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so /app/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so

ENV CLI_ARGS=""
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN test -e /usr/local/cuda/bin/nvcc
ENTRYPOINT ["/app/docker/entrypoint.sh"]
