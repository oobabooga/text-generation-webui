### Builder image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder

RUN  --mount=type=cache,target=/var/cache/apt,rw apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv libmpich-dev cuda-nvcc-11-8 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/oobabooga/GPTQ-for-LLaMa /build

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip,rw \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch torchvision torchaudio deepspeed mpi4py mpi4py-mpich && \
    pip3 install git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit || true && \
    pip3 install monkeypatch || true && \
    pip3 install -r requirements.txt

# https://developer.nvidia.com/cuda-gpus
ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX}"
RUN . venv/bin/activate && python3 setup_cuda.py bdist_wheel -d .

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Docker image for GPTQ-for-LLaMa and Text Generation WebUI"

RUN --mount=type=cache,target=/var/cache/apt,rw apt-get update && \
    apt-get install --no-install-recommends -y python3-dev libportaudio2 libasound-dev git python3 python3-pip \
    make g++ libmpich-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /app

RUN --mount=type=cache,target=/root/.cache/pip,rw pip3 install virtualenv

WORKDIR /app

ARG WEBUI_VERSION
RUN test -n "${WEBUI_VERSION}" && git reset --hard ${WEBUI_VERSION} || echo "Using provided webui source"

# Create virtualenv
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    virtualenv venv && \
    . venv/bin/activate && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch torchvision torchaudio deepspeed mpi4py mpi4py-mpich zarr sentence_transformers xformers bs4

# Create virtualenv
COPY --from=builder /build /app/repositories/GPTQ-for-LLaMa
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    . venv/bin/activate && \
    pip3 install repositories/GPTQ-for-LLaMa/*.whl

# Copying all requirements files
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    . venv/bin/activate && pip3 install -r requirements.txt

COPY . /app/

RUN cp venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so

# Install extension requirements
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    . venv/bin/activate && \
    for ext in /app/extensions/*/requirements.txt; do \
    cd "$(dirname "$ext")"; \
    pip3 install -r requirements.txt; \
    done

ENV CLI_ARGS=""

EXPOSE ${CONTAINER_PORT:-7860} ${CONTAINER_API_PORT:-5000} ${CONTAINER_API_STREAM_PORT:-5005}
CMD . venv/bin/activate && python3 server.py ${CLI_ARGS}
