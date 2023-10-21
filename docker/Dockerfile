FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/oobabooga/GPTQ-for-LLaMa /build

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip,rw \
    python3 -m venv /build/venv && \
    . /build/venv/bin/activate && \
    pip3 install --upgrade pip setuptools wheel ninja && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install -r requirements.txt

# https://developer.nvidia.com/cuda-gpus
# for a rtx 2060: ARG TORCH_CUDA_ARCH_LIST="7.5"
ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX}"
RUN . /build/venv/bin/activate && \
    python3 setup_cuda.py bdist_wheel -d .

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Docker image for GPTQ-for-LLaMa and Text Generation WebUI"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw apt-get update && \
    apt-get install --no-install-recommends -y python3-dev libportaudio2 libasound-dev git python3 python3-pip make g++ ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip,rw pip3 install virtualenv

RUN mkdir /app

WORKDIR /app

ARG WEBUI_VERSION
RUN test -n "${WEBUI_VERSION}" && git reset --hard ${WEBUI_VERSION} || echo "Using provided webui source"

# Create virtualenv
RUN virtualenv /app/venv
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    . /app/venv/bin/activate && \
    pip3 install --upgrade pip setuptools wheel ninja && \
    pip3 install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install torchvision torchaudio sentence_transformers xformers

# Copy and install GPTQ-for-LLaMa
COPY --from=builder /build /app/repositories/GPTQ-for-LLaMa
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    . /app/venv/bin/activate && \
    pip3 install /app/repositories/GPTQ-for-LLaMa/*.whl

# Install main requirements
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    . /app/venv/bin/activate && \
    pip3 install -r requirements.txt

COPY . /app/

RUN cp /app/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda121.so /app/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so

# Install extension requirements
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    . /app/venv/bin/activate && \
    for ext in /app/extensions/*/requirements.txt; do \
    cd "$(dirname "$ext")"; \
    pip3 install -r requirements.txt; \
    done

ENV CLI_ARGS=""

EXPOSE ${CONTAINER_PORT:-7860} ${CONTAINER_API_PORT:-5000} ${CONTAINER_API_STREAM_PORT:-5005}
CMD . /app/venv/bin/activate && python3 server.py ${CLI_ARGS}
