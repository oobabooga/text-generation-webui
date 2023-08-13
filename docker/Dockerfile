FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder

RUN apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv libmpich-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone https://github.com/oobabooga/GPTQ-for-LLaMa . && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch torchvision torchaudio deepspeed mpi4py mpi4py-mpich && \
    pip3 install -r requirements.txt

ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX}"
RUN . venv/bin/activate && python3 setup_cuda.py bdist_wheel -d .

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Docker image for GPTQ-for-LLaMa and Text Generation WebUI"

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-dev libportaudio2 libasound-dev git python3 python3-pip \
    make g++ libmpich-dev && \
    rm -rf /var/lib/apt/lists/* && \
    --mount=type=cache,target=/root/.cache/pip pip3 install virtualenv && \
    mkdir /app

WORKDIR /app

ARG WEBUI_VERSION
RUN test -n "${WEBUI_VERSION}" && git reset --hard ${WEBUI_VERSION} || echo "Using provided webui source"

RUN virtualenv venv && \
    . venv/bin/activate && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch torchvision torchaudio deepspeed mpi4py mpi4py-mpich zarr sentence_transformers xformers

COPY --from=builder /build /app/repositories/GPTQ-for-LLaMa
RUN . venv/bin/activate && \
    pip3 install repositories/GPTQ-for-LLaMa/*.whl

COPY extensions/*/requirements.txt /app/extensions/
RUN --mount=type=cache,target=/root/.cache/pip . venv/bin/activate && \
    cd extensions && \
    for req in *.txt; do pip3 install -r $req; done

COPY requirements.txt requirements.txt
RUN . venv/bin/activate && pip3 install -r requirements.txt

COPY . /app/
RUN cp venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so

ENV CLI_ARGS=""
CMD . venv/bin/activate && python3 server.py ${CLI_ARGS}
