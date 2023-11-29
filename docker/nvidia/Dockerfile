# default UID is 6972
# override the runtime user via docker-compose.override.yml or mount dirs with root group access 

# BUILDER
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder
WORKDIR /builder
ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX}"
# create / update build env
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw \
    apt update && \
    apt install --no-install-recommends -y git vim build-essential python3-dev pip && \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    pip3 install --global --upgrade pip wheel setuptools && \
    # make shared builder & runtime app user
    useradd -m -u 6972 -g 0 app
USER app:0
# build wheels for runtime
WORKDIR /home/app/build
COPY --chown=6972:0 requirements.txt /home/app/build
COPY --chown=6972:0 extensions /home/app/build/extensions
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    # clone and build gptq-for-llama
    git clone --depth=1 https://github.com/oobabooga/GPTQ-for-LLaMa /home/app/gptq-for-llama && \
    # build all requirements files as wheel dists
    # strictly speaking we should build for oobabooga/GPTQ-for-LLaMa here too, but it's requirements are out of date (subset of other deps) and need looser constraints
    pip3 wheel -w wheels -r requirements.txt `ls extensions/*/requirements.txt | sed -e 's/^/-r /'`
    # drop wheel and setuptools .whl to avoid install issues
RUN rm wheels/setuptools*.whl

# RUNTIME
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6}"
ENV CLI_ARGS=""
# create / update runtime env
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw \
    apt update && \
    apt install --no-install-recommends -y git python3 pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --global --no-cache --upgrade pip wheel setuptools && \
    # make shared builder & runtime app user
    useradd -m -u 6972 -g 0 app && \
    # make install dirs accessible to root group
    chmod -R g+rwX /usr/local/ && \
    chmod -R g+rwX /usr/include/ && \
    chmod -R g+rwX /home/app
USER app:0
# install locally built wheels for app
WORKDIR /home/app/wheels
COPY --from=builder /home/app/build/wheels /home/app/wheels
COPY --chown=6972:0 . /home/app/text-generation-webui
RUN umask 0002 && \
    chmod g+rwX /home/app/text-generation-webui && \
    pip3 install --global --no-build-isolation --no-cache --no-index ./*.whl && \
    rm -r /home/app/wheels
WORKDIR /home/app/text-generation-webui
EXPOSE ${CONTAINER_PORT:-7860} ${CONTAINER_API_PORT:-5000} ${CONTAINER_API_STREAM_PORT:-5005}
# set umask to ensure group read / write at runtime
CMD umask 0002 && export HOME=/home/app && python3 server.py ${CLI_ARGS}
