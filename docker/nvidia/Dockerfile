# default UID is 6972
# override the runtime user via docker-compose.override.yml or mount dirs with root group access 

# BUILDER
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as builder
WORKDIR /builder
ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX}"
ARG BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-}"
ARG APP_UID="${APP_UID:-6972}"
ARG APP_GID="${APP_GID:-6972}"
# create / update build env
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw \
    apt update && \
    apt install --no-install-recommends -y git vim build-essential python3-dev pip && \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    pip3 install --global --upgrade pip wheel setuptools && \
    # make shared builder & runtime app user
    addgroup --gid $APP_GID app_grp && \
    useradd -m -u $APP_UID --gid app_grp app
USER app:app_grp
# build wheels for runtime
WORKDIR /home/app/build
COPY --chown=app:app_grp requirements.txt /home/app/build
COPY --chown=app:app_grp extensions /home/app/build/extensions
RUN --mount=type=cache,target=/root/.cache/pip,rw \
    # clone and build gptq-for-llama
    git clone --depth=1 https://github.com/oobabooga/GPTQ-for-LLaMa /home/app/gptq-for-llama && \
    # TODO: remove fixes to gptq-for-llama requirements conflicts
    sed -i -e 's/safetensors==0\.3\.0$/safetensors>=0.3.0,<=2.0/' -e 's/transformers==4\.28\.0$/transformers==4.35.*/' -e 's/accelerate==0\.18\.0$/accelerate==0.24.*/' /home/app/gptq-for-llama/requirements.txt && \
    # build all requirements files as wheel dists
    pip3 wheel -w wheels -r requirements.txt -r /home/app/gptq-for-llama/requirements.txt `echo "$BUILD_EXTENSIONS" | sed -r 's/([^,]+)\s*,?\s*/ -r \/home\/app\/build\/extensions\/\1\/requirements.txt/g'`
    # drop wheel and setuptools .whl to avoid install issues
RUN rm wheels/setuptools*.whl

# RUNTIME
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6}"
ARG APP_UID="${APP_UID:-6972}"
ARG APP_GID="${APP_GID:-6972}"
ENV CLI_ARGS=""
# create / update runtime env
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw \
    apt update && \
    apt install --no-install-recommends -y git python3 pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --global --no-cache --upgrade pip wheel setuptools && \
    # make shared builder & runtime app user
    addgroup --gid $APP_GID app_grp && \
    useradd -m -u $APP_UID --gid app_grp app
    # make install dirs accessible to root group
    # chmod -R g+rwX /usr/local/ && \
    # chmod -R g+rwX /usr/include/ && \
    # chmod -R g+rwX /home/app
USER app:app_grp
# install locally built wheels for app
WORKDIR /home/app/wheels
COPY --from=builder /home/app/build/wheels /home/app/wheels
COPY --chown=app:app_grp . /home/app/text-generation-webui
RUN umask 0002 && \
    chmod g+rwX /home/app/text-generation-webui && \
    pip3 install --global --no-build-isolation --no-cache --no-index ./*.whl && \
    rm -r /home/app/wheels
WORKDIR /home/app/text-generation-webui
EXPOSE ${CONTAINER_PORT:-7860} ${CONTAINER_API_PORT:-5000} ${CONTAINER_API_STREAM_PORT:-5005}
# set umask to ensure group read / write at runtime
CMD umask 0002 && export HOME=/home/app && python3 server.py ${CLI_ARGS}
