# BUILDER
FROM fedora:latest
WORKDIR /builder
ARG TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX}"
ARG BUILD_EXTENSIONS="${BUILD_EXTENSIONS:-}"
ARG APP_UID="${APP_UID:-1000}"
ARG APP_GID="${APP_GID:-1000}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw \
    dnf -y update && \
    dnf -y install git vim python3-devel make automake gcc gcc-c++ kernel-devel pip && \
    dnf clean all
WORKDIR /home/app/
RUN git clone https://github.com/oobabooga/text-generation-webui.git 
WORKDIR /home/app/text-generation-webui
RUN GPU_CHOICE=A LAUNCH_AFTER_INSTALL=FALSE INSTALL_EXTENSIONS=TRUE ./start_linux.sh --verbose
COPY /user_data/CMD_FLAGS.txt /home/app/text-generation-webui/user_data
EXPOSE ${CONTAINER_PORT:-7860} ${CONTAINER_API_PORT:-5000} ${CONTAINER_API_STREAM_PORT:-5005}
WORKDIR /home/app/text-generation-webui
# set umask to ensure group read / write at runtime
CMD umask 0002 && export HOME=/home/app/text-generation-webui && ./start_linux.sh --listen
