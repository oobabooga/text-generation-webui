Docker Compose is a way of installing and launching the web UI in an isolated Ubuntu image using only a few commands.

## Prerequisites

You need Docker Compose v2.17 or higher:

```
~$ docker compose version
Docker Compose version v2.21.0
```

Installation instructions: https://docs.docker.com/engine/install/

For NVIDIA GPUs, you also need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Quick start

There are four Docker variants available under `docker/`:

| Directory | GPU | Notes |
|-----------|-----|-------|
| `docker/nvidia` | NVIDIA | Requires NVIDIA Container Toolkit |
| `docker/amd` | AMD | Requires ROCm-compatible GPU |
| `docker/intel` | Intel Arc | Beta support |
| `docker/cpu` | None | CPU-only inference |

To launch (using NVIDIA as an example):

```bash
cd text-generation-webui/docker/nvidia
cp ../.env.example .env
# Optionally edit .env to customize ports, TORCH_CUDA_ARCH_LIST, etc.
docker compose up --build
```

The web UI will be available at `http://localhost:7860`.

## User data

Create a `user_data/` directory next to the `docker-compose.yml` to persist your models, characters, presets, and settings between container rebuilds:

```bash
mkdir -p user_data
```

This directory is mounted into the container at runtime. You can place a `CMD_FLAGS.txt` inside it to pass persistent flags to the web UI (e.g., `--api`).

Models can be downloaded through the web UI's “Model” tab once it's running, and they will be saved to `user_data/models/`.

## Dedicated docker repository

An external repository maintains a docker wrapper for this project as well as several pre-configured 'one-click' `docker compose` variants. It can be found at: [Atinoda/text-generation-webui-docker](https://github.com/Atinoda/text-generation-webui-docker).
