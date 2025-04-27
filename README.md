# Text generation web UI

A Gradio web UI for Large Language Models.

Its goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

[Try the Deep Reason extension](https://oobabooga.gumroad.com/l/deep_reason)

|![Image1](https://github.com/oobabooga/screenshots/raw/main/AFTER-INSTRUCT.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/AFTER-CHAT.png) |
|:---:|:---:|
|![Image1](https://github.com/oobabooga/screenshots/raw/main/AFTER-DEFAULT.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/AFTER-PARAMETERS.png) |

## Features

- Supports multiple text generation backends in one UI/API, including [Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), and [ExLlamaV2](https://github.com/turboderp-org/exllamav2). [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is supported via its own [Dockerfile](https://github.com/oobabooga/text-generation-webui/blob/main/docker/TensorRT-LLM/Dockerfile), and the Transformers loader is compatible with libraries like [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [HQQ](https://github.com/mobiusml/hqq), and [AQLM](https://github.com/Vahe1994/AQLM), but they must be installed manually.
- OpenAI-compatible API with Chat and Completions endpoints – see [examples](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples).
- Automatic prompt formatting using Jinja2 templates.
- Three chat modes: `instruct`, `chat-instruct`, and `chat`, with automatic prompt templates in `chat-instruct`.
- "Past chats" menu to quickly switch between conversations.
- Free-form text generation in the Default/Notebook tabs without being limited to chat turns. You can send formatted conversations from the Chat tab to these.
- Multiple sampling parameters and generation options for sophisticated text generation control.
- Switch between different models easily in the UI without restarting.
- Simple LoRA fine-tuning tool.
- Requirements installed in a self-contained `installer_files` directory that doesn't interfere with the system environment.
- Extension support, with numerous built-in and user-contributed extensions available. See the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions) and [extensions directory](https://github.com/oobabooga/text-generation-webui-extensions) for details.
- Completely private: no telemetry, no tracking, no remote connections.

## How to install

#### Option 1: Portable builds

Compatible with GGUF (llama.cpp) models, just unzip and run, no installation. Available for Windows, Linux, and macOS.

Download from: https://github.com/oobabooga/text-generation-webui/releases

#### Option 2: One-click installer

1) Clone or [download the repository](https://github.com/oobabooga/text-generation-webui/archive/refs/heads/main.zip).
2) Run the script that matches your OS: `start_linux.sh`, `start_windows.bat`, `start_macos.sh`, or `start_wsl.bat`.
3) Select your GPU vendor when asked.
4) Once the installation ends, browse to `http://localhost:7860`.
5) Have fun!

To restart the web UI later, just run the same `start_` script. If you need to reinstall, delete the `installer_files` folder created during setup and run the script again.

You can use command-line flags, like `./start_linux.sh --help`, or add them to `user_data/CMD_FLAGS.txt` (such as `--api` to enable API use). To update the project, run `update_wizard_linux.sh`, `update_wizard_windows.bat`, `update_wizard_macos.sh`, or `update_wizard_wsl.bat`.

<details>
<summary>
Setup details and information about installing manually
</summary>

### One-click-installer

The script uses Miniconda to set up a Conda environment in the `installer_files` folder.

If you ever need to install something manually in the `installer_files` environment, you can launch an interactive shell using the cmd script: `cmd_linux.sh`, `cmd_windows.bat`, `cmd_macos.sh`, or `cmd_wsl.bat`.

* There is no need to run any of those scripts (`start_`, `update_wizard_`, or `cmd_`) as admin/root.
* To install the requirements for extensions, you can use the `extensions_reqs` script for your OS. At the end, this script will install the main requirements for the project to make sure that they take precedence in case of version conflicts.
* For additional instructions about AMD and WSL setup, consult [the documentation](https://github.com/oobabooga/text-generation-webui/wiki).
* For automated installation, you can use the `GPU_CHOICE`, `USE_CUDA118`, `LAUNCH_AFTER_INSTALL`, and `INSTALL_EXTENSIONS` environment variables. For instance: `GPU_CHOICE=A USE_CUDA118=FALSE LAUNCH_AFTER_INSTALL=FALSE INSTALL_EXTENSIONS=TRUE ./start_linux.sh`.

### Manual installation using Conda

Recommended if you have some experience with the command-line.

#### 0. Install Conda

https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands ([source](https://educe-ubc.github.io/conda.html)):

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```

#### 1. Create a new conda environment

```
conda create -n textgen python=3.11
conda activate textgen
```

#### 2. Install Pytorch

| System | GPU | Command |
|--------|---------|---------|
| Linux/WSL | NVIDIA | `pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124` |
| Linux/WSL | CPU only | `pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu` |
| Linux | AMD | `pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.1` |
| MacOS + MPS | Any | `pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0` |
| Windows | NVIDIA | `pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124` |
| Windows | CPU only | `pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0` |

The up-to-date commands can be found here: https://pytorch.org/get-started/locally/.

If you need `nvcc` to compile some library manually, you will additionally need to install this:

```
conda install -y -c "nvidia/label/cuda-12.4.1" cuda
```

#### 3. Install the web UI

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r <requirements file according to table below>
```

Requirements file to use:

| GPU | CPU | requirements file to use |
|--------|---------|---------|
| NVIDIA | has AVX2 | `requirements.txt` |
| NVIDIA | no AVX2 | `requirements_noavx2.txt` |
| AMD | has AVX2 | `requirements_amd.txt` |
| AMD | no AVX2 | `requirements_amd_noavx2.txt` |
| CPU only | has AVX2 | `requirements_cpu_only.txt` |
| CPU only | no AVX2 | `requirements_cpu_only_noavx2.txt` |
| Apple | Intel | `requirements_apple_intel.txt` |
| Apple | Apple Silicon | `requirements_apple_silicon.txt` |

### Start the web UI

```
conda activate textgen
cd text-generation-webui
python server.py
```

Then browse to

`http://localhost:7860/?__theme=dark`

##### Manual install

The `requirements*.txt` above contain various wheels precompiled through GitHub Actions. If you wish to compile things manually, or if you need to because no suitable wheels are available for your hardware, you can use `requirements_nowheels.txt` and then install your desired loaders manually.

### Alternative: Docker

```
For NVIDIA GPU:
ln -s docker/{nvidia/Dockerfile,nvidia/docker-compose.yml,.dockerignore} .
For AMD GPU: 
ln -s docker/{amd/Dockerfile,intel/docker-compose.yml,.dockerignore} .
For Intel GPU:
ln -s docker/{intel/Dockerfile,amd/docker-compose.yml,.dockerignore} .
For CPU only
ln -s docker/{cpu/Dockerfile,cpu/docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
#Create logs/cache dir : 
mkdir -p logs cache
# Edit .env and set: 
#   TORCH_CUDA_ARCH_LIST based on your GPU model
#   APP_RUNTIME_GID      your host user's group id (run `id -g` in a terminal)
#   BUILD_EXTENIONS      optionally add comma separated list of extensions to build
# Edit user_data/CMD_FLAGS.txt and add in it the options you want to execute (like --listen --cpu)
# 
docker compose up --build
```

* You need to have Docker Compose v2.17 or higher installed. See [this guide](https://github.com/oobabooga/text-generation-webui/wiki/09-%E2%80%90-Docker) for instructions.
* For additional docker files, check out [this repository](https://github.com/Atinoda/text-generation-webui-docker).

### Updating the requirements

From time to time, the `requirements*.txt` change. To update, use these commands:

```
conda activate textgen
cd text-generation-webui
pip install -r <requirements file that you have used> --upgrade
```
</details>

<details>
<summary>
List of command-line flags
</summary>

```txt
usage: server.py [-h] [--multi-user] [--character CHARACTER] [--model MODEL] [--lora LORA [LORA ...]] [--model-dir MODEL_DIR] [--lora-dir LORA_DIR] [--model-menu] [--settings SETTINGS]
                 [--extensions EXTENSIONS [EXTENSIONS ...]] [--verbose] [--idle-timeout IDLE_TIMEOUT] [--loader LOADER] [--cpu] [--cpu-memory CPU_MEMORY] [--disk] [--disk-cache-dir DISK_CACHE_DIR]
                 [--load-in-8bit] [--bf16] [--no-cache] [--trust-remote-code] [--force-safetensors] [--no_use_fast] [--use_flash_attention_2] [--use_eager_attention] [--torch-compile] [--load-in-4bit]
                 [--use_double_quant] [--compute_dtype COMPUTE_DTYPE] [--quant_type QUANT_TYPE] [--flash-attn] [--threads THREADS] [--threads-batch THREADS_BATCH] [--batch-size BATCH_SIZE] [--no-mmap]
                 [--mlock] [--n-gpu-layers N_GPU_LAYERS] [--tensor-split TENSOR_SPLIT] [--numa] [--no-kv-offload] [--row-split] [--extra-flags EXTRA_FLAGS] [--streaming-llm] [--ctx-size N]
                 [--model-draft MODEL_DRAFT] [--draft-max DRAFT_MAX] [--gpu-layers-draft GPU_LAYERS_DRAFT] [--device-draft DEVICE_DRAFT] [--ctx-size-draft CTX_SIZE_DRAFT] [--gpu-split GPU_SPLIT]
                 [--autosplit] [--cfg-cache] [--no_flash_attn] [--no_xformers] [--no_sdpa] [--num_experts_per_token N] [--enable_tp] [--hqq-backend HQQ_BACKEND] [--cpp-runner]
                 [--cache_type CACHE_TYPE] [--deepspeed] [--nvme-offload-dir NVME_OFFLOAD_DIR] [--local_rank LOCAL_RANK] [--alpha_value ALPHA_VALUE] [--rope_freq_base ROPE_FREQ_BASE]
                 [--compress_pos_emb COMPRESS_POS_EMB] [--listen] [--listen-port LISTEN_PORT] [--listen-host LISTEN_HOST] [--share] [--auto-launch] [--gradio-auth GRADIO_AUTH]
                 [--gradio-auth-path GRADIO_AUTH_PATH] [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE] [--subpath SUBPATH] [--old-colors] [--api] [--public-api]
                 [--public-api-id PUBLIC_API_ID] [--api-port API_PORT] [--api-key API_KEY] [--admin-key ADMIN_KEY] [--api-enable-ipv6] [--api-disable-ipv4] [--nowebui]

Text generation web UI

options:
  -h, --help                                show this help message and exit

Basic settings:
  --multi-user                              Multi-user mode. Chat histories are not saved or automatically loaded. Warning: this is likely not safe for sharing publicly.
  --character CHARACTER                     The name of the character to load in chat mode by default.
  --model MODEL                             Name of the model to load by default.
  --lora LORA [LORA ...]                    The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.
  --model-dir MODEL_DIR                     Path to directory with all the models.
  --lora-dir LORA_DIR                       Path to directory with all the loras.
  --model-menu                              Show a model menu in the terminal when the web UI is first launched.
  --settings SETTINGS                       Load the default interface settings from this yaml file. See user_data/settings-template.yaml for an example. If you create a file called
                                            user_data/settings.yaml, this file will be loaded by default without the need to use the --settings flag.
  --extensions EXTENSIONS [EXTENSIONS ...]  The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.
  --verbose                                 Print the prompts to the terminal.
  --idle-timeout IDLE_TIMEOUT               Unload model after this many minutes of inactivity. It will be automatically reloaded when you try to use it again.

Model loader:
  --loader LOADER                           Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, ExLlamav3_HF, ExLlamav2_HF, ExLlamav2, HQQ,
                                            TensorRT-LLM.

Transformers/Accelerate:
  --cpu                                     Use the CPU to generate text. Warning: Training on CPU is extremely slow.
  --cpu-memory CPU_MEMORY                   Maximum CPU memory in GiB. Use this for CPU offloading.
  --disk                                    If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.
  --disk-cache-dir DISK_CACHE_DIR           Directory to save the disk cache to. Defaults to "user_data/cache".
  --load-in-8bit                            Load the model with 8-bit precision (using bitsandbytes).
  --bf16                                    Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
  --no-cache                                Set use_cache to False while generating text. This reduces VRAM usage slightly, but it comes at a performance cost.
  --trust-remote-code                       Set trust_remote_code=True while loading the model. Necessary for some models.
  --force-safetensors                       Set use_safetensors=True while loading the model. This prevents arbitrary code execution.
  --no_use_fast                             Set use_fast=False while loading the tokenizer (it's True by default). Use this if you have any problems related to use_fast.
  --use_flash_attention_2                   Set use_flash_attention_2=True while loading the model.
  --use_eager_attention                     Set attn_implementation= eager while loading the model.
  --torch-compile                           Compile the model with torch.compile for improved performance.

bitsandbytes 4-bit:
  --load-in-4bit                            Load the model with 4-bit precision (using bitsandbytes).
  --use_double_quant                        use_double_quant for 4-bit.
  --compute_dtype COMPUTE_DTYPE             compute dtype for 4-bit. Valid options: bfloat16, float16, float32.
  --quant_type QUANT_TYPE                   quant_type for 4-bit. Valid options: nf4, fp4.

llama.cpp:
  --flash-attn                              Use flash-attention.
  --threads THREADS                         Number of threads to use.
  --threads-batch THREADS_BATCH             Number of threads to use for batches/prompt processing.
  --batch-size BATCH_SIZE                   Maximum number of prompt tokens to batch together when calling llama_eval.
  --no-mmap                                 Prevent mmap from being used.
  --mlock                                   Force the system to keep the model in RAM.
  --n-gpu-layers N_GPU_LAYERS               Number of layers to offload to the GPU.
  --tensor-split TENSOR_SPLIT               Split the model across multiple GPUs. Comma-separated list of proportions. Example: 60,40.
  --numa                                    Activate NUMA task allocation for llama.cpp.
  --no-kv-offload                           Do not offload the K, Q, V to the GPU. This saves VRAM but reduces the performance.
  --row-split                               Split the model by rows across GPUs. This may improve multi-gpu performance.
  --extra-flags EXTRA_FLAGS                 Extra flags to pass to llama-server. Format: "flag1=value1;flag2;flag3=value3". Example: "override-tensor=exps=CPU"
  --streaming-llm                           Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.

Context and cache management:
  --ctx-size N, --n_ctx N, --max_seq_len N  Context size in tokens.

Speculative decoding:
  --model-draft MODEL_DRAFT                 Path to the draft model for speculative decoding.
  --draft-max DRAFT_MAX                     Number of tokens to draft for speculative decoding.
  --gpu-layers-draft GPU_LAYERS_DRAFT       Number of layers to offload to the GPU for the draft model.
  --device-draft DEVICE_DRAFT               Comma-separated list of devices to use for offloading the draft model. Example: CUDA0,CUDA1
  --ctx-size-draft CTX_SIZE_DRAFT           Size of the prompt context for the draft model. If 0, uses the same as the main model.

ExLlamaV2:
  --gpu-split GPU_SPLIT                     Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: 20,7,7.
  --autosplit                               Autosplit the model tensors across the available GPUs. This causes --gpu-split to be ignored.
  --cfg-cache                               ExLlamav2_HF: Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader.
  --no_flash_attn                           Force flash-attention to not be used.
  --no_xformers                             Force xformers to not be used.
  --no_sdpa                                 Force Torch SDPA to not be used.
  --num_experts_per_token N                 Number of experts to use for generation. Applies to MoE models like Mixtral.
  --enable_tp                               Enable Tensor Parallelism (TP) in ExLlamaV2.

HQQ:
  --hqq-backend HQQ_BACKEND                 Backend for the HQQ loader. Valid options: PYTORCH, PYTORCH_COMPILE, ATEN.

TensorRT-LLM:
  --cpp-runner                              Use the ModelRunnerCpp runner, which is faster than the default ModelRunner but doesn't support streaming yet.

Cache:
  --cache_type CACHE_TYPE                   KV cache type; valid options: llama.cpp - fp16, q8_0, q4_0; ExLlamaV2 - fp16, fp8, q8, q6, q4.

DeepSpeed:
  --deepspeed                               Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.
  --nvme-offload-dir NVME_OFFLOAD_DIR       DeepSpeed: Directory to use for ZeRO-3 NVME offloading.
  --local_rank LOCAL_RANK                   DeepSpeed: Optional argument for distributed setups.

RoPE:
  --alpha_value ALPHA_VALUE                 Positional embeddings alpha factor for NTK RoPE scaling. Use either this or compress_pos_emb, not both.
  --rope_freq_base ROPE_FREQ_BASE           If greater than 0, will be used instead of alpha_value. Those two are related by rope_freq_base = 10000 * alpha_value ^ (64 / 63).
  --compress_pos_emb COMPRESS_POS_EMB       Positional embeddings compression factor. Should be set to (context length) / (model's original context length). Equal to 1/rope_freq_scale.

Gradio:
  --listen                                  Make the web UI reachable from your local network.
  --listen-port LISTEN_PORT                 The listening port that the server will use.
  --listen-host LISTEN_HOST                 The hostname that the server will use.
  --share                                   Create a public URL. This is useful for running the web UI on Google Colab or similar.
  --auto-launch                             Open the web UI in the default browser upon launch.
  --gradio-auth GRADIO_AUTH                 Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".
  --gradio-auth-path GRADIO_AUTH_PATH       Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.
  --ssl-keyfile SSL_KEYFILE                 The path to the SSL certificate key file.
  --ssl-certfile SSL_CERTFILE               The path to the SSL certificate cert file.
  --subpath SUBPATH                         Customize the subpath for gradio, use with reverse proxy
  --old-colors                              Use the legacy Gradio colors, before the December/2024 update.

API:
  --api                                     Enable the API extension.
  --public-api                              Create a public URL for the API using Cloudfare.
  --public-api-id PUBLIC_API_ID             Tunnel ID for named Cloudflare Tunnel. Use together with public-api option.
  --api-port API_PORT                       The listening port for the API.
  --api-key API_KEY                         API authentication key.
  --admin-key ADMIN_KEY                     API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key.
  --api-enable-ipv6                         Enable IPv6 for the API
  --api-disable-ipv4                        Disable IPv4 for the API
  --nowebui                                 Do not launch the Gradio UI. Useful for launching the API in standalone mode.
```

</details>

## Documentation

https://github.com/oobabooga/text-generation-webui/wiki

## Downloading models

Models should be placed in the folder `text-generation-webui/user_data/models`. They are usually downloaded from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

* GGUF models are a single file and should be placed directly into `user_data/models`. Example:

```
text-generation-webui
└── user_data
    └── models
        └── llama-2-13b-chat.Q4_K_M.gguf
```

* The remaining model types (like 16-bit Transformers models and EXL2 models) are made of several files and must be placed in a subfolder. Example:

```
text-generation-webui
└── user_data
    └── models
        └── lmsys_vicuna-33b-v1.3
            ├── config.json
            ├── generation_config.json
            ├── pytorch_model-00001-of-00007.bin
            ├── pytorch_model-00002-of-00007.bin
            ├── pytorch_model-00003-of-00007.bin
            ├── pytorch_model-00004-of-00007.bin
            ├── pytorch_model-00005-of-00007.bin
            ├── pytorch_model-00006-of-00007.bin
            ├── pytorch_model-00007-of-00007.bin
            ├── pytorch_model.bin.index.json
            ├── special_tokens_map.json
            ├── tokenizer_config.json
            └── tokenizer.model
```

In both cases, you can use the "Model" tab of the UI to download the model from Hugging Face automatically. It is also possible to download it via the command-line with:

```
python download-model.py organization/model
```

Run `python download-model.py --help` to see all the options.

## Google Colab notebook

https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb

## Community

https://www.reddit.com/r/Oobabooga/

## Acknowledgment

In August 2023, [Andreessen Horowitz](https://a16z.com/) (a16z) provided a generous grant to encourage and support my independent work on this project. I am **extremely** grateful for their trust and recognition.
