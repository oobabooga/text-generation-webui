<div align="center" markdown="1">
   <sup>Special thanks to:</sup>
   <br>
   <br>
   <a href="https://go.warp.dev/text-generation-webui">
      <img alt="Warp sponsorship" width="400" src="https://raw.githubusercontent.com/warpdotdev/brand-assets/refs/heads/main/Github/Sponsor/Warp-Github-LG-02.png">
   </a>

### [Warp, built for coding with multiple AI agents](https://go.warp.dev/text-generation-webui)
[Available for macOS, Linux, & Windows](https://go.warp.dev/text-generation-webui)<br>
</div>
<hr>

# Text Generation Web UI

A Gradio web UI for running Large Language Models locally. 100% private and offline. Supports text generation, vision, tool-calling, training, image generation, and more.

[Try the Deep Reason extension](https://oobabooga.gumroad.com/l/deep_reason)

|![Image1](https://github.com/oobabooga/screenshots/raw/main/INSTRUCT-3.5.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/CHAT-3.5.png) |
|:---:|:---:|
|![Image1](https://github.com/oobabooga/screenshots/raw/main/DEFAULT-3.5.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/PARAMETERS-3.5.png) |

## Features

- **Multiple backends**: [llama.cpp](https://github.com/ggerganov/llama.cpp), [Transformers](https://github.com/huggingface/transformers), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). Switch between backends and models without restarting.
- **File attachments**: Upload text files, PDF documents, and .docx documents to talk about their contents.
- **Vision (multimodal)**: Attach images to messages for visual understanding ([tutorial](https://github.com/oobabooga/text-generation-webui/wiki/Multimodal-Tutorial)).
- **Tool-calling**: Models can call custom functions during chat — web search, page fetching, math, and more. Each tool is a single `.py` file, easy to create and extend ([tutorial](https://github.com/oobabooga/text-generation-webui/wiki/Tool-Calling-Tutorial)).
- **OpenAI-compatible API**: Chat and Completions endpoints with tool-calling support. Use as a local drop-in replacement for the OpenAI API ([examples](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples)).
- **Training**: Fine-tune LoRAs on multi-turn chat or raw text datasets. Supports resuming interrupted runs ([tutorial](https://github.com/oobabooga/text-generation-webui/wiki/05-%E2%80%90-Training-Tab)).
- **Image generation**: A dedicated tab for `diffusers` models like **Z-Image-Turbo**. Features 4-bit/8-bit quantization and a persistent gallery with metadata ([tutorial](https://github.com/oobabooga/text-generation-webui/wiki/Image-Generation-Tutorial)).
- **Easy setup**: [Portable builds](https://github.com/oobabooga/text-generation-webui/releases) (zero setup, just unzip and run) for GGUF models on Windows/Linux/macOS, or a one-click installer for the full feature set.
- 100% offline and private, with zero telemetry, external resources, or remote update requests.
- `instruct` mode for instruction-following (like ChatGPT), and `chat-instruct`/`chat` modes for talking to custom characters. Prompts are automatically formatted with Jinja2 templates.
- Edit messages, navigate between message versions, and branch conversations at any point.
- Free-form text generation in the Notebook tab without being limited to chat turns.
- Multiple sampling parameters and generation options for sophisticated text generation control.
- Aesthetic UI with dark and light themes.
- Syntax highlighting for code blocks and LaTeX rendering for mathematical expressions.
- Extension support, with numerous built-in and user-contributed extensions available. See the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions) and [extensions directory](https://github.com/oobabooga/text-generation-webui-extensions) for details.

## How to install

#### ✅ Option 1: Portable builds (get started in 1 minute)

No installation needed – just download, unzip and run. All dependencies included.

Download from here: **https://github.com/oobabooga/text-generation-webui/releases**

- Builds are provided for Linux, Windows, and macOS, with options for CUDA, Vulkan, ROCm, and CPU-only.
- Compatible with GGUF (llama.cpp) models.

#### Option 2: Manual portable install with venv

Very fast setup that should work on any Python 3.9+:

```bash
# Clone repository
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies (choose appropriate file under requirements/portable for your hardware)
pip install -r requirements/portable/requirements.txt --upgrade

# Launch server (basic command)
python server.py --portable --api --auto-launch

# When done working, deactivate
deactivate
```

#### Option 3: One-click installer

For users who need additional backends (ExLlamaV3, Transformers), training, image generation, or extensions (TTS, voice input, translation, etc). Requires ~10GB disk space and downloads PyTorch.

1. Clone the repository, or [download its source code](https://github.com/oobabooga/text-generation-webui/archive/refs/heads/main.zip) and extract it.
2. Run the startup script for your OS: `start_windows.bat`, `start_linux.sh`, or `start_macos.sh`.
3. When prompted, select your GPU vendor.
4. After installation, open `http://127.0.0.1:7860` in your browser.

To restart the web UI later, run the same `start_` script.

You can pass command-line flags directly (e.g., `./start_linux.sh --help`), or add them to `user_data/CMD_FLAGS.txt` (e.g., `--api` to enable the API).

To update, run the update script for your OS: `update_wizard_windows.bat`, `update_wizard_linux.sh`, or `update_wizard_macos.sh`.

To reinstall with a fresh Python environment, delete the `installer_files` folder and run the `start_` script again.

<details>
<summary>
One-click installer details
</summary>

### One-click-installer

The script uses Miniforge to set up a Conda environment in the `installer_files` folder.

If you ever need to install something manually in the `installer_files` environment, you can launch an interactive shell using the cmd script: `cmd_linux.sh`, `cmd_windows.bat`, or `cmd_macos.sh`.

* There is no need to run any of those scripts (`start_`, `update_wizard_`, or `cmd_`) as admin/root.
* To install requirements for extensions, it is recommended to use the update wizard script with the "Install/update extensions requirements" option. At the end, this script will install the main requirements for the project to make sure that they take precedence in case of version conflicts.
* For automated installation, you can use the `GPU_CHOICE`, `LAUNCH_AFTER_INSTALL`, and `INSTALL_EXTENSIONS` environment variables. For instance: `GPU_CHOICE=A LAUNCH_AFTER_INSTALL=FALSE INSTALL_EXTENSIONS=TRUE ./start_linux.sh`.

</details>

<details>
<summary>
Manual full installation with conda or docker
</summary>

### Full installation with Conda

#### 0. Install Conda

https://github.com/conda-forge/miniforge

On Linux or WSL, Miniforge can be automatically installed with these two commands:

```
curl -sL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" > "Miniforge3.sh"
bash Miniforge3.sh
```

For other platforms, download from: https://github.com/conda-forge/miniforge/releases/latest

#### 1. Create a new conda environment

```
conda create -n textgen python=3.13
conda activate textgen
```

#### 2. Install Pytorch

| System | GPU | Command |
|--------|---------|---------|
| Linux/WSL | NVIDIA | `pip3 install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128` |
| Linux/WSL | CPU only | `pip3 install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu` |
| Linux | AMD | `pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp313-cp313-linux_x86_64.whl` |
| MacOS + MPS | Any | `pip3 install torch==2.9.1` |
| Windows | NVIDIA | `pip3 install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128` |
| Windows | CPU only | `pip3 install torch==2.9.1` |

The up-to-date commands can be found here: https://pytorch.org/get-started/locally/.

If you need `nvcc` to compile some library manually, you will additionally need to install this:

```
conda install -y -c "nvidia/label/cuda-12.8.1" cuda
```

#### 3. Install the web UI

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements/full/<requirements file according to table below>
```

Requirements file to use:

| GPU | requirements file to use |
|--------|---------|
| NVIDIA | `requirements.txt` |
| AMD | `requirements_amd.txt` |
| CPU only | `requirements_cpu_only.txt` |
| Apple Intel | `requirements_apple_intel.txt` |
| Apple Silicon | `requirements_apple_silicon.txt` |

### Start the web UI

```
conda activate textgen
cd text-generation-webui
python server.py
```

Then browse to

`http://127.0.0.1:7860`

#### Manual install

The `requirements*.txt` above contain various wheels precompiled through GitHub Actions. If you wish to compile things manually, or if you need to because no suitable wheels are available for your hardware, you can use `requirements_nowheels.txt` and then install your desired loaders manually.

### Alternative: Docker

```
For NVIDIA GPU:
ln -s docker/{nvidia/Dockerfile,nvidia/docker-compose.yml,.dockerignore} .
For AMD GPU:
ln -s docker/{amd/Dockerfile,amd/docker-compose.yml,.dockerignore} .
For Intel GPU:
ln -s docker/{intel/Dockerfile,intel/docker-compose.yml,.dockerignore} .
For CPU only
ln -s docker/{cpu/Dockerfile,cpu/docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
#Create logs/cache dir :
mkdir -p user_data/logs user_data/cache
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
usage: server.py [-h] [--user-data-dir USER_DATA_DIR] [--multi-user] [--model MODEL] [--lora LORA [LORA ...]] [--model-dir MODEL_DIR] [--lora-dir LORA_DIR] [--model-menu] [--settings SETTINGS]
                 [--extensions EXTENSIONS [EXTENSIONS ...]] [--verbose] [--idle-timeout IDLE_TIMEOUT] [--image-model IMAGE_MODEL] [--image-model-dir IMAGE_MODEL_DIR] [--image-dtype {bfloat16,float16}]
                 [--image-attn-backend {flash_attention_2,sdpa}] [--image-cpu-offload] [--image-compile] [--image-quant {none,bnb-8bit,bnb-4bit,torchao-int8wo,torchao-fp4,torchao-float8wo}]
                 [--loader LOADER] [--ctx-size N] [--cache-type N] [--model-draft MODEL_DRAFT] [--draft-max DRAFT_MAX] [--gpu-layers-draft GPU_LAYERS_DRAFT] [--device-draft DEVICE_DRAFT]
                 [--ctx-size-draft CTX_SIZE_DRAFT] [--spec-type {none,ngram-mod,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-cache}] [--spec-ngram-size-n SPEC_NGRAM_SIZE_N]
                 [--spec-ngram-size-m SPEC_NGRAM_SIZE_M] [--spec-ngram-min-hits SPEC_NGRAM_MIN_HITS] [--gpu-layers N] [--cpu-moe] [--mmproj MMPROJ] [--streaming-llm] [--tensor-split TENSOR_SPLIT]
                 [--row-split] [--no-mmap] [--mlock] [--no-kv-offload] [--batch-size BATCH_SIZE] [--ubatch-size UBATCH_SIZE] [--threads THREADS] [--threads-batch THREADS_BATCH] [--numa]
                 [--parallel PARALLEL] [--fit-target FIT_TARGET] [--extra-flags EXTRA_FLAGS] [--cpu] [--cpu-memory CPU_MEMORY] [--disk] [--disk-cache-dir DISK_CACHE_DIR] [--load-in-8bit] [--bf16]
                 [--no-cache] [--trust-remote-code] [--force-safetensors] [--no_use_fast] [--attn-implementation IMPLEMENTATION] [--load-in-4bit] [--use_double_quant] [--compute_dtype COMPUTE_DTYPE]
                 [--quant_type QUANT_TYPE] [--gpu-split GPU_SPLIT] [--enable-tp] [--tp-backend TP_BACKEND] [--cfg-cache] [--listen] [--listen-port LISTEN_PORT] [--listen-host LISTEN_HOST] [--share]
                 [--auto-launch] [--gradio-auth GRADIO_AUTH] [--gradio-auth-path GRADIO_AUTH_PATH] [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE] [--subpath SUBPATH] [--old-colors]
                 [--portable] [--api] [--public-api] [--public-api-id PUBLIC_API_ID] [--api-port API_PORT] [--api-key API_KEY] [--admin-key ADMIN_KEY] [--api-enable-ipv6] [--api-disable-ipv4]
                 [--nowebui] [--temperature N] [--dynatemp-low N] [--dynatemp-high N] [--dynatemp-exponent N] [--smoothing-factor N] [--smoothing-curve N] [--min-p N] [--top-p N] [--top-k N]
                 [--typical-p N] [--xtc-threshold N] [--xtc-probability N] [--epsilon-cutoff N] [--eta-cutoff N] [--tfs N] [--top-a N] [--top-n-sigma N] [--adaptive-target N] [--adaptive-decay N]
                 [--dry-multiplier N] [--dry-allowed-length N] [--dry-base N] [--repetition-penalty N] [--frequency-penalty N] [--presence-penalty N] [--encoder-repetition-penalty N]
                 [--no-repeat-ngram-size N] [--repetition-penalty-range N] [--penalty-alpha N] [--guidance-scale N] [--mirostat-mode N] [--mirostat-tau N] [--mirostat-eta N]
                 [--do-sample | --no-do-sample] [--dynamic-temperature | --no-dynamic-temperature] [--temperature-last | --no-temperature-last] [--sampler-priority N] [--dry-sequence-breakers N]
                 [--enable-thinking | --no-enable-thinking] [--reasoning-effort N] [--chat-template-file CHAT_TEMPLATE_FILE]

Text Generation Web UI

options:
  -h, --help                                           show this help message and exit

Basic settings:
  --user-data-dir USER_DATA_DIR                        Path to the user data directory. Default: auto-detected.
  --multi-user                                         Multi-user mode. Chat histories are not saved or automatically loaded. Best suited for small trusted teams.
  --model MODEL                                        Name of the model to load by default.
  --lora LORA [LORA ...]                               The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.
  --model-dir MODEL_DIR                                Path to directory with all the models.
  --lora-dir LORA_DIR                                  Path to directory with all the loras.
  --model-menu                                         Show a model menu in the terminal when the web UI is first launched.
  --settings SETTINGS                                  Load the default interface settings from this yaml file. See user_data/settings-template.yaml for an example. If you create a file called
                                                       user_data/settings.yaml, this file will be loaded by default without the need to use the --settings flag.
  --extensions EXTENSIONS [EXTENSIONS ...]             The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.
  --verbose                                            Print the prompts to the terminal.
  --idle-timeout IDLE_TIMEOUT                          Unload model after this many minutes of inactivity. It will be automatically reloaded when you try to use it again.

Image model:
  --image-model IMAGE_MODEL                            Name of the image model to select on startup (overrides saved setting).
  --image-model-dir IMAGE_MODEL_DIR                    Path to directory with all the image models.
  --image-dtype {bfloat16,float16}                     Data type for image model.
  --image-attn-backend {flash_attention_2,sdpa}        Attention backend for image model.
  --image-cpu-offload                                  Enable CPU offloading for image model.
  --image-compile                                      Compile the image model for faster inference.
  --image-quant {none,bnb-8bit,bnb-4bit,torchao-int8wo,torchao-fp4,torchao-float8wo}
                                                       Quantization method for image model.

Model loader:
  --loader LOADER                                      Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, ExLlamav3_HF, ExLlamav3, TensorRT-
                                                       LLM.

Context and cache:
  --ctx-size, --n_ctx, --max_seq_len N                 Context size in tokens. 0 = auto for llama.cpp (requires gpu-layers=-1), 8192 for other loaders.
  --cache-type, --cache_type N                         KV cache type; valid options: llama.cpp - fp16, q8_0, q4_0; ExLlamaV3 - fp16, q2 to q8 (can specify k_bits and v_bits separately, e.g. q4_q8).

Speculative decoding:
  --model-draft MODEL_DRAFT                            Path to the draft model for speculative decoding.
  --draft-max DRAFT_MAX                                Number of tokens to draft for speculative decoding.
  --gpu-layers-draft GPU_LAYERS_DRAFT                  Number of layers to offload to the GPU for the draft model.
  --device-draft DEVICE_DRAFT                          Comma-separated list of devices to use for offloading the draft model. Example: CUDA0,CUDA1
  --ctx-size-draft CTX_SIZE_DRAFT                      Size of the prompt context for the draft model. If 0, uses the same as the main model.
  --spec-type {none,ngram-mod,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-cache}
                                                       Draftless speculative decoding type. Recommended: ngram-mod.
  --spec-ngram-size-n SPEC_NGRAM_SIZE_N                N-gram lookup size for ngram speculative decoding.
  --spec-ngram-size-m SPEC_NGRAM_SIZE_M                Draft n-gram size for ngram speculative decoding.
  --spec-ngram-min-hits SPEC_NGRAM_MIN_HITS            Minimum n-gram hits for ngram-map speculative decoding.

llama.cpp:
  --gpu-layers, --n-gpu-layers N                       Number of layers to offload to the GPU. -1 = auto.
  --cpu-moe                                            Move the experts to the CPU (for MoE models).
  --mmproj MMPROJ                                      Path to the mmproj file for vision models.
  --streaming-llm                                      Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.
  --tensor-split TENSOR_SPLIT                          Split the model across multiple GPUs. Comma-separated list of proportions. Example: 60,40.
  --row-split                                          Split the model by rows across GPUs. This may improve multi-gpu performance.
  --no-mmap                                            Prevent mmap from being used.
  --mlock                                              Force the system to keep the model in RAM.
  --no-kv-offload                                      Do not offload the K, Q, V to the GPU. This saves VRAM but reduces the performance.
  --batch-size BATCH_SIZE                              Maximum number of prompt tokens to batch together when calling llama-server. This is the application level batch size.
  --ubatch-size UBATCH_SIZE                            Maximum number of prompt tokens to batch together when calling llama-server. This is the max physical batch size for computation (device level).
  --threads THREADS                                    Number of threads to use.
  --threads-batch THREADS_BATCH                        Number of threads to use for batches/prompt processing.
  --numa                                               Activate NUMA task allocation for llama.cpp.
  --parallel PARALLEL                                  Number of parallel request slots. The context size is divided equally among slots. For example, to have 4 slots with 8192 context each, set
                                                       ctx_size to 32768.
  --fit-target FIT_TARGET                              Target VRAM margin per device for auto GPU layers, comma-separated list of values in MiB. A single value is broadcast across all devices.
                                                       Default: 1024.
  --extra-flags EXTRA_FLAGS                            Extra flags to pass to llama-server. Format: "flag1=value1,flag2,flag3=value3". Example: "override-tensor=exps=CPU"

Transformers/Accelerate:
  --cpu                                                Use the CPU to generate text. Warning: Training on CPU is extremely slow.
  --cpu-memory CPU_MEMORY                              Maximum CPU memory in GiB. Use this for CPU offloading.
  --disk                                               If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.
  --disk-cache-dir DISK_CACHE_DIR                      Directory to save the disk cache to.
  --load-in-8bit                                       Load the model with 8-bit precision (using bitsandbytes).
  --bf16                                               Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
  --no-cache                                           Set use_cache to False while generating text. This reduces VRAM usage slightly, but it comes at a performance cost.
  --trust-remote-code                                  Set trust_remote_code=True while loading the model. Necessary for some models.
  --force-safetensors                                  Set use_safetensors=True while loading the model. This prevents arbitrary code execution.
  --no_use_fast                                        Set use_fast=False while loading the tokenizer (it's True by default). Use this if you have any problems related to use_fast.
  --attn-implementation IMPLEMENTATION                 Attention implementation. Valid options: sdpa, eager, flash_attention_2.

bitsandbytes 4-bit:
  --load-in-4bit                                       Load the model with 4-bit precision (using bitsandbytes).
  --use_double_quant                                   use_double_quant for 4-bit.
  --compute_dtype COMPUTE_DTYPE                        compute dtype for 4-bit. Valid options: bfloat16, float16, float32.
  --quant_type QUANT_TYPE                              quant_type for 4-bit. Valid options: nf4, fp4.

ExLlamaV3:
  --gpu-split GPU_SPLIT                                Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: 20,7,7.
  --enable-tp, --enable_tp                             Enable Tensor Parallelism (TP) to split the model across GPUs.
  --tp-backend TP_BACKEND                              The backend for tensor parallelism. Valid options: native, nccl. Default: native.
  --cfg-cache                                          Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader.

Gradio:
  --listen                                             Make the web UI reachable from your local network.
  --listen-port LISTEN_PORT                            The listening port that the server will use.
  --listen-host LISTEN_HOST                            The hostname that the server will use.
  --share                                              Create a public URL. This is useful for running the web UI on Google Colab or similar.
  --auto-launch                                        Open the web UI in the default browser upon launch.
  --gradio-auth GRADIO_AUTH                            Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".
  --gradio-auth-path GRADIO_AUTH_PATH                  Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.
  --ssl-keyfile SSL_KEYFILE                            The path to the SSL certificate key file.
  --ssl-certfile SSL_CERTFILE                          The path to the SSL certificate cert file.
  --subpath SUBPATH                                    Customize the subpath for gradio, use with reverse proxy
  --old-colors                                         Use the legacy Gradio colors, before the December/2024 update.
  --portable                                           Hide features not available in portable mode like training.

API:
  --api                                                Enable the API extension.
  --public-api                                         Create a public URL for the API using Cloudflare.
  --public-api-id PUBLIC_API_ID                        Tunnel ID for named Cloudflare Tunnel. Use together with public-api option.
  --api-port API_PORT                                  The listening port for the API.
  --api-key API_KEY                                    API authentication key.
  --admin-key ADMIN_KEY                                API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key.
  --api-enable-ipv6                                    Enable IPv6 for the API
  --api-disable-ipv4                                   Disable IPv4 for the API
  --nowebui                                            Do not launch the Gradio UI. Useful for launching the API in standalone mode.

API generation defaults:
  --temperature N                                      Temperature
  --dynatemp-low N                                     Dynamic temperature low
  --dynatemp-high N                                    Dynamic temperature high
  --dynatemp-exponent N                                Dynamic temperature exponent
  --smoothing-factor N                                 Smoothing factor
  --smoothing-curve N                                  Smoothing curve
  --min-p N                                            Min P
  --top-p N                                            Top P
  --top-k N                                            Top K
  --typical-p N                                        Typical P
  --xtc-threshold N                                    XTC threshold
  --xtc-probability N                                  XTC probability
  --epsilon-cutoff N                                   Epsilon cutoff
  --eta-cutoff N                                       Eta cutoff
  --tfs N                                              TFS
  --top-a N                                            Top A
  --top-n-sigma N                                      Top N Sigma
  --adaptive-target N                                  Adaptive target
  --adaptive-decay N                                   Adaptive decay
  --dry-multiplier N                                   DRY multiplier
  --dry-allowed-length N                               DRY allowed length
  --dry-base N                                         DRY base
  --repetition-penalty N                               Repetition penalty
  --frequency-penalty N                                Frequency penalty
  --presence-penalty N                                 Presence penalty
  --encoder-repetition-penalty N                       Encoder repetition penalty
  --no-repeat-ngram-size N                             No repeat ngram size
  --repetition-penalty-range N                         Repetition penalty range
  --penalty-alpha N                                    Penalty alpha
  --guidance-scale N                                   Guidance scale
  --mirostat-mode N                                    Mirostat mode
  --mirostat-tau N                                     Mirostat tau
  --mirostat-eta N                                     Mirostat eta
  --do-sample, --no-do-sample                          Do sample
  --dynamic-temperature, --no-dynamic-temperature      Dynamic temperature
  --temperature-last, --no-temperature-last            Temperature last
  --sampler-priority N                                 Sampler priority
  --dry-sequence-breakers N                            DRY sequence breakers
  --enable-thinking, --no-enable-thinking              Enable thinking
  --reasoning-effort N                                 Reasoning effort
  --chat-template-file CHAT_TEMPLATE_FILE              Path to a chat template file (.jinja, .jinja2, or .yaml) to use as the default instruction template for API requests. Overrides the model's
                                                       built-in template.
```

</details>

## Downloading models

1. Download a GGUF model file from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads&search=gguf).
2. Place it in the `user_data/models` folder.

That's it. The UI will detect it automatically.

To check what will fit your GPU, you can use the [VRAM Calculator](https://huggingface.co/spaces/oobabooga/accurate-gguf-vram-calculator).

<details>
<summary>Other model types (Transformers, EXL3)</summary>

Models that consist of multiple files (like 16-bit Transformers models and EXL3 models) should be placed in a subfolder inside `user_data/models`:

```
text-generation-webui
└── user_data
    └── models
        └── Qwen_Qwen3-8B
            ├── config.json
            ├── generation_config.json
            ├── model-00001-of-00004.safetensors
            ├── ...
            ├── tokenizer_config.json
            └── tokenizer.json
```

These formats require the one-click installer (not the portable build).
</details>

## Documentation

https://github.com/oobabooga/text-generation-webui/wiki

## Community

https://www.reddit.com/r/Oobabooga/

## Acknowledgments

- In August 2023, [Andreessen Horowitz](https://a16z.com/) (a16z) provided a generous grant to encourage and support my independent work on this project. I am **extremely** grateful for their trust and recognition.
- This project was inspired by [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and wouldn't exist without it.
