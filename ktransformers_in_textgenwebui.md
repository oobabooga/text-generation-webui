# I was able to run Qwen3-4B-Instruct-2507-FP8 on my laptop with 8gb vram and 64 GB Ram with ktransformers as loader in textgenwebui over the gui, with context lenght 4096, flash_attention_2, cache fp8, no cpu offload besides a bug appearing otherwise
# For bigger models the hybrid offloading did not work, but that seems to be a problem of this version of textgenwebui, since it happened with other loaders too if i try to offload to cpu the hybrid ffload failed was only gpu or cpu
# In Sum: ktransformers brings now the possibility to run small models better on normal consumer hardware with textgenwebui. If textgenwebui team fix the bug with hybrid offloading to CPU and disk the Big models of DeepSeek, Gwen and others are reachable for 5k- 10k cost lokal server builds. That means breakthrough for textgenwebui and local AI developers and a whole new user base for loacal AI as a community. 

# 1. Priority is ktransformers must be installed in the same environment ass the one click installation of textgenwebui to be found so open a terminal
cd ~/text-generation-webui
./cmd_linux.sh -c 'echo "CONDA_PREFIX=$CONDA_PREFIX"; which python'
# Erwartet:
# CONDA_PREFIX=/home/<USER>/text-generation-webui/installer_files/env
# /home/<USER>/text-generation-webui/installer_files/env/bin/python

# Du bist jetzt in der "installer_files" Conda-Umgebung der WebUI
python -c "import sys; print(sys.executable)"


# 2. perhaps some tools are needed before installing
./cmd_linux.sh
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build patchelf
# numpy i needed, if some conflicts arise modern llm can help you to assist with solving version conflicts
pip install -U packaging ninja cpufeature numpy
# minimaler CUDA-Compiler in *diese* Conda-Env conda 12.4.1 or higher:
conda install -y -c nvidia/label/cuda-12.4.1 cuda-nvcc
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
nvcc -V


# 3. Do not install Ktransformers with pip, it has too old versions, use git instead for new version with HTTP/1.1
# im WebUI-Conda-Shell:
mkdir -p repositories && cd repositories
git -c http.version=HTTP/1.1 clone --depth 1 --recurse-submodules \
  https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git -c http.version=HTTP/1.1 submodule update --init --recursive --depth 1 --recommend-shallow --jobs 1


# build without pip
python setup.py build_ext --inplace
python - <<'PY'
import site, os
repo = os.path.abspath(".")
cands = site.getsitepackages() or [site.getusersitepackages()]
pth = os.path.join(cands[0], "ktransformers_local.pth")
with open(pth, "w") as f: f.write(repo + "\n")
print("Wrote:", pth, "->", repo)
PY


# 4. sanity check out of the one click environment
cd ~/text-generation-webui
./cmd_linux.sh -c 'python - <<PY
import sys, ktransformers
print("python:", sys.executable)
print("ktransformers:", getattr(ktransformers,"__version__","git"), "from:", ktransformers.__file__)
PY'
# should show: ~/text-generation-webui/repositories/ktransformers/...


---------------------------------------------------------------------------------------------------------


# after installation you must modify some config files in textgenwebui
# Go into textgenwebui folder -> modules 

# open loaders.pyin folder modules
# under loaders_and_params =  after transformers fill in ktransformers
    ],
    'ktransformers': [
        'ctx_size',
        'gpu_split',
        'cache_type',
        'cpu',            # CPU-Offload (HF accelerate)
        'disk',           # Disk-Offload (HF accelerate)
        'cpu_memory',     # z.B. "48GiB" (String)
        'quant_type',     # falls du 4/8-bit via bitsandbytes/awq testen willst
        'compute_dtype',  # bf16/fp16 perhaps torch_dtype would be better here ???
        'attn_implementation',  # sdpa/flash_attention_2 (je nach Build)
    ],

# under loaders_samplers = after Transformers fill in ktransformers
    'ktransformers': {
        'temperature',
        'top_p',
        'top_k',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'seed',
    },


# open models.py in folder modules
# the def load_model(model_name, loader=None) we fill in ktransformers after that the def looks like this


def load_model(model_name, loader=None):
    logger.info(f"Loading \"{model_name}\"")
    t0 = time.time()

    shared.is_seq2seq = False
    shared.model_name = model_name
    load_func_map = {
        'llama.cpp': llama_cpp_server_loader,
        'Transformers': transformers_loader,
        'ExLlamav3_HF': ExLlamav3_HF_loader,
        'ExLlamav3': ExLlamav3_loader,
        'ExLlamav2_HF': ExLlamav2_HF_loader,
        'ExLlamav2': ExLlamav2_loader,
        'TensorRT-LLM': TensorRT_LLM_loader,
        'ktransformers': ktransformers_loader,
    }



#  before the def unload_model(keep_model_name=False) fill between two spaces


def ktransformers_loader(model_name):
    try:
        import ktransformers  # aktiviert die Patches / Beschleuniger
    except ModuleNotFoundError as e:
        from modules.logging_colors import logger
        logger.error("KTransformers ist nicht installiert: pip install ktransformers")
        raise 
    from modules.transformers_loader import load_model_HF
    return load_model_HF(model_name)


