import subprocess
import os
import sys
import importlib.util
import platform

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")


def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None  


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def run_pip(command, desc=None, live=default_command_live):
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def commit_hash():
    try:
        return subprocess.check_output([git, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"


def git_tag():
    try:
        return subprocess.check_output([git, "describe", "--tags"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"



def prepare_packages():
    
    peft_url = "git+https://github.com/huggingface/peft@3714aa2fff158fdfa637b2b65952580801d890b2"
    transformers_url = "git+https://github.com/huggingface/transformers@e45e756d22206ca8fa9fb057c8c3d8fa79bf81c6"
    accelerate_url = "git+https://github.com/huggingface/accelerate@0226f750257b3bf2cadc4f189f9eef0c764a0467"
    bitsandbytes_no_win = "bitsandbytes==0.39.0"
    bitsandbytes_win = "https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.39.0-py3-none-any.whl"
    llama_cpp_python_no_win = "llama-cpp-python==0.1.56"
    llama_cpp_python_win = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.1.56/llama_cpp_python-0.1.56-cp310-cp310-win_amd64.whl"
    AutoGPTQ_win = "https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.2.0/auto_gptq-0.2.0+cu117-cp310-cp310-win_amd64.whl"
    AutoGPTQ_Linux = "https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.2.0/auto_gptq-0.2.0+cu117-cp310-cp310-linux_x86_64.whl"
    
    print("Installing packages...")
    
    if not is_installed("colorma"):
        run_pip("install colorama", desc="colorama")
    
    if not is_installed("datasets"):
        run_pip("install datasets", desc="datasets")
        
    if not is_installed("einops"):
        run_pip("install einops", desc="einops")
    
    if not is_installed("flexgen"):
        run_pip("install flexgen==0.1.7", desc="flexgen")
        
    if not is_installed("gradio_client"):
        run_pip("install gradio_client==0.2.5", desc="gradio_client")
        
    if not is_installed("gradio"):
        run_pip("install gradio==3.31.0", desc="gradio")
        
    if not is_installed("markdown"):
        run_pip("install markdown", desc="markdown")
        
    if not is_installed("numpy"):
        run_pip("install numpy", desc="numpy")
        
    if not is_installed("pandas"):
        run_pip("install pandas", desc="pandas")
        
    if not is_installed("Pillow"):
        run_pip("install Pillow>=9.5.0", desc="Pillow")
        
    if not is_installed("pyyaml"):
        run_pip("install pyyaml", desc="pyyaml")
        
    if not is_installed("requests"):
        run_pip("install requests", desc="requests")
        
    if not is_installed("safetensors"):
        run_pip("install safetensors==0.3.1", desc="safetensors")
        
    if not is_installed("sentencepiece"):
        run_pip("install sentencepiece", desc="sentencepiece")
        
    if not is_installed("tqdm"):
        run_pip("install tqdm", desc="tqdm")
        
    if not is_installed("scipy"):
        run_pip("install scipy", desc="scipy")
        
    if not is_installed("peft"):
        run_pip(f"install {peft_url}", desc="peft")
        
    if not is_installed("transformers"):
        run_pip(f"install {transformers_url}", desc="transformers")
        
    if not is_installed("accelerate"):
        run_pip(f"install {accelerate_url}", desc="accelerate")
        
    if not is_installed("bitsandbytes"):
        if platform.system() == "Windows":
            run_pip(f"install {bitsandbytes_win}", desc="bitsandbytes")
        else:
            run_pip(f"install {bitsandbytes_no_win}", desc="bitsandbytes")
    
    if not is_installed("llama-cpp-python"):
        if platform.system() == "Windows":
            run_pip(f"install {llama_cpp_python_win}", desc="llama-cpp-python")
        else:
            run_pip(f"install {llama_cpp_python_no_win}", desc="llama-cpp-python")
    
    if not is_installed("auto-gptq"):
        if platform.system() == "Windows":
            run_pip(f"install {AutoGPTQ_win}", desc="auto-gptq")
        elif platform.system() == "Linux":
            run_pip(f"install {AutoGPTQ_Linux}", desc="auto-gptq")
        else:
            print("auto-gptq is not supported on your platform.")
            

def check_environment():
    commit = commit_hash()
    tag = git_tag()

    print(f"Python {sys.version}")
    print(f"Version: {tag}")
    print(f"Commit hash: {commit}")
    