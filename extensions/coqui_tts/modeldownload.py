import os
from pathlib import Path
import requests
from tqdm import tqdm
import importlib.metadata as metadata  # Use importlib.metadata
from packaging import version

def create_directory_if_not_exists(directory):
    if not directory.exists():
        directory.mkdir(parents=True)

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

def check_tts_version():
    try:
        tts_version = metadata.version("tts")
        print(f"[COQUI TTS] STARTUP: TTS version: {tts_version}")

        if version.parse(tts_version) < version.parse("0.21.1"):
            print("[COQUI TTS] STARTUP: \033[91mTTS version is too old. Please upgrade to version 0.21.1 or later.\033[0m")
            print("[COQUI TTS] STARTUP: \033[91mpip install --upgrade tts\033[0m")
        else:
            print("[COQUI TTS] STARTUP: TTS version is up to date.")
    except metadata.PackageNotFoundError:
        print("[COQUI TTS] STARTUP: TTS is not installed.")

# Use this_dir in the downloader script
this_dir = Path(__file__).parent.resolve()

# Define paths
base_path = this_dir / 'models'
model_path = base_path / 'xttsv2_2.0.2'

# Check and create directories
create_directory_if_not_exists(base_path)
create_directory_if_not_exists(model_path)

# Define files and their corresponding URLs
files_to_download = {
    'LICENSE.txt': 'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true',
    'README.md': 'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true',
    'config.json': 'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true',
    'model.pth': 'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true',
    'vocab.json': 'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true',
}

# Download files if they don't exist
print("[COQUI TTS] STARTUP: Checking Model is Downloaded.")
for filename, url in files_to_download.items():
    destination = model_path / filename
    if not destination.exists():
        print(f"[COQUI TTS] STARTUP: Downloading {filename}...")
        download_file(url, destination)

check_tts_version()
