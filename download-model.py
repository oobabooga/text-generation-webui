'''
Downloads models from Hugging Face to user_data/models/username_modelname.

Example:
python download-model.py facebook/opt-1.3b

'''

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
from multiprocessing import Array
from pathlib import Path
from time import sleep

import requests
import tqdm
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from tqdm.contrib.concurrent import thread_map

base = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"


class ModelDownloader:
    def __init__(self, max_retries=7):
        self.max_retries = max_retries
        self.session = self.get_session()
        self._progress_bar_slots = None
        self.progress_queue = None

    def get_session(self):
        session = requests.Session()
        if self.max_retries:
            session.mount('https://cdn-lfs.huggingface.co', HTTPAdapter(max_retries=self.max_retries))
            session.mount('https://huggingface.co', HTTPAdapter(max_retries=self.max_retries))

        if os.getenv('HF_USER') is not None and os.getenv('HF_PASS') is not None:
            session.auth = (os.getenv('HF_USER'), os.getenv('HF_PASS'))

        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            token = os.getenv("HF_TOKEN")

        if token is not None:
            session.headers = {'authorization': f'Bearer {token}'}

        return session

    def sanitize_model_and_branch_names(self, model, branch):
        if model[-1] == '/':
            model = model[:-1]

        if model.startswith(base + '/'):
            model = model[len(base) + 1:]

        model_parts = model.split(":")
        model = model_parts[0] if len(model_parts) > 0 else model
        branch = model_parts[1] if len(model_parts) > 1 else branch

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

        return model, branch

    def get_download_links_from_huggingface(self, model, branch, text_only=False, specific_file=None, exclude_pattern=None):
        session = self.session
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
        file_sizes = []
        classifications = []
        has_pytorch = False
        has_pt = False
        has_gguf = False
        has_safetensors = False
        is_lora = False
        while True:
            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            r = session.get(url, timeout=10)
            r.raise_for_status()
            content = r.content

            dict = json.loads(content)
            if len(dict) == 0:
                break

            for i in range(len(dict)):
                fname = dict[i]['path']
                if specific_file not in [None, ''] and fname != specific_file:
                    continue

                # Exclude files matching the exclude pattern
                if exclude_pattern is not None and re.match(exclude_pattern, fname):
                    continue

                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                is_safetensors = re.match(r".*\.safetensors", fname)
                is_pt = re.match(r".*\.pt", fname)
                is_gguf = re.match(r".*\.gguf", fname)
                is_tiktoken = re.match(r".*\.tiktoken", fname)
                is_tokenizer = re.match(r"(tokenizer|ice|spiece).*\.model", fname) or is_tiktoken
                is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any((is_pytorch, is_safetensors, is_pt, is_gguf, is_tokenizer, is_text)):
                    file_size = 0
                    if 'lfs' in dict[i]:
                        sha256.append([fname, dict[i]['lfs']['oid']])
                        file_size = dict[i]['lfs'].get('size', 0)
                    elif 'size' in dict[i]:
                        file_size = dict[i]['size']

                    file_sizes.append(file_size)

                    if is_text:
                        links.append(f"{base}/{model}/resolve/{branch}/{fname}")
                        classifications.append('text')
                        continue

                    if not text_only:
                        links.append(f"{base}/{model}/resolve/{branch}/{fname}")
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append('safetensors')
                        elif is_pytorch:
                            has_pytorch = True
                            classifications.append('pytorch')
                        elif is_pt:
                            has_pt = True
                            classifications.append('pt')
                        elif is_gguf:
                            has_gguf = True
                            classifications.append('gguf')

            cursor = base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode()) + b':50'
            cursor = base64.b64encode(cursor)
            cursor = cursor.replace(b'=', b'%3D')

        # If both pytorch and safetensors are available, download safetensors only
        # Also if GGUF and safetensors are available, download only safetensors
        if (has_pytorch or has_pt or has_gguf) and has_safetensors:
            has_gguf = False
            for i in range(len(classifications) - 1, -1, -1):
                if classifications[i] in ['pytorch', 'pt', 'gguf']:
                    links.pop(i)
                    file_sizes.pop(i)

        # For GGUF, try to download only the Q4_K_M if no specific file is specified.
        if has_gguf and specific_file is None:
            has_q4km = False
            for i in range(len(classifications) - 1, -1, -1):
                if 'q4_k_m' in links[i].lower():
                    has_q4km = True

            if has_q4km:
                for i in range(len(classifications) - 1, -1, -1):
                    if 'q4_k_m' not in links[i].lower():
                        links.pop(i)
                        file_sizes.pop(i)
            else:
                for i in range(len(classifications) - 1, -1, -1):
                    if links[i].lower().endswith('.gguf'):
                        links.pop(i)
                        file_sizes.pop(i)

        is_llamacpp = has_gguf and specific_file is not None
        return links, sha256, is_lora, is_llamacpp, file_sizes

    def get_output_folder(self, model, branch, is_lora, is_llamacpp=False, model_dir=None):
        if model_dir:
            base_folder = model_dir
        else:
            base_folder = 'user_data/models' if not is_lora else 'user_data/loras'

        # If the model is of type GGUF, save directly in the base_folder
        if is_llamacpp:
            return Path(base_folder)

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'

        output_folder = Path(base_folder) / output_folder
        return output_folder

    @property
    def progress_bar_slots(self):
        if self._progress_bar_slots is None:
            raise RuntimeError("Progress bar slots not initialized. Start download threads first.")

        return self._progress_bar_slots

    def initialize_progress_bar_slots(self, num_threads):
        self._progress_bar_slots = Array("B", [0] * num_threads)

    def get_progress_bar_position(self):
        with self.progress_bar_slots.get_lock():
            for i in range(len(self.progress_bar_slots)):
                if self.progress_bar_slots[i] == 0:
                    self.progress_bar_slots[i] = 1
                    return i

        return 0  # fallback

    def release_progress_bar_position(self, slot):
        with self.progress_bar_slots.get_lock():
            self.progress_bar_slots[slot] = 0

    def get_single_file(self, url, output_folder, start_from_scratch=False):
        filename = Path(url.rsplit('/', 1)[1])
        output_path = output_folder / filename
        progress_bar_position = self.get_progress_bar_position()

        max_retries = self.max_retries
        attempt = 0
        file_downloaded_count_for_progress = 0

        try:
            while attempt < max_retries:
                attempt += 1
                session = self.session
                headers = {}
                mode = 'wb'
                current_file_size_on_disk = 0

                try:
                    if output_path.exists() and not start_from_scratch:
                        current_file_size_on_disk = output_path.stat().st_size
                        r_head = session.head(url, timeout=20)
                        r_head.raise_for_status()
                        total_size = int(r_head.headers.get('content-length', 0))

                        if current_file_size_on_disk >= total_size and total_size > 0:
                            if self.progress_queue is not None and total_size > 0:
                                self.progress_queue.put((1.0, str(filename)))
                            return

                        headers = {'Range': f'bytes={current_file_size_on_disk}-'}
                        mode = 'ab'

                    with session.get(url, stream=True, headers=headers, timeout=30) as r:
                        r.raise_for_status()
                        total_size_from_stream = int(r.headers.get('content-length', 0))
                        if mode == 'ab':
                            effective_total_size = current_file_size_on_disk + total_size_from_stream
                        else:
                            effective_total_size = total_size_from_stream

                        block_size = 1024 * 1024
                        filename_str = str(filename)

                        tqdm_kwargs = {
                            'total': effective_total_size,
                            'initial': current_file_size_on_disk if mode == 'ab' else 0,
                            'unit': 'B',
                            'unit_scale': True,
                            'unit_divisor': 1024,
                            'bar_format': '{desc}{percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                            'desc': f"{filename_str}: ",
                            'position': progress_bar_position,
                            'leave': False
                        }

                        if 'COLAB_GPU' in os.environ:
                            tqdm_kwargs.update({
                                'position': 0,
                                'leave': True
                            })

                        with open(output_path, mode) as f:
                            if mode == 'ab':
                                f.seek(current_file_size_on_disk)

                            with tqdm.tqdm(**tqdm_kwargs) as t:
                                file_downloaded_count_for_progress = current_file_size_on_disk
                                for data in r.iter_content(block_size):
                                    f.write(data)
                                    t.update(len(data))
                                    if effective_total_size != 0 and self.progress_queue is not None:
                                        file_downloaded_count_for_progress += len(data)
                                        progress_fraction = float(file_downloaded_count_for_progress) / float(effective_total_size)
                                        self.progress_queue.put((progress_fraction, filename_str))
                        break

                except (RequestException, ConnectionError, Timeout) as e:
                    print(f"Error downloading {filename}: {e}.")
                    print(f"That was attempt {attempt}/{max_retries}.", end=' ')
                    if attempt < max_retries:
                        print(f"Retry begins in {2 ** attempt} seconds.")
                        sleep(2 ** attempt)
                    else:
                        print("Failed to download after the maximum number of attempts.")
        finally:
            self.release_progress_bar_position(progress_bar_position)

    def start_download_threads(self, file_list, output_folder, start_from_scratch=False, threads=4):
        self.initialize_progress_bar_slots(threads)
        tqdm.tqdm.set_lock(tqdm.tqdm.get_lock())
        try:
            thread_map(
                lambda url: self.get_single_file(url, output_folder, start_from_scratch=start_from_scratch),
                file_list,
                max_workers=threads,
                disable=True
            )
        finally:
            print(f"\nDownload of {len(file_list)} files to {output_folder} completed.")

    def download_model_files(self, model, branch, links, sha256, output_folder, progress_queue=None, start_from_scratch=False, threads=4, specific_file=None, is_llamacpp=False):
        self.progress_queue = progress_queue

        output_folder.mkdir(parents=True, exist_ok=True)

        if not is_llamacpp:
            metadata = f'url: https://huggingface.co/{model}\n' \
                       f'branch: {branch}\n' \
                       f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'

            sha256_str = '\n'.join([f'    {item[1]} {item[0]}' for item in sha256])
            if sha256_str:
                metadata += f'sha256sum:\n{sha256_str}'

            metadata += '\n'
            (output_folder / 'huggingface-metadata.txt').write_text(metadata)

        if specific_file:
            print(f"Downloading {specific_file} to {output_folder}")
        else:
            print(f"Downloading the model to {output_folder}")

        self.start_download_threads(links, output_folder, start_from_scratch=start_from_scratch, threads=threads)

    def check_model_files(self, model, branch, links, sha256, output_folder):
        # Validate the checksums
        validated = True
        for i in range(len(sha256)):
            fpath = (output_folder / sha256[i][0])

            if not fpath.exists():
                print(f"The following file is missing: {fpath}")
                validated = False
                continue

            with open(output_folder / sha256[i][0], "rb") as f:
                bytes = f.read()
                file_hash = hashlib.sha256(bytes).hexdigest()
                if file_hash != sha256[i][1]:
                    print(f'Checksum failed: {sha256[i][0]}  {sha256[i][1]}')
                    validated = False
                else:
                    print(f'Checksum validated: {sha256[i][0]}  {sha256[i][1]}')

        if validated:
            print('[+] Validated checksums of all model files!')
        else:
            print('[-] Invalid checksums. Rerun download-model.py with the --clean flag.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', type=str, default=None, nargs='?')
    parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
    parser.add_argument('--threads', type=int, default=4, help='Number of files to download simultaneously.')
    parser.add_argument('--text-only', action='store_true', help='Only download text files (txt/json).')
    parser.add_argument('--specific-file', type=str, default=None, help='Name of the specific file to download (if not provided, downloads all).')
    parser.add_argument('--exclude-pattern', type=str, default=None, help='Regex pattern to exclude files from download.')
    parser.add_argument('--output', type=str, default=None, help='Save the model files to this folder.')
    parser.add_argument('--model-dir', type=str, default=None, help='Save the model files to a subfolder of this folder instead of the default one (text-generation-webui/user_data/models).')
    parser.add_argument('--clean', action='store_true', help='Does not resume the previous download.')
    parser.add_argument('--check', action='store_true', help='Validates the checksums of model files.')
    parser.add_argument('--max-retries', type=int, default=7, help='Max retries count when get error in download time.')
    args = parser.parse_args()

    branch = args.branch
    model = args.MODEL
    specific_file = args.specific_file
    exclude_pattern = args.exclude_pattern

    if model is None:
        print("Error: Please specify the model you'd like to download (e.g. 'python download-model.py facebook/opt-1.3b').")
        sys.exit()

    downloader = ModelDownloader(max_retries=args.max_retries)
    # Clean up the model/branch names
    try:
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)
    except ValueError as err_branch:
        print(f"Error: {err_branch}")
        sys.exit()

    # Get the download links from Hugging Face
    links, sha256, is_lora, is_llamacpp, file_sizes = downloader.get_download_links_from_huggingface(
        model, branch, text_only=args.text_only, specific_file=specific_file, exclude_pattern=exclude_pattern
    )

    # Get the output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp, model_dir=args.model_dir)

    if args.check:
        # Check previously downloaded files
        downloader.check_model_files(model, branch, links, sha256, output_folder)
    else:
        # Download files
        downloader.download_model_files(
            model, branch, links, sha256, output_folder,
            specific_file=specific_file, threads=args.threads, is_llamacpp=is_llamacpp
        )
