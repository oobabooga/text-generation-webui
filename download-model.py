'''
Downloads models from Hugging Face to models/username_modelname.

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
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException

base = "https://huggingface.co"


class ModelDownloader:
    def __init__(self, max_retries=5):
        self.max_retries = max_retries

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

    def get_download_links_from_huggingface(self, model, branch, text_only=False, specific_file=None):
        session = self.get_session()
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
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

                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                is_safetensors = re.match(r".*\.safetensors", fname)
                is_pt = re.match(r".*\.pt", fname)
                is_gguf = re.match(r'.*\.gguf', fname)
                is_tiktoken = re.match(r".*\.tiktoken", fname)
                is_tokenizer = re.match(r"(tokenizer|ice|spiece).*\.model", fname) or is_tiktoken
                is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any((is_pytorch, is_safetensors, is_pt, is_gguf, is_tokenizer, is_text)):
                    if 'lfs' in dict[i]:
                        sha256.append([fname, dict[i]['lfs']['oid']])

                    if is_text:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                        classifications.append('text')
                        continue

                    if not text_only:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
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
        if (has_pytorch or has_pt) and has_safetensors:
            for i in range(len(classifications) - 1, -1, -1):
                if classifications[i] in ['pytorch', 'pt']:
                    links.pop(i)

        # For GGUF, try to download only the Q4_K_M if no specific file is specified.
        # If not present, exclude all GGUFs, as that's likely a repository with both
        # GGUF and fp16 files.
        if has_gguf and specific_file is None:
            has_q4km = False
            for i in range(len(classifications) - 1, -1, -1):
                if 'q4_k_m' in links[i].lower():
                    has_q4km = True

            if has_q4km:
                for i in range(len(classifications) - 1, -1, -1):
                    if 'q4_k_m' not in links[i].lower():
                        links.pop(i)
            else:
                for i in range(len(classifications) - 1, -1, -1):
                    if links[i].lower().endswith('.gguf'):
                        links.pop(i)

        is_llamacpp = has_gguf and specific_file is not None
        return links, sha256, is_lora, is_llamacpp

    def get_output_folder(self, model, branch, is_lora, is_llamacpp=False):
        base_folder = 'models' if not is_lora else 'loras'

        # If the model is of type GGUF, save directly in the base_folder
        if is_llamacpp:
            return Path(base_folder)

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'

        output_folder = Path(base_folder) / output_folder
        return output_folder

    def get_single_file(self, url, output_folder, part, progress_bars, start_from_scratch, timeout=30):
        filename = Path(url.rsplit('/', 1)[-1])
        part_filename = output_folder / f"{filename}.part{part}"
        headers = {}

        session = requests.Session()
        retries = Retry(total=self.max_retries, backoff_factor=0, status_forcelist=[502, 503, 504, 429])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        if not start_from_scratch and part_filename.exists():
            existing_size = part_filename.stat().st_size
            headers['Range'] = f'bytes={existing_size}-'
        else:
            existing_size = 0

        # Ensure the progress bar for this URL exists
        if url not in progress_bars:
            progress_bars[url] = tqdm.tqdm(total=0, desc=f"Downloading {filename}", unit='iB', unit_scale=True)

        while True:
            try:
                with session.get(url, stream=True, headers=headers, timeout=timeout) as r:
                    r.raise_for_status()
                    # If server supports range requests, adjust total size based on the range we requested
                    if 'content-range' in r.headers:
                        total_size = int(r.headers.get('content-range').split('/')[-1])
                    else:
                        total_size = int(r.headers.get('content-length', 0))

                    # Set the total size in the corresponding progress bar if it's the first part
                    if part == 0:
                        progress_bars[url].total = total_size
                        progress_bars[url].refresh()

                    mode = 'ab' if existing_size else 'wb'
                    with open(part_filename, mode) as f:
                        for data in r.iter_content(1024 * 1024):  # 1MB chunks
                            f.write(data)
                            progress_bars[url].update(len(data))
                break  # Successfully completed download, exit loop
            except RequestException as e:
                print(f"Download {filename} failed with error: {e}, retrying.")
                # Reset the progress bar for this URL to the existing size
                progress_bars[url].n = existing_size
                progress_bars[url].refresh()

    def download_and_merge_file(self, url, output_folder, total_parts, progress_bars, start_from_scratch):
        # Download all parts of the file concurrently
        threads = []
        for part in range(total_parts):
            thread = threading.Thread(target=self.get_single_file, args=(url, output_folder, part, progress_bars, start_from_scratch))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Merge the parts into a single file
        filename = Path(url.rsplit('/', 1)[-1])
        output_path = output_folder / filename
        with open(output_path, 'wb') as f_out:
            for part in range(total_parts):
                part_filename = output_folder / f"{filename}.part{part}"
                with open(part_filename, 'rb') as f_part:
                    f_out.write(f_part.read())
                part_filename.unlink()

        # Manually set progress bar as complete
        progress_bars[url].n = progress_bars[url].total
        progress_bars[url].refresh()

    def start_download_threads(self, file_list, output_folder, threads=8, total_parts=8, start_from_scratch=False):
        progress_bars = {}
        all_threads = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            for url in file_list:
                filename = Path(url.rsplit('/', 1)[-1])
                progress_bar = tqdm.tqdm(total=0, desc=f"Downloading {filename}", unit='iB', unit_scale=True)
                progress_bars[url] = progress_bar

                future = executor.submit(self.download_and_merge_file, url, output_folder, total_parts, progress_bars, start_from_scratch)
                all_threads.append(future)

        # Wait for all threads to complete
        for future in all_threads:
            future.result()

        # Update progress bars to complete state and refresh
        for url in file_list:
            progress_bars[url].n = progress_bars[url].total
            progress_bars[url].refresh()

                    
    def download_model_files(self, model, branch, links, sha256, output_folder, progress_bar=None, start_from_scratch=False, threads=4, specific_file=None, is_llamacpp=False):
        self.progress_bar = progress_bar

        # Create the folder and writing the metadata
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
    parser.add_argument('--threads', type=int, default=8, help='Number of threads to download simultaneously.')
    parser.add_argument('--text-only', action='store_true', help='Only download text files (txt/json).')
    parser.add_argument('--specific-file', type=str, default=None, help='Name of the specific file to download (if not provided, downloads all).')
    parser.add_argument('--output', type=str, default=None, help='The folder where the model should be saved.')
    parser.add_argument('--clean', action='store_true', help='Does not resume the previous download.')
    parser.add_argument('--check', action='store_true', help='Validates the checksums of model files.')
    parser.add_argument('--max-retries', type=int, default=5, help='Max retries count when get error in download time.')
    args = parser.parse_args()

    branch = args.branch
    model = args.MODEL
    specific_file = args.specific_file

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
    links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=args.text_only, specific_file=specific_file)

    # Get the output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp)

    if args.check:
        # Check previously downloaded files
        downloader.check_model_files(model, branch, links, sha256, output_folder)
    else:
        # Download files
        downloader.download_model_files(model, branch, links, sha256, output_folder, specific_file=specific_file, threads=args.threads, is_llamacpp=is_llamacpp)
