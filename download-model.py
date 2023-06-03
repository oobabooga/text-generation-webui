'''
Downloads models from Hugging Face to models/model-name.

Examples:
python download-model.py facebook/opt-1.3b

Using the built-in short name:
python download-model.py "OPT 1.3B"

Specifying a subset of model files within a repository folder
python download-model.py --select "ggml-vicuna-13B-1.1-q5_0.bin|ggml-vicuna-13B-1.1-q5_1.bin" CRD716/ggml-vicuna-1.1-quantized

'''

import argparse
import base64
import datetime
import hashlib
import re
import os
import sys
from pathlib import Path

import requests
import tqdm
from tqdm.contrib.concurrent import thread_map


def select_model_from_default_options():
    models = {
        "OPT 6.7B": ("facebook", "opt-6.7b", "main"),
        "OPT 2.7B": ("facebook", "opt-2.7b", "main"),
        "OPT 1.3B": ("facebook", "opt-1.3b", "main"),
        "OPT 350M": ("facebook", "opt-350m", "main"),
        "GALACTICA 6.7B": ("facebook", "galactica-6.7b", "main"),
        "GALACTICA 1.3B": ("facebook", "galactica-1.3b", "main"),
        "GALACTICA 125M": ("facebook", "galactica-125m", "main"),
        "Pythia-6.9B-deduped": ("EleutherAI", "pythia-6.9b-deduped", "main"),
        "Pythia-2.8B-deduped": ("EleutherAI", "pythia-2.8b-deduped", "main"),
        "Pythia-1.4B-deduped": ("EleutherAI", "pythia-1.4b-deduped", "main"),
        "Pythia-410M-deduped": ("EleutherAI", "pythia-410m-deduped", "main"),
    }

    choices = {}
    print("Select the model that you want to download:\n")
    for i, name in enumerate(models):
        char = chr(ord('A') + i)
        choices[char] = name
        print(f"{char}) {name}")

    char_hugging = chr(ord('A') + len(models))
    print(f"{char_hugging}) Manually specify a Hugging Face model")
    char_exit = chr(ord('A') + len(models) + 1)
    print(f"{char_exit}) Do not download a model")
    print()
    print("Input> ", end='')
    choice = input()[0].strip().upper()
    if choice == char_exit:
        exit()
    elif choice == char_hugging:
        print("""\nType the name of your desired Hugging Face model in the format organization/name.

Examples:
facebook/opt-1.3b
EleutherAI/pythia-1.4b-deduped
""")

        print("Input> ", end='')
        model = input()
        branch = "main"
    else:
        arr = models[choices[choice]]
        model = f"{arr[0]}/{arr[1]}"
        branch = arr[2]

    return model, branch


class ModelDownloader:
    """
    Manages state for downloading from HF, especially with regard to auth,
    allowing download of protected models
    """
    def __init__(self):
        self._session = requests.Session()
        if all((os.getenv('HF_USER'), os.getenv('HF_PASS'))):
            self._session.auth = (os.getenv('HF_USER'), os.getenv('HF_PASS'))

    def sanitize_model_and_branch_names(self, model, branch):
        """
        Make sure model & branch names are valid & do some clean-up, if need be
        """
        model = model.rstrip("/")

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

        return model, branch

    def get_download_links_from_huggingface(self, model, branch, text_only=False, select=None):
        """
        Gather a list of download links & checksums (SHA256) from all pages of the files manifest for the HF repository
        """
        base = "https://huggingface.co"
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        if select:
            select = select.split("|")

        links = []
        sha256 = []
        classifications = []
        has_pytorch = False
        has_pt = False
        has_ggml = False
        has_safetensors = False
        is_lora = False
        # The loop is for paging over repos with many files
        while True:
            # Get the file manifest from the HF repo
            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            manifest = resp.json()

            # We've hit an empty page, so we're done
            if len(manifest) == 0:
                break

            for i in range(len(manifest)):
                fname = manifest[i]['path']
                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match("(pytorch|adapter)_model.*\.bin", fname)
                is_safetensors = re.match(".*\.safetensors", fname)
                is_pt = re.match(".*\.pt", fname)
                is_ggml = re.match(".*ggml.*\.bin", fname)
                is_tokenizer = re.match("(tokenizer|ice).*\.model", fname)
                is_text = re.match(".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any((is_pytorch, is_safetensors, is_pt, is_ggml, is_tokenizer, is_text)):
                    # if a select option is given, and current file is a binary, make sure it's in the select list
                    if select and any((is_pytorch, is_pt, is_safetensors, is_ggml)):
                        if fname not in select:
                            continue

                    if 'lfs' in manifest[i]:
                        sha256.append([fname, manifest[i]['lfs']['oid']])

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
                        elif is_ggml:
                            has_ggml = True
                            classifications.append('ggml')

            # Set up to move the next page (if any) of HF repo contents
            cursor = base64.b64encode(f'{{"file_name":"{manifest[-1]["path"]}"}}'.encode()) + b':50'
            cursor = base64.b64encode(cursor)
            cursor = cursor.replace(b'=', b'%3D')

        # If both pytorch and safetensors are available, download safetensors only
        if (has_pytorch or has_pt) and has_safetensors:
            for i in range(len(classifications) - 1, -1, -1):
                if classifications[i] in ['pytorch', 'pt']:
                    links.pop(i)

        return links, sha256, is_lora

    def get_output_folder(self, model, branch, is_lora, base_folder=None):
        """
        Figure out where to put the downloaded files, based on user params,
        model type & HF repo branch
        """
        if base_folder is None:
            base_folder = 'models' if not is_lora else 'loras'

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'
        output_folder = Path(base_folder) / output_folder
        return output_folder

    def get_single_file(self, url, output_folder, start_from_scratch=False):
        """
        Download an individual file from the HF repo, within a thread. Also handles resume.
        """
        filename = Path(url.rsplit('/', 1)[1])
        output_path = output_folder / filename
        if output_path.exists() and not start_from_scratch:
            # Check if the file has already been downloaded completely
            resp = self._session.get(url, stream=True, timeout=10)
            total_size = int(resp.headers.get('content-length', 0))
            if output_path.stat().st_size >= total_size:
                return
            # Otherwise, resume the download from where it left off
            headers = {'Range': f'bytes={output_path.stat().st_size}-'}
            mode = 'ab'
        else:
            headers = {}
            mode = 'wb'

        resp = self._session.get(url, stream=True, headers=headers, timeout=10)
        with open(output_path, mode) as f:
            total_size = int(resp.headers.get('content-length', 0))
            block_size = 1024
            with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True, bar_format='{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6}') as t:
                for data in resp.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)

    def start_download_threads(self, file_list, output_folder, start_from_scratch=False, threads=1):
        thread_map(lambda url: self.get_single_file(url, output_folder, start_from_scratch=start_from_scratch), file_list, max_workers=threads, disable=True)

    def download_model_files(self, model, branch, links, sha256, output_folder, start_from_scratch=False, threads=1):
        """
        Make the preparations to launch the threads to download the individual files
        """
        # Creating the folder and writing the metadata
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
        with open(output_folder / 'huggingface-metadata.txt', 'w') as f:
            f.write(f'url: https://huggingface.co/{model}\n')
            f.write(f'branch: {branch}\n')
            f.write(f'download date: {str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}\n')
            sha256_str = ''
            for i in range(len(sha256)):
                sha256_str += f'    {sha256[i][1]} {sha256[i][0]}\n'
            if sha256_str != '':
                f.write(f'sha256sum:\n{sha256_str}')

        # Downloading the files
        print(f"Downloading the model to {output_folder}")
        self.start_download_threads(links, output_folder, start_from_scratch=start_from_scratch, threads=threads)

    def check_model_files(self, model, branch, links, sha256, output_folder):
        """
        Validate the checksums of model files, to make sure we completely downloaded exactly what we should have
        """
        # Validate the checksums
        validated = True
        for i in range(len(sha256)):
            fpath = (output_folder / sha256[i][0])

            if not fpath.exists():
                print(f"The following file is missing: {fpath}")
                validated = False
                continue

            with open(output_folder / sha256[i][0], "rb") as f:
                fbytes = f.read()
                file_hash = hashlib.sha256(fbytes).hexdigest()
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
    parser.add_argument('--threads', type=int, default=1, help='Number of files to download simultaneously.')
    parser.add_argument('--text-only', action='store_true', help='Only download text files (txt/json).')
    parser.add_argument('--output', type=str, default=None, help='The folder where the model should be saved.')
    parser.add_argument('--clean', action='store_true', help='Does not resume the previous download.')
    parser.add_argument('--check', action='store_true', help='Validates the checksums of model files.')
    parser.add_argument('--select', type=str, help='Pipe separated list of .bin or .safetensors files to download. No other files of these extensions will be.')
    args = parser.parse_args()

    branch = args.branch
    model = args.MODEL
    if model is None:
        model, branch = select_model_from_default_options()

    downloader = ModelDownloader()
    # Cleaning up the model/branch names
    try:
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)
    except ValueError as err_branch:
        print(f"Error: {err_branch}")
        sys.exit()

    # Getting the download links from Hugging Face
    links, sha256, is_lora = downloader.get_download_links_from_huggingface(model, branch, text_only=args.text_only, select=args.select)

    # Getting the output folder
    output_folder = downloader.get_output_folder(model, branch, is_lora, base_folder=args.output)

    if args.check:
        # Check previously downloaded files
        downloader.check_model_files(model, branch, links, sha256, output_folder)
    else:
        # Download files
        downloader.download_model_files(model, branch, links, sha256, output_folder, threads=args.threads)
