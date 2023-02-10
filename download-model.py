'''
Downloads models from Hugging Face to models/model-name.

Example:
python download-model.py facebook/opt-1.3b

'''
import argparse
import multiprocessing
import re
import sys
from pathlib import Path

import requests
import tqdm
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument('MODEL', type=str)
parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
parser.add_argument('--threads', type=int, default=1, help='Number of files to download simultaneously.')
args = parser.parse_args()

def get_file(args):
    url = args[0]
    output_folder = args[1]
    idx = args[2]
    tot = args[3]

    print(f"Downloading file {idx} of {tot}...")
    r = requests.get(url, stream=True)
    with open(output_folder / Path(url.split('/')[-1]), 'wb') as f:
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
        t.close()

def sanitize_branch_name(branch_name):
    pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
    if pattern.match(branch_name):
        return branch_name
    else:
        raise ValueError("Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

if __name__ == '__main__':
    model = args.MODEL
    if model[-1] == '/':
        model = model[:-1]
        branch = args.branch
    if args.branch is None:
        branch = 'main'
    else:
        try:
            branch_name = args.branch
            branch = sanitize_branch_name(branch_name)
        except ValueError as err_branch:
            print(f"Error: {err_branch}")
            sys.exit()
    url = f'https://huggingface.co/{model}/tree/{branch}'
    if branch != 'main':
        output_folder = Path("models") / (model.split('/')[-1] + f'_{branch}')
    else:
        output_folder = Path("models") / model.split('/')[-1]
    if not output_folder.exists():
        output_folder.mkdir()

    # Finding the relevant files to download
    page = requests.get(url) 
    soup = BeautifulSoup(page.content, 'html.parser') 
    links = soup.find_all('a')
    downloads = []
    for link in links:
        href = link.get('href')[1:]
        if href.startswith(f'{model}/resolve/{branch}'):
            is_pytorch = href.endswith('.bin') and 'pytorch_model' in href
            is_safetensors = href.endswith('.safetensors') and 'model' in href
            if href.endswith(('.json', '.txt')) or is_pytorch or is_safetensors:
                downloads.append(f'https://huggingface.co/{href}')

    # Downloading the files
    print(f"Downloading the model to {output_folder}")
    pool = multiprocessing.Pool(processes=args.threads)
    results = pool.map(get_file, [[downloads[i], output_folder, i+1, len(downloads)] for i in range(len(downloads))])
    pool.close()
    pool.join()
