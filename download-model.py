'''
Downloads models from Hugging Face to models/model-name.

Example:
python download-model.py facebook/opt-1.3b

'''

import requests 
from bs4 import BeautifulSoup 
import multiprocessing
import os
import tqdm
from sys import argv

def get_file(args):
    url = args[0]
    output_folder = args[1]

    r = requests.get(url, stream=True)
    with open(f"{output_folder}/{url.split('/')[-1]}", 'wb') as f:
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
        t.close()

model = argv[1]
if model.endswith('/'):
    model = model[:-1]    
url = f'https://huggingface.co/{model}/tree/main'
output_folder = f"models/{model.split('/')[-1]}"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Finding the relevant files to download
page = requests.get(url) 
soup = BeautifulSoup(page.content, 'html.parser') 
links = soup.find_all('a')
downloads = []
for link in links:
    href = link.get('href')[1:]
    if href.startswith(f'{model}/resolve/main'):
        if href.endswith(('.json', '.txt')) or (href.endswith('.bin') and 'pytorch_model' in href):
            downloads.append(f'https://huggingface.co/{href}')

# Downloading the files
print(f"Downloading the model to {output_folder}...")
pool = multiprocessing.Pool(processes=4)
results = pool.map(get_file, [[downloads[i], output_folder] for i in range(len(downloads))])
pool.close()
pool.join()
