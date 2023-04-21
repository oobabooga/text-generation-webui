# Text generation web UI | Now with Moderation!

A [gradio web UI](https://gradio.app/) for running Large Language Models, with content moderation powered by [LanceDB](https://github.com/lancedb/lancedb).

Built off of  [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui).

|![Image1](https://github.com/oobabooga/screenshots/raw/main/qa.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/cai3.png) |
|:---:|:---:|
|![Image3](https://github.com/oobabooga/screenshots/raw/main/gpt4chan.png) | ![Image4](https://github.com/oobabooga/screenshots/raw/main/galactica.png) |

## Features

* Chat moderation 
    * toxicity, obscenity, threats, insults and identity hate [Jigsaw Dataset](https://www.kaggle.com/datasets/adldotori/all-in-one-jigsaw)
    * Lightning fast chat moderation inference with [LanceDB](https://github.com/lancedb/lancedb)
* Switch between models: Alpaca, Vicuna, Open Assistant, Dolly, Koala, and ChatGLM formats 
* Efficient text streaming and parameter presets
    * Layers splitting across GPU(s), CPU, and disk
    * CPU mode
* API [with](https://github.com/mrubash1/text-generation-webui/blob/main/api-example-stream.py) streaming and [without](https://github.com/mrubash1/text-generation-webui/blob/main/api-example.py) streaming
* [LoRA (loading and training)](https://github.com/mrubash1/text-generation-webui/wiki/Using-LoRAs)
* [Extensions](https://github.com/mrubash1/text-generation-webui/wiki/Extensions)

## Installation (Mac only)

### Manual installation using Conda

Recommended if you have some experience with the command-line.

#### 0. [Install Conda](https://docs.conda.io/en/latest/miniconda.html)


#### 1. Create a new conda environment

```bash
conda create -n textgen python=3.10.9
conda activate textgen
```

#### 2. Install Pytorch
First you need to install macOS Ventura 13.3 Beta, **it does not work on 13.2.**
Then you have to install torch dev version, it does not work on 2.0.0.
```bash
pip install -U --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

The up to date commands can be found here: https://pytorch.org/get-started/locally/. 

#### 3. Install the web UI

```
git clone https://github.com/mrubash1/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```
## Downloading models

Models should be placed inside the `models` folder.

[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is the main place to download models. These are some examples:

* [Pythia](https://huggingface.co/models?sort=downloads&search=eleutherai%2Fpythia+deduped)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)

You can automatically download a model from HF using the script `download-model.py`, for example for Facebook opt-1.3b:
```bash
conda activate textgen
cd text-generation-webui
python download-model.py facebook/opt-1.3b
```

If you want to download a model manually, note that all you need are the json, txt, and pytorch\*.bin (or model*.safetensors) files. The remaining files are not necessary.

## Build LanceDB for the Moderation API
Download [Kaggle's all-in-one-jigsaw dataset](https://www.kaggle.com/datasets/adldotori/all-in-one-jigsaw). To do programmaticaly :
* Go to your Kaggle account settings (https://www.kaggle.com/<your_username>/account) and click on "Create New API Token" to download your kaggle.json file to ~/Downloads
* Place the kaggle.json file in your home directory under a folder named .kaggle (e.g., /home/username/.kaggle/kaggle.json)
```bash
 mkdir ~/.kaggle 
 mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
```
* Set the file permissions for the kaggle.json file to read and write access only for the owner:
```bash
chmod 600 ~/.kaggle/kaggle.json
```
Now that your Kaggle API is set up, you can build a vectorDB (1000 jigsaw samples by default, can increase to 2.2M at max, with overnight training)
```bash

python build-lancedb.py --samples_of_jigsaw_to_process 1000
```

You can test a query by running
```bash
python modules/test_embeddings.py -q "I hate those people and everything they stand for"
```

## Starting the web UI
```bash
python server.py
```

Then browse to 

`http://localhost:7860/?__theme=dark`

Optionally, you can use the following command-line flags:

#### Basic settings

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | Show this help message and exit. |
| `--notebook`                               | Launch the web UI in notebook mode, where the output is written to the same text box as the input. |
| `--chat`                                   | Launch the web UI in chat mode. |
| `--model MODEL`                            | Name of the model to load by default. |
| `--lora LORA`                              | Name of the LoRA to apply to the model by default. |
| `--model-dir MODEL_DIR`                    | Path to directory with all the models. |
| `--lora-dir LORA_DIR`                      | Path to directory with all the loras. |
| `--model-menu`                             | Show a model menu in the terminal when the web UI is first launched. |
| `--no-stream`                              | Don't stream the text output in real time. |
| `--settings SETTINGS_FILE`                 | Load the default interface settings from this json file. See `settings-template.json` for an example. If you create a file called `settings.json`, this file will be loaded by default without the need to use the `--settings` flag. |
| `--extensions EXTENSIONS [EXTENSIONS ...]` | The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--verbose`                                | Print the prompts to the terminal. |

#### Accelerate/transformers

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--cpu`                                     | Use the CPU to generate text. Warning: Training on CPU is extremely slow.|
| `--auto-devices`                            | Automatically split the model across the available GPU(s) and CPU. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Maxmimum GPU memory in GiB to be allocated per GPU. Example: `--gpu-memory 10` for a single GPU, `--gpu-memory 10 5` for two GPUs. You can also set values in MiB like `--gpu-memory 3500MiB`. |
| `--cpu-memory CPU_MEMORY`                   | Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.|
| `--disk`                                    | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR`           | Directory to save the disk cache to. Defaults to `cache/`. |
| `--load-in-8bit`                            | Load the model with 8-bit precision.|
| `--bf16`                                    | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--no-cache`                                | Set `use_cache` to False while generating text. This reduces the VRAM usage a bit with a performance cost. |
| `--xformers`                                | Use xformer's memory efficient attention. This should increase your tokens/s. |
| `--sdp-attention`                           | Use torch 2.0's sdp attention. |
| `--trust-remote-code`                       | Set trust_remote_code=True while loading a model. Necessary for ChatGLM. |

#### Gradio

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--listen`                            | Make the web UI reachable from your local network. |
| `--listen-host LISTEN_HOST`           | The hostname that the server will use. |
| `--listen-port LISTEN_PORT`           | The listening port that the server will use. |
| `--share`                             | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch`                       | Open the web UI in the default browser upon launch. |
| `--gradio-auth-path GRADIO_AUTH_PATH` | Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3" |

Out of memory errors? [Check the low VRAM guide](https://github.com/oobabooga/text-generation-webui/wiki/Low-VRAM-guide).

## System requirements

Check the [wiki](https://github.com/oobabooga/text-generation-webui/wiki/System-requirements) for some examples of VRAM and RAM usage in both GPU and CPU mode.

## Contributing

Pull requests, suggestions, and issue reports are welcome. That said, encourage you to go to the source: [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui) 

## Credits

- [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui) and all the great contributors to text-generation-webui!
- Incredible guidance, coding and advice from the [LanceDB and Eto Team](https://github.com/lancedb/lancedb): [Chang](https://github.com/changhiskhan) and [Lei](https://github.com/eddyxu) 
