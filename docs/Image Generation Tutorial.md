# Image Generation Tutorial

This feature allows you to generate images using `diffusers` models like [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) directly within the web UI.

<img alt="print" src="https://github.com/user-attachments/assets/5108de50-658b-4e93-b2ae-4656d076bc9d" />


## Installation

1. Clone the repository with

```
git clone https://github.com/oobabooga/text-generation-webui
```

or download it from [here](https://github.com/oobabooga/text-generation-webui/archive/refs/heads/main.zip) and unzip it.

2. Use the one-click installer.

- Windows: Double click on `start_windows.bat`
- Linux: Run `./start_linux.sh`
- macOS: Run `./start_macos.sh`

Note: Image generation does not work with the portable builds in `.zip` format in the [Releases page](https://github.com/oobabooga/text-generation-webui/releases). You need the "full" version of the web UI.

## Downloading a model

1. Once installation ends, browse to `http://127.0.0.1:7860/`.
2. Click on "Image AI" on the left.
3. Click on "Model" at the top.
4. In the "Download model" field, paste `https://huggingface.co/Tongyi-MAI/Z-Image-Turbo` and click "Download".
5. Wait for the download to finish (it's 31 GB).

## Loading the model

Select the quantization option in the "Quantization" menu and click "Load".

The memory usage for `Z-Image-Turbo` for each option is:

| Quantization Method | VRAM Usage |
| :--- | :--- |
| None (FP16/BF16) | 25613 MiB |
| bnb-8bit | 16301 MiB |
| bnb-8bit + CPU Offload | 16235 MiB |
| bnb-4bit | 11533 MiB |
| bnb-4bit + CPU Offload | 7677 MiB |

The `torchao` options support `torch.compile` for faster image generation, with `float8wo` specifically providing native hardware acceleration for RTX 40-series and newer GPUs.

Note: The next time you launch the web UI, the model will get automatically loaded with your last settings when you try to generate an image. You do not need to go to the Model tab and click "Load" each time.

## Generating images:

1. While still in the "Image AI" page, go to the "Generate" tab.
2. Type your prompt and click on the Generate button.

### Model-specific settings

- For Z-Image-Turbo, make sure to keep CFG Scale at 0 and Steps at 9. Do not write a Negative Prompt as it will get ignored with this CFG Scale value.

### LLM Prompt Variations

To use this feature, you need to load an LLM in the main "Model" page on the left.

If you have no idea what to use, do this to get started:

1. Download [Qwen3-4B-Q3_K_M.gguf](https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q3_K_M.gguf) to your `text-generation-webui/user_data/models` folder.
2. Select the model in the dropdown menu in the "Model" page.
3. Click Load.

Then go back to the "Image AI" page and check "LLM Prompt Variations".

After that, your prompts will be automatically updated by the LLM each time you generate an image. If you use a "Sequential Count" value greater than 1, a new prompt will be created for each sequential batch.

The improvement in creativity is striking (prompt: `Photo of a beautiful woman at night under moonlight`):

<img  alt="comparison_collage" src="https://github.com/user-attachments/assets/67884832-2800-41cb-a146-e88e25af89c4" />

## Generating images over API

It is possible to generate images using the project's API. Just make sure to start the server with `--api`, either by

1. Passing the `--api` flag to your `start` script, like `./start_linux.sh --api`, or
2. Writing `--api` to your `user_data/CMD_FLAGS.txt` file and relaunching the web UI.

Here is an API call example:

```
curl http://127.0.0.1:5000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "an orange tree",
    "steps": 9,
    "cfg_scale": 0,
    "batch_size": 1,
    "batch_count": 1
  }'
```
