# Image Generation Tutorial

This feature allows you to generate images using `diffusers` models like [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) directly within the web UI.

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

If you have less GPU memory than _, check the "CPU Offload" option.

Note: The next time you launch the web UI, the model will get automatically loaded with your last settings when you try to generate an image. You do not need to go to the Model tab and click "Load" each time.

## Generating images:

1. While still in the "Image AI" page, go to the "Generate" tab.
2. Type your prompt and click on the Generate button.

### LLM Prompt Variations

To use this feature, you need to load an LLM in the main "Model" tab on the left.

If you have no idea what to use, do this to get started:

1. Download [Qwen3-4B-Q3_K_M.gguf](https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q3_K_M.gguf) to your `text-generation-webui/user_data/models` folder.
2. Select the model in the dropdown menu in the "Model" page.
3. Click Load.

Then go back to the "Image AI" page and check "LLM Prompt Variations".

After that, your prompts will be automatically updated by the LLM each time you generate an image. If you use sequential batch count value greater than 1, a new prompt will be created for each sequential batch.

The improvement in creativity is striking:

### Model-specific settings

- For Z-Image-Turbo, make sure to keep CFG Scale at 0 and Steps at 9. Do not write a Negative Prompt as it will get ignored with this CFG Scale value.

