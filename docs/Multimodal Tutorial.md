## Getting started

### 1. Find a multimodal model

GGUF models with vision capabilities are uploaded along a `mmproj` file to Hugging Face.

For instance, [unsloth/gemma-3-4b-it-GGUF](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/tree/main) has this:

<img width="414" height="270" alt="print1" src="https://github.com/user-attachments/assets/ac5aeb61-f6a2-491e-a1f0-47d6e27ea286" />

### 2. Download the model to `user_data/models`

As an example, download

https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_S.gguf?download=true

to your `text-generation-webui/user_data/models` folder.

### 3. Download the associated mmproj file to `user_data/mmproj`

Then download

https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf?download=true

to your `text-generation-webui/user_data/mmproj` folder. Name it `mmproj-gemma-3-4b-it-F16.gguf` to give it a recognizable name.

### 4. Load the model

1. Launch the web UI
2. Navigate to the Model tab
3. Select the GGUF model in the Model dropdown:

<img width="545" height="92" alt="print2" src="https://github.com/user-attachments/assets/3f920f50-e6c3-4768-91e2-20828dd63a1c" />

4. Select the mmproj file in the Multimodal (vision) menu:

<img width="454" height="172" alt="print3" src="https://github.com/user-attachments/assets/a657e20f-0ceb-4d71-9fe4-2b78571d20a6" />

5. Click "Load"

### 5. Send a message with an image

Select your image by clicking on the 📎 icon and send your message:

<img width="368" height="135" alt="print5" src="https://github.com/user-attachments/assets/6175ec9f-04f4-4dba-9382-4ac80d5b0b1f" />

The model will reply with great understanding of the image contents:

<img width="809" height="884" alt="print6" src="https://github.com/user-attachments/assets/be4a8f4d-619d-49e6-86f5-012d89f8db8d" />

## Multimodal with ExLlamaV3

Multimodal also works with the ExLlamaV3 loader (the non-HF one).

No additional files are necessary, just load a multimodal EXL3 model and send an image.

Examples of models that you can use:

- https://huggingface.co/turboderp/gemma-3-27b-it-exl3
- https://huggingface.co/turboderp/Mistral-Small-3.1-24B-Instruct-2503-exl3

## Multimodal API examples

In the page below you can find some ready-to-use examples:

[Multimodal/vision (llama.cpp and ExLlamaV3)](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#multimodalvision-llamacpp-and-exllamav3)
