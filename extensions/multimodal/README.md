# Multimodal

## Description

Adds support for multimodality (text+images) to text-generation-webui.

https://user-images.githubusercontent.com/3718215/233817203-69b57e77-0c55-4fd6-b742-3204bb13b8fc.mp4

## Usage

To run this extension, download a LLM that supports multimodality, and then start server.py with the appropriate `--multimodal-pipeline` argument. Examples:

```
python server.py --model wojtab_llava-7b-v0-4bit-128g --multimodal-pipeline llava-7b --chat
python3 server.py --model wojtab_llava-13b-v0-4bit-128g --multimodal-pipeline llava-13b --chat
python server.py --model anon8231489123_vicuna-13b-GPTQ-4bit-128g --multimodal-pipeline minigpt4-13b --chat
python server.py --model llama-7b-4bit --multimodal-pipeline minigpt4-7b --chat
```

There is built-in support for LLaVA-v0-13B and LLaVA-v0-7b. To install `minigpt4`:

- clone https://github.com/Wojtab/minigpt-4-pipeline into `extensions/multimodal/pipelines`
- install the requirements.txt

The same procedure should be used to install other pipelines, which can then be used with `--multimodal-pipeline [pipeline name]`. For additional multimodal pipelines refer to the compatibility section below.

Do note, that each image takes up a considerable amount of tokens, so adjust `max_new_tokens` to be at most 1700 (recommended value is between 200 to 500), so the images don't get truncated.

To send an image, just upload it to the extension field below chat, and send a prompt as always. The image will be added to the end of your message. If you wish to modify the placement, include a string `<image>` in your prompt.

Additionally, there is *Embed all images, not only the last one* checkbox. It modifies the image embeddings, by default (if it's unchecked), all but the most recent images have their embeddings empty, so they are not fed to the network. It seems as if some multimodal networks consider the features in all images at the same time as if they were a single image. Due to this behavior, by default, the extension skips previous images. However, it can lead to sub-par generation on other pipelines. If you want to include all images, just tick this checkbox.

## Compatibility
As of now, the following multimodal pipelines are supported:
|Pipeline|`--multimodal-pipeline`|Default LLM|LLM info(for the linked model)|Pipeline repository|
|-|-|-|-|-|
|[LLaVA 13B](https://github.com/haotian-liu/LLaVA)|`llava-13b`|[LLaVA 13B](https://huggingface.co/wojtab/llava-13b-v0-4bit-128g)|GPTQ 4-bit quant, old CUDA|built-in|
|[LLaVA 7B](https://github.com/haotian-liu/LLaVA)|`llava-7b`|[LLaVA 7B](https://huggingface.co/wojtab/llava-7b-v0-4bit-128g)|GPTQ 4-bit quant, old CUDA|built-in|
|[MiniGPT-4 7B](https://github.com/Vision-CAIR/MiniGPT-4)|`minigpt4-7b`|[Vicuna v0 7B](https://huggingface.co/TheBloke/vicuna-7B-GPTQ-4bit-128g)|GPTQ 4-bit quant, new format|[Wojtab/minigpt-4-pipeline](https://github.com/Wojtab/minigpt-4-pipeline)|
|[MiniGPT-4 13B](https://github.com/Vision-CAIR/MiniGPT-4)|`minigpt4-13b`|[Vicuna v0 13B](https://huggingface.co/anon8231489123/vicuna-13b-GPTQ-4bit-128g)|GPTQ 4-bit quant, old CUDA|[Wojtab/minigpt-4-pipeline](https://github.com/Wojtab/minigpt-4-pipeline)|
|[InstructBLIP 7B](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)|`instructblip-7b`|[Vicuna v1.1 7B](https://huggingface.co/TheBloke/vicuna-7B-1.1-GPTQ-4bit-128g)|GPTQ 4-bit quant|[kjerk/instructblip-pipeline](https://github.com/kjerk/instructblip-pipeline)|
|[InstructBLIP 13B](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)|`instructblip-13b`|[Vicuna v1.1 13B](https://huggingface.co/TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g)|GPTQ 4-bit quant|[kjerk/instructblip-pipeline](https://github.com/kjerk/instructblip-pipeline)|

Some pipelines could support different LLMs but do note that while it might work, it isn't a supported configuration.

DO NOT report bugs if you are using a different LLM.

DO NOT report bugs with pipelines in this repository (unless they are built-in)

## Extension config
This extension uses the following parameters (from `settings.json`):
|Parameter|Description|
|---------|-----------|
|`multimodal-vision_bits`|Number of bits to load vision models (CLIP/ViT) feature extractor in (most pipelines should support either 32 or 16, default=32)|
|`multimodal-vision_device`|Torch device to run the feature extractor on, for example, `cpu` or `cuda:0`, by default `cuda:0` if available|
|`multimodal-projector_bits`|Number of bits to load feature projector model(s) in (most pipelines should support either 32 or 16, default=32)|
|`multimodal-projector_device`|Torch device to run the feature projector model(s) on, for example `cpu` or `cuda:0`, by default `cuda:0` if available|
|`multimodal-add_all_images_to_prompt`|Default value of "Embed all images, not only the last one" checkbox|

## Usage through API

You can run the multimodal inference through API, by inputting the images to prompt. Images are embedded like so: `f'<img src="data:image/jpeg;base64,{img_str}">'`, where `img_str` is base-64 jpeg data. Note that you will need to launch `server.py` with the arguments `--api --extensions multimodal`. 

Python example:

```Python
import base64
import requests

CONTEXT = "You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. Follow the instructions carefully and explain your answers in detail.### Human: Hi!### Assistant: Hi there! How can I help you today?\n"

with open('extreme_ironing.jpg', 'rb') as f:
    img_str = base64.b64encode(f.read()).decode('utf-8')
    prompt = CONTEXT + f'### Human: What is unusual about this image: \n<img src="data:image/jpeg;base64,{img_str}">### Assistant: '
    print(requests.post('http://127.0.0.1:5000/api/v1/generate', json={'prompt': prompt, 'stopping_strings': ['\n###']}).json())
```
script output:
```Python
{'results': [{'text': "The unusual aspect of this image is that a man is standing on top of a yellow minivan while doing his laundry. He has set up a makeshift clothes line using the car's rooftop as an outdoor drying area. This scene is uncommon because people typically do their laundry indoors, in a dedicated space like a laundromat or a room in their home, rather than on top of a moving vehicle. Additionally, hanging clothes on the car could be potentially hazardous or illegal in some jurisdictions due to the risk of damaging the vehicle or causing accidents on the road.\n##"}]}
```

## For pipeline developers/technical description
see [DOCS.md](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/multimodal/DOCS.md)
