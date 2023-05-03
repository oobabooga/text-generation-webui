# LLaVA

## Description
Adds [LLaVA 13B](https://github.com/haotian-liu/LLaVA) multimodality support to text-generation-webui.

https://user-images.githubusercontent.com/3718215/233817203-69b57e77-0c55-4fd6-b742-3204bb13b8fc.mp4

## LLaVA-7B
7B version currently isn't supported. It will be supported if/when [more generic multimodality support](https://github.com/oobabooga/text-generation-webui/discussions/1687) gets implemented.

## Usage
To run this extension, download LLaVA weights, for example from [here](https://huggingface.co/wojtab/llava-13b-v0-4bit-128g) (note: it's a 4-bit [GPTQ quantization](https://github.com/oobabooga/text-generation-webui/tree/main/docs/GPTQ-models-(4-bit-mode).md), done on "old CUDA" branch), and then start server.py with `--extensions llava` argument.

Do note, that each image takes up 258 tokens, so adjust max_new_tokens to be at most 1700 (recommended value is between 200 to 500), so the images don't get truncated.

To send an image, just upload it to the extension field below chat, and send a prompt as always. The image will be added to the end of your message. If you wish to modify the placement, include a string `<image>` in your prompt.

Additionally, there is *Embed all images, not only the last one* checkbox. It modifies the image embeddings, by default (if it's unchecked), all but the most recent images have their embeddings empty, so they are not fed to the network. From initial testing, it seems as LLaVA considers the features in all images at the same time, so by default the extension skips previous images. If you want to include them anyway, just tick this checkbox.

## Extension config
This extension uses following parameters (from settings.json):
|Parameter|Description|
|---------|-----------|
|`llava-clip_bits`|Number of bits to load CLIP feature extractor in (either 32 or 16, default=32)|
|`llava-clip_device`|Torch device to run the extractor on, for example `cpu` or `cuda:0`, by default `cuda:0` if available|
|`llava-clip_repo`|Huggingface repository of CLIP model, `openai/clip-vit-large-patch14` by default. There should be no need to change it|
|`llava-projector_bits`|Number of bits to load CLIP->LLaMA feature projector in (either 32 or 16, default=32)|
|`llava-projector_device`|Torch device to run the CLIP->LLaMA feature projector on, for example `cpu` or `cuda:0`, by default `cuda:0` if available|
|`llava-projector_repo`|Huggingface repository of multimodal projector, `liuhaotian/LLaVA-13b-delta-v0` by default. There should be no need to change it|
|`llava-projector_filename`|The filename of multimodal projector weights, `mm_projector.bin` by default. There should be no need to change it|
|`llava-add_all_images_to_prompt`|Default value of "Embed all images, not only the last one" checkbox|
## Technical description

### Original LLaVA
The default LLaVA implementation uses modified `transformers` library, however this extension forgoes this requirement. The transformers are modified in LLaVA in such a way, that the entire LLaVA model gets loaded, and the inference now looks as follows:
```
images --> CLIP --> projector --> input embeddings for images --> | 
                                                                  | --> LLaMA
prompt -------------------------> input embeddings for text ----> |
```
The images are represented in the prompt by the following token IDs:
- 32000 - `<im_patch>` - placeholder token for embeddings from projector
- 32001 - `<im_start>` - token marking start of an image
- 32002 - `<im_end>` - token marking end of an image

By default, image will be represented as `<im_start><im_patch>*256<im_end>`. The input embeddings for an image are converted with a single linear layer of the projector, then they are placed instead of `<im_patch>` tokens.
The concatenated prompt then gets fed to fine-tuned LLaMA.

### In this extension

Using default transformers, they only load the LLaMA part of LLaVA, ignoring the added projector weights, and not loading CLIP. We then reconstruct the `images -> CLIP -> projector` pipeline ourselves, then concatenate the input embeddings, and feed it to LLaMA loaded by transformers. This allows us to use normal flow from webui to load this model, and just hijack the model input with additional features.
Splitting it to 3 separate models, allows us to configure each of them, and to move them to different devices(for example we can run CLIP+projector on CPU and LLaMA on GPU). Also, it enables us to use 4-bit GPTQ quantization for LLaVA, massively cutting down the VRAM requirement (it should be possible to fit on 12GB of VRAM with full context size by moving CLIP and projector to CPU).

### Usage through API

You can run the multimodal inference through API, by inputting the images to prompt. Images are embedded like so: `f'<img src="data:image/jpeg;base64,{img_str}">'`, where `img_str` is base-64 jpeg data. Python example:
```Python
import base64
import requests

CONTEXT = "You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. Follow the instructions carefully and explain your answers in detail.\n### Human: \nHi!\n### Assistant: \nHi there! How can I help you today?\n"

with open('extreme_ironing.jpg', 'rb') as f:
    img_str = base64.b64encode(f.read()).decode('utf-8')
    prompt = CONTEXT + f'### Human: \nWhat is unusual about this image: \n<img src="data:image/jpeg;base64,{img_str}">\n### Assistant: \n'
    print(requests.post('http://127.0.0.1:5000/api/v1/generate', json={'prompt': prompt, 'stopping_strings': ['\n###']}).json())
```
script output:
```Python
{'results': [{'text': "The unusual aspect of this image is that a man is standing on top of a yellow minivan while doing his laundry. He has set up a makeshift clothes line using the car's rooftop as an outdoor drying area. This scene is uncommon because people typically do their laundry indoors, in a dedicated space like a laundromat or a room in their home, rather than on top of a moving vehicle. Additionally, hanging clothes on the car could be potentially hazardous or illegal in some jurisdictions due to the risk of damaging the vehicle or causing accidents on the road.\n##"}]}
```