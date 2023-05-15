# Technical description of multimodal extension

## Working principle
Multimodality extension does most of the stuff which is required for any image input:

- adds the UI
- saves the images as base64 JPEGs to history
- provides the hooks to the UI
- if there are images in the prompt, it:
    - splits the prompt to text and image parts
    - adds image start/end markers to text parts, then encodes and embeds the text parts
    - calls the vision pipeline to embed the images
    - stitches the embeddings together, and returns them to text generation
- loads the appropriate vision pipeline, selected either from model name, or by specifying --multimodal-pipeline parameter

Now, for the pipelines, they:

- load the required vision models
- return some consts, for example the number of tokens taken up by image
- and most importantly: return the embeddings for LLM, given a list of images

## Prompts/history

To save images in prompt/history, this extension is using a base64 JPEG, wrapped in a HTML tag, like so:
```
<img src="data:image/jpeg;base64,{img_str}">
```
where `{img_str}` is the actual image data. This format makes displaying them in the UI for free. Do note, that this format is required to be exactly the same, the regex used to find the images is: `<img src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)">`.

## LLM input
To describe the input, let's see it on an example prompt:
```
text1<image1>text2<image2>text3
```
where `textN` is N-th text, `<imageN>` is N-th image, in HTML format specified above.

**The first step is to split the prompt into image/text parts**, so we get:
```
['text1', '<image1>', 'text2', '<image2>', 'text3']
```
this is done in `MultimodalEmbedder._split_prompt(...)` function, which returns a list of `PromptPart`s - dataclasses wrapping the separate parts.

This function also appends the image start/end markers to text, which are provided by `AbstractMultimodalPipeline.image_start()` / `AbstractMultimodalPipeline.image_end()` functions. If image start is `<Img>`, and end is `</Img>`, this function will return:
```
['text1<Img>', '<image1>', '</Img>text2<Img>', '<image2>', '</Img>text3']
```

**The returned prompt parts are then turned into token embeddings.**

First, they are modified to token IDs, for the text it is done using standard `modules.text_generation.encode()` function, and for the images the returned token IDs are changed to placeholders. The placeholder is a list of `N` times `placeholder token id`, where `N` is specified using `AbstractMultimodalPipeline.num_image_embeds()`, and placeholder token IDs using  `AbstractMultimodalPipeline.placeholder_token_id()`.

Now, based on the token IDs, the prompt might get truncated, especially if `max_new_tokens` are unreasonably high. Unfortunately, it can't be done simply, just by trimming the prompt to be short enough. This way will lead to sometimes splitting the prompt in the middle of an image embedding, which usually breaks the generation. Therefore, in this case, the entire image needs to be removed from input. This is done inside `MultimodalEmbedder._encode_text(...)` function.

**After the tokenization, the tokens need to get embedded**, the text and images are once again treated separately.

The text parts are turned to embeddings, using `AbstractMultimodalPipeline.embed_tokens(...)` function. It uses standard embedding function from the model, but to support many LLMs, the actual function is returned by the pipeline (as it might be different for different LLMs), for LLaMA it is `shared.model.model.embed_tokens(...)`.

The image parts are turned to embeddings, using `AbstractMultimodalPipeline.embed_images(...)` function. This function is specific for a given pipeline, it takes the images as input, forwards them through vision model/projector, and returns the embeddings.

**Now, the returned embeddings are stitched together**, using `torch.cat()`, this is creating the final input to the LLM.

## Pipelines

All of the pipelines should subclass `AbstractMultimodalPipeline` class. The idea is to allow for new pipelines to be added in the same way as user extensions - git clone into `extensions/multimodal/pipelines`.

The pipelines are the description of the vision part, containing vision model/multimodal projector. All of the pipelines should have an unique `name()`, which is then selected by user, in `--multimodal-pipeline` CLI argument. For an example, see `pipelines/llava/llava.py`.

## Pipeline modules

Pipelines are organized into "pipeline modules" - subdirectories in `pipelines` directory. The pipeline modules should contain a file called `pipelines.py`, that should contain the following fields:
- `available_pipelines: List[str]` - list of pipelines provided by this module, shown as the list of available pipelines to the user
- `def get_pipeline(name: str, params: dict) -> Optional[AbstractMultimodalPipeline]`: - a function to get a concrete pipeline by `name`, if `name` doesn't match any, should return `None`. `params` is the user settings for multimodal extension
- `def get_pipeline_from_model_name(model_name: str, params: dict) -> Optional[AbstractMultimodalPipeline]`: - a function to get a pipeline from `model_name`, should be eager to return `None`, unless the determination can be done clearly (for example: minigpt-4 bases on vicuna - it should never return the pipeline, but llava can, as it has its own specific LLM finetune)

**NOTE**: A pipeline module should lazy-import the pipelines only when necessary, and it should keep its imports to minimum

## Pipeline params

The pipelines will get the extension `params` in the constructor. They should honor the following fields:
- `vision_device` - string, specifying `torch.device` to run the vision model (CLIP/ViT) on
- `vision_bits` - int, number of fp bits to load the vision model(s) in
- `projector_device` - string, specifying `torch.device` to run the projector models (Linear layers, QFormer, etc.) on
- `projector_bits` - int, number of fp bits to load the projector models in

As a helper, `AbstractMultimodalPipeline` has `_get_device(self, setting_name: str, params: dict)` and `_get_dtype(self, setting_name: str, params: dict)` helper functions, which parse string/int and return `torch.device` / `torch.dtype`.
