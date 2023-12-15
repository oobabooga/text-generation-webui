# Extensions

Extensions are defined by files named `script.py` inside subfolders of `text-generation-webui/extensions`. They are loaded at startup if the folder name is specified after the `--extensions` flag.

For instance, `extensions/silero_tts/script.py` gets loaded with `python server.py --extensions silero_tts`.

## [text-generation-webui-extensions](https://github.com/oobabooga/text-generation-webui-extensions)

The repository above contains a directory of user extensions.

If you create an extension, you are welcome to host it in a GitHub repository and submit a PR adding it to the list.

## Built-in extensions

|Extension|Description|
|---------|-----------|
|[openai](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)| Creates an API that mimics the OpenAI API and can be used as a drop-in replacement. |
|[multimodal](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal) | Adds multimodality support (text+images). For a detailed description see [README.md](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal/README.md) in the extension directory. |
|[google_translate](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/google_translate)| Automatically translates inputs and outputs using Google Translate.|
|[silero_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/silero_tts)| Text-to-speech extension using [Silero](https://github.com/snakers4/silero-models). When used in chat mode, responses are replaced with an audio widget. |
|[whisper_stt](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/whisper_stt)| Allows you to enter your inputs in chat mode using your microphone. |
|[sd_api_pictures](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/sd_api_pictures)| Allows you to request pictures from the bot in chat mode, which will be generated using the AUTOMATIC1111 Stable Diffusion API. See examples [here](https://github.com/oobabooga/text-generation-webui/pull/309). |
|[character_bias](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/character_bias)| Just a very simple example that adds a hidden string at the beginning of the bot's reply in chat mode. |
|[send_pictures](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/send_pictures/)| Creates an image upload field that can be used to send images to the bot in chat mode. Captions are automatically generated using BLIP. |
|[gallery](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/gallery/)| Creates a gallery with the chat characters and their pictures. |
|[superbooga](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/superbooga)| An extension that uses ChromaDB to create an arbitrarily large pseudocontext, taking as input text files, URLs, or pasted text. Based on https://github.com/kaiokendev/superbig. |
|[ngrok](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/ngrok)| Allows you to access the web UI remotely using the ngrok reverse tunnel service (free). It's an alternative to the built-in Gradio `--share` feature. |
|[perplexity_colors](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/perplexity_colors)| Colors each token in the output text by its associated probability, as derived from the model logits. |

## How to write an extension

The extensions framework is based on special functions and variables that you can define in `script.py`. The functions are the following:

| Function        | Description |
|-------------|-------------|
| `def setup()` | Is executed when the extension gets imported. |
| `def ui()` | Creates custom gradio elements when the UI is launched. | 
| `def custom_css()` | Returns custom CSS as a string. It is applied whenever the web UI is loaded. |
| `def custom_js()` | Same as above but for javascript. |
| `def input_modifier(string, state, is_chat=False)`  | Modifies the input string before it enters the model. In chat mode, it is applied to the user message. Otherwise, it is applied to the entire prompt. |
| `def output_modifier(string, state, is_chat=False)`  | Modifies the output string before it is presented in the UI. In chat mode, it is applied to the bot's reply. Otherwise, it is applied to the entire output. |
| `def chat_input_modifier(text, visible_text, state)` | Modifies both the visible and internal inputs in chat mode. Can be used to hijack the chat input with custom content. |
| `def bot_prefix_modifier(string, state)`  | Applied in chat mode to the prefix for the bot's reply. |
| `def state_modifier(state)`  | Modifies the dictionary containing the UI input parameters before it is used by the text generation functions. |
| `def history_modifier(history)`  | Modifies the chat history before the text generation in chat mode begins. |
| `def custom_generate_reply(...)` | Overrides the main text generation function. |
| `def custom_generate_chat_prompt(...)` | Overrides the prompt generator in chat mode. |
| `def tokenizer_modifier(state, prompt, input_ids, input_embeds)` | Modifies the `input_ids`/`input_embeds` fed to the model. Should return `prompt`, `input_ids`, `input_embeds`. See the `multimodal` extension for an example. |
| `def custom_tokenized_length(prompt)` | Used in conjunction with `tokenizer_modifier`, returns the length in tokens of `prompt`. See the `multimodal` extension for an example. |

Additionally, you can define a special `params` dictionary. In it, the `display_name` key is used to define the displayed name of the extension in the UI, and the `is_tab` key is used to define whether the extension should appear in a new tab. By default, extensions appear at the bottom of the "Text generation" tab.

Example:

```python
params = {
    "display_name": "Google Translate",
    "is_tab": True,
}
```

The `params` dict may also contain variables that you want to be customizable through a `settings.yaml` file. For instance, assuming the extension is in `extensions/google_translate`, the variable `language string` in

```python
params = {
    "display_name": "Google Translate",
    "is_tab": True,
    "language string": "jp"
}
```

can be customized by adding a key called `google_translate-language string` to `settings.yaml`:

```python
google_translate-language string: 'fr'
``` 

That is, the syntax for the key is `extension_name-variable_name`.

## Using multiple extensions at the same time

You can activate more than one extension at a time by providing their names separated by spaces after `--extensions`. The input, output, and bot prefix modifiers will be applied in the specified order. 

Example:

```
python server.py --extensions enthusiasm translate # First apply enthusiasm, then translate
python server.py --extensions translate enthusiasm # First apply translate, then enthusiasm
```

Do note, that for:
- `custom_generate_chat_prompt`
- `custom_generate_reply`
- `custom_tokenized_length`

only the first declaration encountered will be used and the rest will be ignored. 

## A full example

The source code below can be found at [extensions/example/script.py](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/example/script.py).

```python
"""
An example of extension. It does nothing, but you can add transformations
before the return statements to customize the webui behavior.

Starting from history_modifier and ending in output_modifier, the
functions are declared in the same order that they are called at
generation time.
"""

import gradio as gr
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

params = {
    "display_name": "Example Extension",
    "is_tab": False,
}

class MyLogits(LogitsProcessor):
    """
    Manipulates the probabilities for the next token before it gets sampled.
    Used in the logits_processor_modifier function below.
    """
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores

def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history

def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    return state

def chat_input_modifier(text, visible_text, state):
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text

def input_modifier(string, state, is_chat=False):
    """
    In default/notebook modes, modifies the whole prompt.

    In chat mode, it is the same as chat_input_modifier but only applied
    to "text", here called "string", and not to "visible_text".
    """
    return string

def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifies the input ids and embeds.
    Used by the multimodal extension to put image embeddings in the prompt.
    Only used by loaders that use the transformers library for sampling.
    """
    return prompt, input_ids, input_embeds

def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    return string

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result

def custom_css():
    """
    Returns a CSS string that gets appended to the CSS for the webui.
    """
    return ''

def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ''

def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    pass

def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """
    pass
```
