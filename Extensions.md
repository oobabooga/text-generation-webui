This web UI supports extensions. They are simply files under 

```
extensions/your_extension_name/script.py
```

which can be invoked with the 

```
--extension your_extension_name
```

command-line flag.

## Gallery

|Extension|Description|
|---------|-----------|
|[character_bias](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/character_bias/script.py)| Just a very simple example that biases the bot's responses in chat mode.|
|[google_translate](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/google_translate/script.py)| Automatically translates inputs and outputs using Google Translate.|
|[silero_tts](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/silero_tts/script.py)| Text-to-speech extension using [Silero](https://github.com/snakers4/silero-models). When used in chat mode, it replaces the responses with an audio widget. Authors: me and [@xanthousm](https://github.com/xanthousm). |
|[elevenlabs_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/elevenlabs_tts)| Text-to-speech extension using the [ElevenLabs](https://beta.elevenlabs.io/) API. You need an API key to use it. Author: [@MetaIX](https://github.com/MetaIX). |
|[send_pictures](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/send_pictures/script.py)| Creates an image upload field that can be used to send images to the bot in chat mode. Captions are automatically generated using BLIP. Author: [@SillyLossy](https://github.com/sillylossy).|
|[gallery](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/gallery/script.py)| Creates a gallery with the chat characters and their pictures. |
|[llama_prompts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/llama_prompts)| Creates a dropdown menu with a selection of interesting prompts to choose from. Based on [devbrones/llama-prompts](https://github.com/devbrones/llama-prompts). Only applies in regular or `--notebook` mode. |
|[api](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/api)| Creates an API similar to the one provided by KoboldAI. Works with TavernAI: start the web UI with `python server.py --no-stream --extensions api` and set the API URL to `http://127.0.0.1:5000/api`. Author: [@mayaeary](https://github.com/mayaeary).|
|[whisper_stt](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/whisper_stt)| Allows you to enter your inputs in chat mode using your microphone. Author: [@EliasVincent](https://github.com/EliasVincent).|
|[sd_api_pictures](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/sd_api_pictures)| Allows you to request pictures from the bot in chat mode, which will be generated using the AUTOMATIC1111 Stable Diffusion API. See examples [here](https://github.com/oobabooga/text-generation-webui/pull/309). Author: [@Brawlence](https://github.com/Brawlence).|
|[long_term_memory](https://github.com/wawawario2/long_term_memory) | A sophisticated extension that creates a long term memory for bots in chat mode. Author: [@wawawario2](https://github.com/wawawario2). |

## How to write an extension

`script.py` has access to all variables in the UI through the `modules.shared` module, and it may define the following functions:

| Function        | Description |
|-------------|-------------|
| `def ui()` | Creates custom gradio elements when the UI is launched. | 
| `def input_modifier(string)`  | Modifies the input string before it enters the model. In chat mode, it is applied to the user message. Otherwise, it is applied to the entire prompt. |
| `def output_modifier(string)`  | Modifies the output string before it is presented in the UI. In chat mode, it is applied to the bot's reply. Otherwise, it is applied to the entire output. |
| `def bot_prefix_modifier(string)`  | Applied in chat mode to the prefix for the bot's reply (more on that below). |
| `def custom_generate_chat_prompt(...)` | Overrides the prompt generator in chat mode. |

Additionally, the script may define two special global variables:

#### `params` dictionary

```python
params = {
    "language string": "ja",
}
```

This dicionary can be used to make the extension parameters customizable by adding entries to a `settings.json` file like this:

```python
"google_translate-language string": "fr",
``` 

#### `input_hijack` dictionary

```python
input_hijack = {
    'state': False,
    'value': ["", ""]
}
```
This is only relevant in chat mode. If your extension sets `input_hijack['state']` to `True` at any moment, the next call to `modules.chat.chatbot_wrapper` will use the vales inside `input_hijack['value']` as the user input for text generation. See the `send_pictures` extension above for an example.

## The `bot_prefix_modifier`

In chat mode, this function modifies the prefix for a new bot message. For instance, if your bot is named `Marie Antoinette`, the default prefix for a new message will be

```
Marie Antoinette:
```

Using `bot_prefix_modifier`, you can change it to:

```
Marie Antoinette: *I am very enthusiastic*
```
 
Marie Antoinette will become very enthusiastic in all her messages.

## Using multiple extensions at the same time

In order to use your extension, you must start the web UI with the `--extensions` flag followed by the name of your extension (the folder under `text-generation-webui/extension` where `script.py` resides).

You can activate more than one extension at a time by providing their names separated by spaces. The input, output and bot prefix modifiers will be applied in the specified order. For `custom_generate_chat_prompt`, only the first declaration encountered will be used and the rest will be ignored.

```
python server.py --extensions enthusiasm translate # First apply enthusiasm, then translate
python server.py --extensions translate enthusiasm # First apply translate, then enthusiasm
```

## `custom_generate_chat_prompt` example

Below is an extension that just reproduces the default prompt generator in `modules/chat.py`. You can modify it freely to come up with your own prompts in chat mode.

```python
import gradio as gr
import modules.shared as shared
from modules.chat import clean_chat_message
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length


def custom_generate_chat_prompt(user_input, max_new_tokens, name1, name2, context, chat_prompt_size, impersonate=False):
    user_input = clean_chat_message(user_input)
    rows = [f"{context.strip()}\n"]

    if shared.soft_prompt:
       chat_prompt_size -= shared.soft_prompt_tensor.shape[1]
    max_length = min(get_max_prompt_length(max_new_tokens), chat_prompt_size)

    i = len(shared.history['internal'])-1
    while i >= 0 and len(encode(''.join(rows), max_new_tokens)[0]) < max_length:
        rows.insert(1, f"{name2}: {shared.history['internal'][i][1].strip()}\n")
        if not (shared.history['internal'][i][0] == '<|BEGIN-VISIBLE-CHAT|>'):
            rows.insert(1, f"{name1}: {shared.history['internal'][i][0].strip()}\n")
        i -= 1

    if not impersonate:
        rows.append(f"{name1}: {user_input}\n")
        rows.append(apply_extensions(f"{name2}:", "bot_prefix"))
        limit = 3
    else:
        rows.append(f"{name1}:")
        limit = 2

    while len(rows) > limit and len(encode(''.join(rows), max_new_tokens)[0]) >= max_length:
        rows.pop(1)

    prompt = ''.join(rows)
    return prompt

def ui():
    pass
```
