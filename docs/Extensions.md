Extensions are defined by files named `script.py` inside subfolders of `text-generation-webui/extensions`. They are loaded at startup if specified with the `--extensions` flag.

For instance, `extensions/silero_tts/script.py` gets loaded with `python server.py --extensions silero_tts`.

## [text-generation-webui-extensions](https://github.com/oobabooga/text-generation-webui-extensions)

The link above contains a directory of user extensions for text-generation-webui.

If you create an extension, you are welcome to host it in a GitHub repository and submit it to the list above.

## Built-in extensions

Most of these have been created by the extremely talented contributors that you can find here: [contributors](https://github.com/oobabooga/text-generation-webui/graphs/contributors?from=2022-12-18&to=&type=a).

|Extension|Description|
|---------|-----------|
|[api](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/api)| Creates an API with two endpoints, one for streaming at `/api/v1/stream` port 5005 and another for blocking at `/api/v1/generate` port 5000. This is the main API for this web UI. |
|[google_translate](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/google_translate)| Automatically translates inputs and outputs using Google Translate.|
|[character_bias](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/character_bias)| Just a very simple example that biases the bot's responses in chat mode.|
|[gallery](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/gallery/)| Creates a gallery with the chat characters and their pictures. |
|[silero_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/silero_tts)| Text-to-speech extension using [Silero](https://github.com/snakers4/silero-models). When used in chat mode, it replaces the responses with an audio widget. |
|[elevenlabs_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/elevenlabs_tts)| Text-to-speech extension using the [ElevenLabs](https://beta.elevenlabs.io/) API. You need an API key to use it. |
|[send_pictures](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/send_pictures/)| Creates an image upload field that can be used to send images to the bot in chat mode. Captions are automatically generated using BLIP. |
|[whisper_stt](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/whisper_stt)| Allows you to enter your inputs in chat mode using your microphone. |
|[sd_api_pictures](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/sd_api_pictures)| Allows you to request pictures from the bot in chat mode, which will be generated using the AUTOMATIC1111 Stable Diffusion API. See examples [here](https://github.com/oobabooga/text-generation-webui/pull/309). |
|[multimodal](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal) | Adds multimodality support (text+images). For a detailed description see [README.md](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal/README.md) in the extension directory. |
|[openai](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)| Creates an API that mimics the OpenAI API and can be used as a drop-in replacement. |
|[superbooga](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/superbooga)| An extension that uses ChromaDB to create an arbitrarily large pseudocontext, taking as input text files, URLs, or pasted text. Based on https://github.com/kaiokendev/superbig. |

## How to write an extension

script.py may define the special functions and variables below.

#### Predefined functions

| Function        | Description |
|-------------|-------------|
| `def ui()` | Creates custom gradio elements when the UI is launched. | 
| `def custom_css()` | Returns custom CSS as a string. It is applied whenever the web UI is loaded. |
| `def custom_js()` | Same as above but for javascript. |
| `def input_modifier(string, state)`  | Modifies the input string before it enters the model. In chat mode, it is applied to the user message. Otherwise, it is applied to the entire prompt. |
| `def output_modifier(string, state)`  | Modifies the output string before it is presented in the UI. In chat mode, it is applied to the bot's reply. Otherwise, it is applied to the entire output. |
| `def bot_prefix_modifier(string, state)`  | Applied in chat mode to the prefix for the bot's reply. |
| `def state_modifier(state)`  | Modifies the dictionary containing the UI input parameters before it is used by the text generation functions. |
| `def history_modifier(history)`  | Modifies the chat history before the text generation in chat mode begins. |
| `def custom_generate_reply(...)` | Overrides the main text generation function. |
| `def custom_generate_chat_prompt(...)` | Overrides the prompt generator in chat mode. |
| `def tokenizer_modifier(state, prompt, input_ids, input_embeds)` | Modifies the `input_ids`/`input_embeds` fed to the model. Should return `prompt`, `input_ids`, `input_embeds`. See the `multimodal` extension for an example. |
| `def custom_tokenized_length(prompt)` | Used in conjunction with `tokenizer_modifier`, returns the length in tokens of `prompt`. See the `multimodal` extension for an example. |

#### `params` dictionary

In this dictionary, `display_name` is used to define the displayed name of the extension in the UI, and `is_tab` is used to define whether the extension should appear in a new tab. By default, extensions appear at the bottom of the "Text generation" tab.

Example:

```python
params = {
    "display_name": "Google Translate",
    "is_tab": True,
}
```

Additionally, `params` may contain variables that you want to be customizable through a `settings.json` file. For instance, assuming the extension is in `extensions/google_translate`, the variable `language string` in

```python
params = {
    "display_name": "Google Translate",
    "is_tab": True,
    "language string": "jp"
}
```

can be customized by adding a key called `google_translate-language string` to `settings.json`:

```python
"google_translate-language string": "fr",
``` 

That is, the syntax is `extension_name-variable_name`.

#### `input_hijack` dictionary

```python
input_hijack = {
    'state': False,
    'value': ["", ""]
}
```
This is only used in chat mode. If your extension sets `input_hijack['state'] = True` at any moment, the next call to `modules.chat.chatbot_wrapper` will use the values inside `input_hijack['value']` as the user input for text generation. See the `send_pictures` extension above for an example. 

Additionally, your extension can set the value to be a callback in the form of `def cb(text: str, visible_text: str) -> [str, str]`. See the `multimodal` extension above for an example.

## Using multiple extensions at the same time

In order to use your extension, you must start the web UI with the `--extensions` flag followed by the name of your extension (the folder under `text-generation-webui/extension` where `script.py` resides).

You can activate more than one extension at a time by providing their names separated by spaces. The input, output, and bot prefix modifiers will be applied in the specified order. 


```
python server.py --extensions enthusiasm translate # First apply enthusiasm, then translate
python server.py --extensions translate enthusiasm # First apply translate, then enthusiasm
```

Do note, that for:
- `custom_generate_chat_prompt`
- `custom_generate_reply`
- `tokenizer_modifier`
- `custom_tokenized_length`

only the first declaration encountered will be used and the rest will be ignored. 

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

## `custom_generate_reply` example

Once defined in a `script.py`, this function is executed in place of the main generation functions. You can use it to connect the web UI to an external API, or to load a custom model that is not supported yet.

Note that in chat mode, this function must only return the new text, whereas in other modes it must return the original prompt + the new text.

```python
import datetime

def custom_generate_reply(question, original_question, seed, state, stopping_strings):
    cumulative = ''
    for i in range(10):
        cumulative += f"Counting: {i}...\n"
        yield cumulative

    cumulative += f"Done! {str(datetime.datetime.now())}"
    yield cumulative
```

## `custom_generate_chat_prompt` example

Below is an extension that just reproduces the default prompt generator in `modules/chat.py`. You can modify it freely to come up with your own prompts in chat mode.

```python
from modules import chat

def custom_generate_chat_prompt(user_input, state, **kwargs):
    
    # Do something with kwargs['history'] or state

    return chat.generate_chat_prompt(user_input, state, **kwargs)
```
