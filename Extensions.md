# Extensions

This web UI supports extensions. They are simply files under 

```
extensions/your_extension_name/script.py
```

which can be invoked with the 

```
--extension your_extension_name
```

command-line flag.

## Examples

|Extension|Description|
|---------|-----------|
|[character_bias](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/character_bias/script.py)| Just a very simple example that biases the bot's responses in chat mode.|
|[google_translate](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/google_translate/script.py)| Automatically translates inputs and outputs using Google Translate.|
|[silero_tts](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/silero_tts/script.py)| Text-to-speech extension using [Silero](https://github.com/snakers4/silero-models). When used in chat mode, it replaces the responses with an audio widget.|
|[send_pictures](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/send_pictures/script.py)| Creates an image upload field that can be used to send images to the bot in chat mode. Captions are automatically generated using BLIP. Author: [@SillyLossy](https://github.com/sillylossy).|

## How to write an extension

`script.py` has access to all variables in the UI through the `modules.shared` module, and it may define the following functions:

| Function        | Description |
|-------------|-------------|
| `def ui()` | Creates custom gradio elements when the UI is launched. | 
| `def input_modifier(string)`  | Modifies the input string before it enters the model. In chat mode, it is applied to the user message. Otherwise, it is applied to the entire prompt. |
| `def output_modifier(string)`  | Modifies the output string before it is presented in the UI. In chat mode, it is applied to the bot's reply. Otherwise, it is applied to the entire output. |
| `def bot_prefix_modifier(string)`  | Applied in chat mode to the prefix for the bot's reply (more on that below). |
| `def custom_generate_chat_prompt(...)` | Overrides the prompt generator in chat mode. |

Additionally, the script may define two special global variables. 

#### `params` dictionary

```python
params = {
    "language string": "ja",
}
```

This dicionary can be used to make the extension parameters customizable by adding entries to a `settings.json` file like this:

```python
google_translate-language string: "fr",
``` 

#### `input_hijack` dictionary

```python
input_hijack = {
    'state': False,
    'value': ["", ""]
}
```
This is only relevant in chat mode. If your extension sets `input_hijack['state']` to `True` at any moment, the next call for `modules.chat.chatbot_wrapper` will use the vales inside `input_hijack['value']` as the user input for the text generation. See the `send_pictures` extension above for an example.

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

