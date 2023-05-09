This web UI supports extensions. They are simply files under 

```
extensions/your_extension_name/script.py
```

which can be invoked with the 

```
--extension your_extension_name
```

command-line flag.

## [text-generation-webui-extensions](https://github.com/oobabooga/text-generation-webui-extensions)

The link above contains a directory of user extensions for text-generation-webui.

If you create an extension, you are welcome to host it in a GitHub repository and submit it to the list above.

## Built-in extensions

Most of these have been created by the extremely talented contributors that you can find here: [contributors](https://github.com/oobabooga/text-generation-webui/graphs/contributors?from=2022-12-18&to=&type=a).

|Extension|Description|
|---------|-----------|
|[api](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/api)| Creates an API with two endpoints, one for streaming at `/api/v1/stream` port 5005 and another for blocking at `/api/v1/generate` por 5000. This is the main API for this web UI. |
|[google_translate](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/google_translate)| Automatically translates inputs and outputs using Google Translate.|
|[character_bias](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/character_bias)| Just a very simple example that biases the bot's responses in chat mode.|
|[gallery](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/gallery/)| Creates a gallery with the chat characters and their pictures. |
|[silero_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/silero_tts)| Text-to-speech extension using [Silero](https://github.com/snakers4/silero-models). When used in chat mode, it replaces the responses with an audio widget. |
|[elevenlabs_tts](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/elevenlabs_tts)| Text-to-speech extension using the [ElevenLabs](https://beta.elevenlabs.io/) API. You need an API key to use it. |
|[send_pictures](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/send_pictures/)| Creates an image upload field that can be used to send images to the bot in chat mode. Captions are automatically generated using BLIP. |
|[whisper_stt](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/whisper_stt)| Allows you to enter your inputs in chat mode using your microphone. |
|[sd_api_pictures](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/sd_api_pictures)| Allows you to request pictures from the bot in chat mode, which will be generated using the AUTOMATIC1111 Stable Diffusion API. See examples [here](https://github.com/oobabooga/text-generation-webui/pull/309). |
|[llava](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/llava) | Adds LLaVA multimodal model support. For detailed description see [README.md](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/llava/README.md) in the extension directory. |
|[openai](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)| Creates an API that mimics the OpenAI API and can be used as a drop-in replacement. |
|[superbooga](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/superbooga)| An extension that uses ChromaDB to create an arbitrarily large pseudocontext, taking as input text files, URLs, or pasted text. Based on https://github.com/kaiokendev/superbig. |

## How to write an extension

`script.py` has access to all variables in the UI through the `modules.shared` module, and it may define the following functions:

| Function        | Description |
|-------------|-------------|
| `def ui()` | Creates custom gradio elements when the UI is launched. | 
| `def input_modifier(string)`  | Modifies the input string before it enters the model. In chat mode, it is applied to the user message. Otherwise, it is applied to the entire prompt. |
| `def output_modifier(string)`  | Modifies the output string before it is presented in the UI. In chat mode, it is applied to the bot's reply. Otherwise, it is applied to the entire output. |
| `def state_modifier(state)`  | Modifies the dictionary containing the input parameters before it is used by the text generation functions. |
| `def bot_prefix_modifier(string)`  | Applied in chat mode to the prefix for the bot's reply (more on that below). |
| `def custom_generate_reply(...)` | Overrides the main text generation function. |
| `def custom_generate_chat_prompt(...)` | Overrides the prompt generator in chat mode. |
| `def tokenizer_modifier(state, prompt, input_ids, input_embeds)` | Modifies the `input_ids`/`input_embeds` fed to the model. Should return `prompt`, `input_ids`, `input_embeds`. See `llava` extension for an example |

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
This is only relevant in chat mode. If your extension sets `input_hijack['state']` to `True` at any moment, the next call to `modules.chat.chatbot_wrapper` will use the values inside `input_hijack['value']` as the user input for text generation. See the `send_pictures` extension above for an example. 

Additionally, your extension can set the value to be a callback, in the form of `def cb(text: str, visible_text: str) -> [str, str]`. See the `llava` extension above for an example.

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

## `custom_generate_reply` example

Once defined in a `script.py`, this function is executed in place of the main generation functions. You can use it to connect the web UI to an external API, or to load a custom model that is not supported yet.

Note that in chat mode, this function must only return the new text, whereas in other modes it must return the original prompt + the new text.

```python
import datetime

def custom_generate_reply(question, original_question, seed, state, eos_token, stopping_strings):
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
def custom_generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs['impersonate'] if 'impersonate' in kwargs else False
    _continue = kwargs['_continue'] if '_continue' in kwargs else False
    also_return_rows = kwargs['also_return_rows'] if 'also_return_rows' in kwargs else False
    is_instruct = state['mode'] == 'instruct'
    rows = [state['context'] if is_instruct else f"{state['context'].strip()}\n"]
    min_rows = 3

    # Finding the maximum prompt size
    chat_prompt_size = state['chat_prompt_size']
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]

    max_length = min(get_max_prompt_length(state), chat_prompt_size)

    # Building the turn templates
    if 'turn_template' not in state or state['turn_template'] == '':
        if is_instruct:
            template = '<|user|>\n<|user-message|>\n<|bot|>\n<|bot-message|>\n'
        else:
            template = '<|user|>: <|user-message|>\n<|bot|>: <|bot-message|>\n'
    else:
        template = state['turn_template'].replace(r'\n', '\n')

    replacements = {
        '<|user|>': state['name1'].strip(),
        '<|bot|>': state['name2'].strip(),
    }

    user_turn = replace_all(template.split('<|bot|>')[0], replacements)
    bot_turn = replace_all('<|bot|>' + template.split('<|bot|>')[1], replacements)
    user_turn_stripped = replace_all(user_turn.split('<|user-message|>')[0], replacements)
    bot_turn_stripped = replace_all(bot_turn.split('<|bot-message|>')[0], replacements)

    # Building the prompt
    i = len(shared.history['internal']) - 1
    while i >= 0 and len(encode(''.join(rows))[0]) < max_length:
        if _continue and i == len(shared.history['internal']) - 1:
            rows.insert(1, bot_turn_stripped + shared.history['internal'][i][1].strip())
        else:
            rows.insert(1, bot_turn.replace('<|bot-message|>', shared.history['internal'][i][1].strip()))

        string = shared.history['internal'][i][0]
        if string not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            rows.insert(1, replace_all(user_turn, {'<|user-message|>': string.strip(), '<|round|>': str(i)}))

        i -= 1

    if impersonate:
        min_rows = 2
        rows.append(user_turn_stripped.rstrip(' '))
    elif not _continue:
        # Adding the user message
        if len(user_input) > 0:
            rows.append(replace_all(user_turn, {'<|user-message|>': user_input.strip(), '<|round|>': str(len(shared.history["internal"]))}))

        # Adding the Character prefix
        rows.append(apply_extensions("bot_prefix", bot_turn_stripped.rstrip(' ')))

    while len(rows) > min_rows and len(encode(''.join(rows))[0]) >= max_length:
        rows.pop(1)

    prompt = ''.join(rows)
    if also_return_rows:
        return prompt, rows
    else:
        return prompt
```
