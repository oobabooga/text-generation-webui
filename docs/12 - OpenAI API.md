## OpenAI compatible API

This project includes an API compatible with multiple OpenAI endpoints, including Chat and Completions.

If you did not use the one-click installers, you may need to install the requirements first:

```
pip install -r extensions/openai/requirements.txt
```

### Starting the API

Add `--extensions openai` to your command-line flags.

* To create a public Cloudflare URL, add the `--public-api` flag.
* To listen on your local network, add the `--listen` flag.
* To change the port, which is 5000 by default, use `--port 1234` (change 1234 to your desired port number).
* To use SSL, add `--ssl-keyfile key.pem --ssl-certfile cert.pem`. Note that it doesn't work with `--public-api`.

#### Environment variables

The following environment variables can be used (they take precendence over everything else):

| Variable Name          | Description                                                                                        | Example Value              |
|------------------------|------------------------------------|----------------------------|
| `OPENEDAI_PORT`           | Port number         |             5000               |
| `OPENEDAI_CERT_PATH`      | SSL certificate file path         |            cert.pem                |
| `OPENEDAI_KEY_PATH`       | SSL key file path                    |             key.pem               |
| `OPENEDAI_DEBUG`          | Enable debugging (set to 1)    | 1                          |
| `SD_WEBUI_URL`           | WebUI URL (used by endpoint) | http://127.0.0.1:7861 |
| `OPENEDAI_EMBEDDING_MODEL` | Embedding model (if applicable) |          all-mpnet-base-v2                  |
| `OPENEDAI_EMBEDDING_DEVICE` | Embedding device (if applicable) |           cuda                 |

#### Persistent settings in `settings.yaml`

You can also set default values by adding these lines to your `settings.yaml` file:

```
openai-embedding_device: cuda
openai-embedding_model: all-mpnet-base-v2
openai-sd_webui_url: http://127.0.0.1:7861
openai-debug: 1
```

### Examples

#### Client Application Setup


You can usually force an application that uses the OpenAI API to connect to the local API by using the following environment variables:

```shell
OPENAI_API_HOST=http://127.0.0.1:5000
```

or

```shell
OPENAI_API_KEY=sk-111111111111111111111111111111111111111111111111
OPENAI_API_BASE=http://127.0.0.1:500/v1
```

With the [official python openai client](https://github.com/openai/openai-python), set the `OPENAI_API_BASE` environment variables:

```shell
# Sample .env file:
OPENAI_API_KEY=sk-111111111111111111111111111111111111111111111111
OPENAI_API_BASE=http://0.0.0.0:5001/v1
```

If needed, replace 127.0.0.1 with the IP/port of your server.

If using .env files to save the `OPENAI_API_BASE` and `OPENAI_API_KEY` variables, make sure the .env file is loaded before the openai module is imported:

```python
from dotenv import load_dotenv
load_dotenv() # make sure the environment variables are set before import
import openai
```

With the [official Node.js openai client](https://github.com/openai/openai-node) it is slightly more more complex because the environment variables are not used by default, so small source code changes may be required to use the environment variables, like so:

```js
const openai = OpenAI(
  Configuration({
    apiKey: process.env.OPENAI_API_KEY,
    basePath: process.env.OPENAI_API_BASE
  })
);
```

For apps made with the [chatgpt-api Node.js client library](https://github.com/transitive-bullshit/chatgpt-api):

```js
const api = new ChatGPTAPI({
  apiKey: process.env.OPENAI_API_KEY,
  apiBaseUrl: process.env.OPENAI_API_BASE
});
```
### Embeddings (alpha)

Embeddings requires `sentence-transformers` installed, but chat and completions will function without it loaded. The embeddings endpoint is currently using the HuggingFace model: `sentence-transformers/all-mpnet-base-v2` for embeddings. This produces 768 dimensional embeddings (the same as the text-davinci-002 embeddings), which is different from OpenAI's current default `text-embedding-ada-002` model which produces 1536 dimensional embeddings. The model is small-ish and fast-ish. This model and embedding size may change in the future.

| model name             | dimensions | input max tokens | speed | size | Avg. performance |
| ---------------------- | ---------- | ---------------- | ----- | ---- | ---------------- |
| text-embedding-ada-002 | 1536       | 8192             | -     | -    | -                |
| text-davinci-002       | 768        | 2046             | -     | -    | -                |
| all-mpnet-base-v2      | 768        | 384              | 2800  | 420M | 63.3             |
| all-MiniLM-L6-v2       | 384        | 256              | 14200 | 80M  | 58.8             |

In short, the all-MiniLM-L6-v2 model is 5x faster, 5x smaller ram, 2x smaller storage, and still offers good quality. Stats from (https://www.sbert.net/docs/pretrained_models.html). To change the model from the default you can set the environment variable `OPENEDAI_EMBEDDING_MODEL`, ex. "OPENEDAI_EMBEDDING_MODEL=all-MiniLM-L6-v2".

Warning: You cannot mix embeddings from different models even if they have the same dimensions. They are not comparable.

### API Documentation & Examples

The OpenAI API is well documented, you can view the documentation here: https://platform.openai.com/docs/api-reference

Examples of how to use the Completions API in Python can be found here: https://platform.openai.com/examples
Not all of them will work with all models unfortunately, See the notes on Models for how to get the best results.

Here is a simple python example.

```python
import os
os.environ['OPENAI_API_KEY']="sk-111111111111111111111111111111111111111111111111"
os.environ['OPENAI_API_BASE']="http://0.0.0.0:5001/v1"
import openai

response = openai.ChatCompletion.create(
  model="x",
  messages = [{ 'role': 'system', 'content': "Answer in a consistent style." },
    {'role': 'user', 'content': "Teach me about patience."},
    {'role': 'assistant', 'content': "The river that carves the deepest valley flows from a modest spring; the grandest symphony originates from a single note; the most intricate tapestry begins with a solitary thread."},
    {'role': 'user', 'content': "Teach me about the ocean."},
  ]
)
text = response['choices'][0]['message']['content']
print(text)
```

### Compatibility & not so compatibility

| API endpoint              | tested with                        | notes                                                                       |
| ------------------------- | ---------------------------------- | --------------------------------------------------------------------------- |
| /v1/chat/completions      | openai.ChatCompletion.create()     | Use it with instruction following models                                    |
| /v1/embeddings            | openai.Embedding.create()          | Using SentenceTransformer embeddings                                        |
| /v1/images/generations    | openai.Image.create()              | Bare bones, no model configuration, response_format='b64_json' only.        |
| /v1/moderations           | openai.Moderation.create()         | Basic initial support via embeddings                                        |
| /v1/models                | openai.Model.list()                | Lists models, Currently loaded model first, plus some compatibility options |
| /v1/models/{id}           | openai.Model.get()                 | returns whatever you ask for                                                |
| /v1/edits                 | openai.Edit.create()               | Deprecated by openai, good with instruction following models                |
| /v1/text_completion       | openai.Completion.create()         | Legacy endpoint, variable quality based on the model                        |
| /v1/completions           | openai api completions.create      | Legacy endpoint (v0.25)                                                     |
| /v1/engines/\*/embeddings | python-openai v0.25                | Legacy endpoint                                                             |
| /v1/engines/\*/generate   | openai engines.generate            | Legacy endpoint                                                             |
| /v1/engines               | openai engines.list                | Legacy Lists models                                                         |
| /v1/engines/{model_name}  | openai engines.get -i {model_name} | You can use this legacy endpoint to load models via the api or command line |
| /v1/images/edits          | openai.Image.create_edit()         | not yet supported                                                           |
| /v1/images/variations     | openai.Image.create_variation()    | not yet supported                                                           |
| /v1/audio/\*              | openai.Audio.\*                    | supported                                                                   |
| /v1/files\*               | openai.Files.\*                    | not yet supported                                                           |
| /v1/fine-tunes\*          | openai.FineTune.\*                 | not yet supported                                                           |
| /v1/search                | openai.search, engines.search      | not yet supported                                                           |

Because of the differences in OpenAI model context sizes (2k, 4k, 8k, 16k, etc,) you may need to adjust the max_tokens to fit into the context of the model you choose.

Streaming, temperature, top_p, max_tokens, stop, should all work as expected, but not all parameters are mapped correctly.

Some hacky mappings:

| OpenAI                  | text-generation-webui      | note                                                                                                                                                                                                 |
| ----------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model                   | -                          | Ignored, the model is not changed                                                                                                                                                                    |
| frequency_penalty       | encoder_repetition_penalty | this seems to operate with a different scale and defaults, I tried to scale it based on range & defaults, but the results are terrible. hardcoded to 1.18 until there is a better way                |
| presence_penalty        | repetition_penalty         | same issues as frequency_penalty, hardcoded to 1.0                                                                                                                                                   |
| best_of                 | top_k                      | default is 1 (top_k is 20 for chat, which doesn't support best_of)                                                                                                                                   |
| n                       | 1                          | variations are not supported yet.                                                                                                                                                                    |
| 1                       | num_beams                  | hardcoded to 1                                                                                                                                                                                       |
| 1.0                     | typical_p                  | hardcoded to 1.0                                                                                                                                                                                     |
| logprobs & logit_bias   | -                          | experimental, llama only, transformers-kin only (ExLlama_HF ok), can also use llama tokens if 'model' is not an openai model or will convert from tiktoken for the openai model specified in 'model' |
| messages.name           | -                          | not supported yet                                                                                                                                                                                    |
| suffix                  | -                          | not supported yet                                                                                                                                                                                    |
| user                    | -                          | not supported yet                                                                                                                                                                                    |
| functions/function_call | -                          | function calls are not supported yet                                                                                                                                                                 |

#### Applications

Almost everything needs the `OPENAI_API_KEY` and `OPENAI_API_BASE` environment variable set, but there are some exceptions.

| Compatibility | Application/Library    | Website                                                                        | Notes                                                                                                                                                                                                        |
| ------------- | ---------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ✅❌          | openai-python (v0.25+) | https://github.com/openai/openai-python                                        | only the endpoints from above are working. OPENAI_API_BASE=http://127.0.0.1:5001/v1                                                                                                                          |
| ✅❌          | openai-node            | https://github.com/openai/openai-node                                          | only the endpoints from above are working. environment variables don't work by default, but can be configured (see above)                                                                                    |
| ✅❌          | chatgpt-api            | https://github.com/transitive-bullshit/chatgpt-api                             | only the endpoints from above are working. environment variables don't work by default, but can be configured (see above)                                                                                    |
| ✅            | anse                   | https://github.com/anse-app/anse                                               | API Key & URL configurable in UI, Images also work                                                                                                                                                           |
| ✅            | shell_gpt              | https://github.com/TheR1D/shell_gpt                                            | OPENAI_API_HOST=http://127.0.0.1:5001                                                                                                                                                                        |
| ✅            | gpt-shell              | https://github.com/jla/gpt-shell                                               | OPENAI_API_BASE=http://127.0.0.1:5001/v1                                                                                                                                                                     |
| ✅            | gpt-discord-bot        | https://github.com/openai/gpt-discord-bot                                      | OPENAI_API_BASE=http://127.0.0.1:5001/v1                                                                                                                                                                     |
| ✅            | OpenAI for Notepad++   | https://github.com/Krazal/nppopenai                                            | api_url=http://127.0.0.1:5001 in the config file, or environment variables                                                                                                                                   |
| ✅            | vscode-openai          | https://marketplace.visualstudio.com/items?itemName=AndrewButson.vscode-openai | OPENAI_API_BASE=http://127.0.0.1:5001/v1                                                                                                                                                                     |
| ✅❌          | langchain              | https://github.com/hwchase17/langchain                                         | OPENAI_API_BASE=http://127.0.0.1:5001/v1 even with a good 30B-4bit model the result is poor so far. It assumes zero shot python/json coding. Some model tailored prompt formatting improves results greatly. |
| ✅❌          | Auto-GPT               | https://github.com/Significant-Gravitas/Auto-GPT                               | OPENAI_API_BASE=http://127.0.0.1:5001/v1 Same issues as langchain. Also assumes a 4k+ context                                                                                                                |
| ✅❌          | babyagi                | https://github.com/yoheinakajima/babyagi                                       | OPENAI_API_BASE=http://127.0.0.1:5001/v1                                                                                                                                                                     |
| ❌            | guidance               | https://github.com/microsoft/guidance                                          | logit_bias and logprobs not yet supported                                                                                                                                                                    |
