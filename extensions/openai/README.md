# An OpenedAI API (openai like)

This extension creates an API that works kind of like openai (ie. api.openai.com).
It's incomplete so far but perhaps is functional enough for you.

## Setup & installation 

Optional (for flask_cloudflared, embeddings):

```
pip3 install -r requirements.txt
```

It listens on tcp port 5001 by default. You can use the OPENEDAI_PORT environment variable to change this.

Make sure you enable it in server launch parameters, it should include:

```
--extensions openai
```

You can also use the ``--listen`` argument to make the server available on the networ, and/or the ```--share``` argument to enable a public Cloudflare endpoint.

To enable the basic image generation support (txt2img) set the environment variable SD_WEBUI_URL to point to your Stable Diffusion API ([Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)).

For example:
```
SD_WEBUI_URL=http://127.0.0.1:7861
```

### Models

This has been successfully tested with Alpaca, Koala, Vicuna, WizardLM and their variants, (ex. gpt4-x-alpaca, GPT4all-snoozy, stable-vicuna, wizard-vicuna, etc.) and many others. Models that have been trained for **Instruction Following** work best. If you test with other models please let me know how it goes. Less than satisfying results (so far) from: RWKV-4-Raven, llama, mpt-7b-instruct/chat.

For best results across all API endpoints, a model like [vicuna-13b-v1.3-GPTQ](https://huggingface.co/TheBloke/vicuna-13b-v1.3-GPTQ), [stable-vicuna-13B-GPTQ](https://huggingface.co/TheBloke/stable-vicuna-13B-GPTQ) or [airoboros-13B-gpt4-1.3-GPTQ](https://huggingface.co/TheBloke/airoboros-13B-gpt4-1.3-GPTQ) is a good start.

For good results with the [Completions](https://platform.openai.com/docs/api-reference/completions) API endpoint, in addition to the above models, you can also try using a base model like [falcon-7b](https://huggingface.co/tiiuae/falcon-7b) or Llama.

For good results with the [ChatCompletions](https://platform.openai.com/docs/api-reference/chat) or [Edits](https://platform.openai.com/docs/api-reference/edits) API endpoints you can use almost any model trained for instruction following - within the limits of the model. Be sure that the proper instruction template is detected and loaded or the results will not be good.

For the proper instruction format to be detected you need to have a matching model entry in your ```models/config.yaml``` file. Be sure to keep this file up to date.
A matching instruction template file in the characters/instruction-following/ folder will loaded and applied to format messages correctly for the model - this is critical for good results.

For example, the Wizard-Vicuna family of models are trained with the Vicuna 1.1 format. In the models/config.yaml file there is this matching entry:

```
.*wizard.*vicuna:
  mode: 'instruct'
  instruction_template: 'Vicuna-v1.1'
```

This refers to ```characters/instruction-following/Vicuna-v1.1.yaml```, which looks like this:

```
user: "USER:"
bot: "ASSISTANT:"
turn_template: "<|user|> <|user-message|>\n<|bot|> <|bot-message|></s>\n"
context: "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
```

For most common models this is already setup, but if you are using a new or uncommon model you may need add a matching entry to the models/config.yaml and possibly create your own instruction-following template and for best results.

If you see this in your logs, it probably means that the correct format could not be loaded:
```
Warning: Loaded default instruction-following template for model.
```

### Embeddings (alpha)

Embeddings requires ```sentence-transformers``` installed, but chat and completions will function without it loaded. The embeddings endpoint is currently using the HuggingFace model: ```sentence-transformers/all-mpnet-base-v2``` for embeddings. This produces 768 dimensional embeddings (the same as the text-davinci-002 embeddings), which is different from OpenAI's current default ```text-embedding-ada-002``` model which produces 1536 dimensional embeddings. The model is small-ish and fast-ish. This model and embedding size may change in the future.

| model name | dimensions | input max tokens | speed | size | Avg. performance | 
| --- | --- | --- | --- | --- | --- |
| text-embedding-ada-002 | 1536 | 8192| - | - | - |
| text-davinci-002 | 768 | 2046 | - | - | - |
| all-mpnet-base-v2 | 768 | 384 | 2800 | 420M | 63.3 |
| all-MiniLM-L6-v2 | 384 | 256 | 14200 | 80M | 58.8 |

In short, the all-MiniLM-L6-v2 model is 5x faster, 5x smaller ram, 2x smaller storage, and still offers good quality. Stats from (https://www.sbert.net/docs/pretrained_models.html). To change the model from the default you can set the environment variable OPENEDAI_EMBEDDING_MODEL, ex. "OPENEDAI_EMBEDDING_MODEL=all-MiniLM-L6-v2".

Warning: You cannot mix embeddings from different models even if they have the same dimensions. They are not comparable.

### Client Application Setup


Almost everything you use it with will require you to set a dummy OpenAI API key environment variable.

With the [official python openai client](https://github.com/openai/openai-python), you can set the OPENAI_API_BASE environment variable before you import the openai module, like so:

```
OPENAI_API_KEY=sk-111111111111111111111111111111111111111111111111
OPENAI_API_BASE=http://127.0.0.1:5001/v1
```

If needed, replace 127.0.0.1 with the IP/port of your server.

If using .env files to save the OPENAI_API_BASE and OPENAI_API_KEY variables, you can ensure compatibility by loading the .env file before loading the openai module, like so in python:

```
from dotenv import load_dotenv
load_dotenv()
import openai
```

With the [official Node.js openai client](https://github.com/openai/openai-node) it is slightly more more complex because the environment variables are not used by default, so small source code changes may be required to use the environment variables, like so:

```
const openai = OpenAI(Configuration({
  apiKey: process.env.OPENAI_API_KEY,
  basePath: process.env.OPENAI_API_BASE,
}));
```

For apps made with the [chatgpt-api Node.js client library](https://github.com/transitive-bullshit/chatgpt-api):

```
const api = new ChatGPTAPI({
  apiKey: process.env.OPENAI_API_KEY,
  apiBaseUrl: process.env.OPENAI_API_BASE,
})
```

## API Documentation & Examples

The OpenAI API is well documented, you can view the documentation here: https://platform.openai.com/docs/api-reference

Examples of how to use the Completions API in Python can be found here: https://platform.openai.com/examples
Not all of them will work with all models unfortunately, See the notes on Models for how to get the best results.

Here is a simple python example of how you can use the Edit endpoint as a translator.

```python
import openai
response = openai.Edit.create(
  model="x",
  instruction="Translate this into French",
  input="Our mission is to ensure that artificial general intelligence benefits all of humanity.",
)
print(response['choices'][0]['text'])
# Sample Output:
# Notre mission est de garantir que l'intelligence artificielle généralisée profite à tous les membres de l'humanité.
```



## Compatibility & not so compatibility

| API endpoint | tested with | notes |
| --- | --- | --- |
| /v1/models | openai.Model.list() | Lists models, Currently loaded model first, plus some compatibility options |
| /v1/models/{id} | openai.Model.get() | returns whatever you ask for, model does nothing yet anyways |
| /v1/text_completion | openai.Completion.create() | the most tested, only supports single string input so far, variable quality based on the model |
| /v1/chat/completions | openai.ChatCompletion.create() | Quality depends a lot on the model |
| /v1/edits | openai.Edit.create() | Works the best of all, perfect for instruction following models |
| /v1/images/generations | openai.Image.create() | Bare bones, no model configuration, response_format='b64_json' only. |
| /v1/embeddings | openai.Embedding.create() | Using Sentence Transformer, dimensions are different and may never be directly comparable to openai embeddings. |
| /v1/moderations | openai.Moderation.create() | does nothing. successfully. |
| /v1/completions | openai api completions.create | Legacy endpoint (v0.25) |
| /v1/engines/*/embeddings | python-openai v0.25 | Legacy endpoint |
| /v1/engines/*/generate | openai engines.generate | Legacy endpoint |
| /v1/engines | openai engines.list | Legacy Lists models |
| /v1/engines/{model_name} | openai engines.get -i {model_name} | You can use this legacy endpoint to load models via the api |
| /v1/images/edits | openai.Image.create_edit() | not yet supported |
| /v1/images/variations | openai.Image.create_variation() | not yet supported |
| /v1/audio/\* | openai.Audio.\* | not yet supported |
| /v1/files\* | openai.Files.\* | not yet supported |
| /v1/fine-tunes\* | openai.FineTune.\* | not yet supported |
| /v1/search | openai.search, engines.search | not yet supported |

The model name setting is ignored in completions, but you may need to adjust the maximum token length to fit the model (ie. set to <2048 tokens instead of 4096, 8k, etc). To mitigate some of this, the max_tokens value is halved until it is less than truncation_length for the model (typically 2k).

Streaming, temperature, top_p, max_tokens, stop, should all work as expected, but not all parameters are mapped correctly.

Some hacky mappings:

| OpenAI | text-generation-webui | note |
| --- | --- | --- |
| frequency_penalty | encoder_repetition_penalty | this seems to operate with a different scale and defaults, I tried to scale it based on range & defaults, but the results are terrible. hardcoded to 1.18 until there is a better way |
| presence_penalty | repetition_penalty | same issues as frequency_penalty, hardcoded to 1.0 |
| best_of | top_k | default is 1 |
| stop | custom_stopping_strings | this is also stuffed with ['\n###', "\n{user prompt}", "{user prompt}" ] for good measure. |
| n | 1 | variations are not supported yet. |
| 1 | num_beams | hardcoded to 1 |
| 1.0 | typical_p | hardcoded to 1.0 |
| max_tokens | max_new_tokens | For Text Completions max_tokens is set smaller than the truncation_length minus the prompt length. This can cause no input to be generated if the prompt is too large. For ChatCompletions, the older chat messages may be dropped to fit the max_new_tokens requested |
| logprobs | - | not supported yet |
| logit_bias | - | not supported yet |
| messages.name | - | not supported yet |
| user | - | not supported yet |
| functions/function_call | - | function calls are not supported yet |

defaults are mostly from openai, so are different. I use the openai defaults where I can and try to scale them to the webui defaults with the same intent.

### Applications

Almost everything needs the OPENAI_API_KEY environment variable set, for example:
```
OPENAI_API_KEY=sk-111111111111111111111111111111111111111111111111
```
Some apps are picky about key format, but 'dummy' or 'sk-dummy' also work in most cases.
Most application will work if you also set:
```
OPENAI_API_BASE=http://127.0.0.1:5001/v1
```
but there are some exceptions.

| Compatibility | Application/Library | url | notes / setting |
| --- | --- | --- | --- |
| ✅❌ | openai-python (v0.25+) | https://github.com/openai/openai-python | only the endpoints from above are working. OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅❌ | openai-node | https://github.com/openai/openai-node | only the endpoints from above are working. environment variables don't work by default, but can be configured (see above) |
| ✅❌ | chatgpt-api | https://github.com/transitive-bullshit/chatgpt-api | only the endpoints from above are working. environment variables don't work by default, but can be configured (see above) |
| ✅ | anse | https://github.com/anse-app/anse | API Key & URL configurable in UI |
| ✅ | shell_gpt | https://github.com/TheR1D/shell_gpt | OPENAI_API_HOST=http://127.0.0.1:5001 |
| ✅ | gpt-shell | https://github.com/jla/gpt-shell | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅ | gpt-discord-bot | https://github.com/openai/gpt-discord-bot | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅ | OpenAI for Notepad++ | https://github.com/Krazal/nppopenai | api_url=http://127.0.0.1:5001 in the config file, or environment variables |
| ✅ | vscode-openai | https://marketplace.visualstudio.com/items?itemName=AndrewButson.vscode-openai | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅❌ | langchain | https://github.com/hwchase17/langchain | OPENAI_API_BASE=http://127.0.0.1:5001/v1 even with a good 30B-4bit model the result is poor so far. It assumes zero shot python/json coding. Some model tailored prompt formatting improves results greatly. |
| ✅❌ | Auto-GPT | https://github.com/Significant-Gravitas/Auto-GPT | OPENAI_API_BASE=http://127.0.0.1:5001/v1 Same issues as langchain. Also assumes a 4k+ context |
| ✅❌ | babyagi | https://github.com/yoheinakajima/babyagi | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |

## Future plans
* better error handling
* model changing, esp. something for swapping loras or embedding models
* consider switching to FastAPI + starlette for SSE (openai SSE seems non-standard)
* do something about rate limiting or locking requests for completions, most systems will only be able handle a single request at a time before OOM

## Bugs? Feedback? Comments? Pull requests?

To enable debugging and get copious output you can set the OPENEDAI_DEBUG=1 environment variable.

Are all appreciated, please @matatonic and I'll try to get back to you as soon as possible.
