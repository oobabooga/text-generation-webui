# An OpenedAI API (openai like)

This extension creates an API that works kind of like openai (ie. api.openai.com).
It's incomplete so far but perhaps is functional enough for you.

## Setup & installation 

Optional (for flask_cloudflared, embeddings):

```
pip3 install -r requirements.txt
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
OPENAI_API_KEY=dummy
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

## Compatibility & not so compatibility

What's working:

| API endpoint | tested with | notes |
| --- | --- | --- |
| /v1/models | openai.Model.list() | returns the currently loaded model_name and some mock compatibility options |
| /v1/models/{id} | openai.Model.get() | returns whatever you ask for, model does nothing yet anyways |
| /v1/text_completion | openai.Completion.create() | the most tested, only supports single string input so far |
| /v1/chat/completions | openai.ChatCompletion.create() | depending on the model, this may add leading linefeeds |
| /v1/embeddings | openai.Embedding.create() | Using Sentence Transformer, dimensions are different and may never be directly comparable to openai embeddings. |
| /v1/moderations | openai.Moderation.create() | does nothing. successfully. |
| /v1/engines/\*/... completions, embeddings, generate | python-openai v0.25 and earlier | Legacy engines endpoints |

The model name setting is ignored in completions, but you may need to adjust the maximum token length to fit the model (ie. set to <2048 tokens instead of 4096, 8k, etc). To mitigate some of this, the max_tokens value is halved until it is less than truncation_length for the model (typically 2k).

Streaming, temperature, top_p, max_tokens, stop, should all work as expected, but not all parameters are mapped correctly.

Some hacky mappings:

| OpenAI | text-generation-webui | note |
| --- | --- | --- |
| frequency_penalty | encoder_repetition_penalty | this seems to operate with a different scale and defaults, I tried to scale it based on range & defaults, but the results are terrible. hardcoded to 1.18 until there is a better way |
| presence_penalty | repetition_penalty | same issues as frequency_penalty, hardcoded to 1.0 |
| best_of | top_k | |
| stop | custom_stopping_strings | this is also stuffed with ['\nsystem:', '\nuser:', '\nhuman:', '\nassistant:', '\n###', ] for good measure. |
| n | 1 | hardcoded, it may be worth implementing this but I'm not sure how yet |
| 1.0 | typical_p | hardcoded |
| 1 | num_beams | hardcoded |
| max_tokens | max_new_tokens | max_tokens is scaled down by powers of 2 until it's smaller than truncation length. |
| logprobs | - | ignored |

defaults are mostly from openai, so are different. I use the openai defaults where I can and try to scale them to the webui defaults with the same intent.

### Applications

Everything needs OPENAI_API_KEY=dummy set.

| Compatibility | Application/Library | url | notes / setting |
| --- | --- | --- | --- |
| ✅❌ | openai-python | https://github.com/openai/openai-python | only the endpoints from above are working. OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅❌ | openai-node | https://github.com/openai/openai-node | only the endpoints from above are working. environment variables don't work by default, but can be configured (see above) |
| ✅❌ | chatgpt-api | https://github.com/transitive-bullshit/chatgpt-api | only the endpoints from above are working. environment variables don't work by default, but can be configured (see above) |
| ✅ | shell_gpt | https://github.com/TheR1D/shell_gpt | OPENAI_API_HOST=http://127.0.0.1:5001 |
| ✅ | gpt-shell | https://github.com/jla/gpt-shell | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅ | gpt-discord-bot | https://github.com/openai/gpt-discord-bot | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅❌ | langchain | https://github.com/hwchase17/langchain | OPENAI_API_BASE=http://127.0.0.1:5001/v1 even with a good 30B-4bit model the result is poor so far. It assumes zero shot python/json coding. Some model tailored prompt formatting improves results greatly. |
| ✅❌ | Auto-GPT | https://github.com/Significant-Gravitas/Auto-GPT | OPENAI_API_BASE=http://127.0.0.1:5001/v1 Same issues as langchain. Also assumes a 4k+ context |
| ✅❌ | babyagi | https://github.com/yoheinakajima/babyagi | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |

## Future plans
* better error handling
* model changing, esp. something for swapping loras or embedding models
* consider switching to FastAPI + starlette for SSE (openai SSE seems non-standard)
* do something about rate limiting or locking requests for completions, most systems will only be able handle a single request at a time before OOM
* the whole api, images (stable diffusion), audio (whisper), fine-tunes (training), edits, files, etc.