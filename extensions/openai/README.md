# An OpenedAI API (openai like)

This extension creates an API that works kind of like openai (ie. api.openai.com).
It's incomplete so far but perhaps is functional enough for you.

## Setup & installation 

Optional:
```
pip3 install -r requirements.txt
```

Chat & completions don't require anything extra installed unless you want to run flask_cloudflared.

### Embeddings (alpha)

Embeddings requires ```sentence-transformers``` installed, but the chat and completions will function without it loaded. The embeddings endpoint is currently using the HuggingFace model: ```sentence-transformers/all-MiniLM-L6-v2``` or ```sentence-transformers/all-mpnet-base-v2``` for embeddings, which must be installed separately. This produces 384 or 768 dimensional embeddings respectively, which is different from OpenAI's ada model which produces 1536 dimensional embeddings. The model is small and fast. This model and embedding size may change in the future. The default is hard coded into script.py as "st_model_default" and is currently all-MiniLM-L6-v2.

sentence-transformers models can be downloaded from the models's tab, or via download-model.py from the command line. Additionally, you will also need to download the 1_Pooling sub folder which isn't (as of 2023-04-29) downloaded automatically. It contains the config.json for the model, in the end you should have config.json file like this (all-MiniLM-L6-v2 example):

```
text-generation-webui$ ./download-model.py sentence-transformers/all-MiniLM-L6-v2
...
text-generation-webui/models/sentence-transformers_all-MiniLM-L6-v2$ cat 1_Pooling/config.json 
{
  "word_embedding_dimension": 384,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}
```

### Client Application Setup

Almost everything you use it with will require you to set a dummy OpenAI API key environment variable:

```
OPENAI_API_KEY=dummy
```

With the [official python openai client](https://github.com/openai/openai-python), you can set the OPENAI_API_BASE environment variable before you import the openai module, like so:

```
OPENAI_API_BASE=http://127.0.0.1:5001/v1
```

If needed, replace 127.0.0.1 with the IP/port of your server.

Alternate method (in python):

```
import os
os.environ('OPENAI_API_KEY=dummy')
os.environ('OPENAI_API_BASE=http://127.0.0.1:5001/v1')
import openai
```

Other programs I've seen will work if you set some other environment variable before you start your program. See the Applications table for more info.


## Compatibility & not so compatibility

What's kinda working:

| API endpoint | tested with | notes |
| --- | --- | --- |
| /v1/models | openai.Model.list() | returns the currently loaded model_name and some mock compatibility options |
| /v1/models/{id} | openai.Model.get() | returns whatever you ask for, model does nothing yet anyways |
| /v1/text_completion | openai.Completion.create() | the most tested |
| /v1/chat/completions | openai.ChatCompletion.create() | may still have some issues with roles & stopping, something still isn't right. |
| /v1/embeddings | openai.Embedding.create() | Using Sentence Transformer, 384 dim instead of 1536. 256 word pieces input, longer is truncated. single string input only so far. |
| /v1/moderations | openai.Moderation.create() | does nothing. successfully |
| /v1/engines/*/completions | openai.Completion.create() v0.25 | Legacy engines endpoints |

The model name setting is ignored in completions, but you may need to adjust the maximum token length to fit the model (ie. set to <2048 tokens instead of 4096, 8k, etc). To mitigate some of this, the max_tokens value is halved until it is less than truncation_length for the model (typically 2k).

Not all parameters are mapped correctly, but temperature, top_p, max_tokens and stream, all work as expected. Sort of? streaming was a ... difficult situation, it seems the protocol has changed? I've stuffed it with two protocols I found...

Still doesn't seem right at times and some clients I tested don't work at all yet, see the Applications table for more details.

Some hacky mappings:

| OpenAI | text-generation-webui | note |
| --- | --- | --- |
| frequency_penalty | encoder_repetition_penalty | this seems to operate with a different scale and defaults, but I map it 1:1 |
| presence_penalty | repetition_penalty | this seems to operate with a different scale and defaults, but I map it 1:1 |
| best_of | top_k | |
| stop | custom_stopping_strings | this is also stuffed with ['\nsystem:', '\nuser:', '\nhuman:', '\nassistant:', '\n###', ] for good measure. |
| n | 1 | it may be worth implementing this but I'm not sure how yet |
| 1.0 | typical_p | |
| 1 | num_beams | |

defaults are mostly from openai, so are different.

### Applications

Everything needs OPENAI_API_KEY=dummy set.

| Compatibility | Application/Library | url | notes / setting |
| --- | --- | --- | --- |
| ✅❌ | openai-python | https://github.com/openai/openai-python | only the endpoints from above are working. OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅ | shell_gpt | https://github.com/TheR1D/shell_gpt | OPENAI_API_HOST=http://127.0.0.1:5001 |
| ✅ | gpt-shell | https://github.com/jla/gpt-shell | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅ | gpt-discord-bot | https://github.com/openai/gpt-discord-bot | OPENAI_API_BASE=http://127.0.0.1:5001/v1 |
| ✅❌ | langchain | https://github.com/hwchase17/langchain | OPENAI_API_BASE=http://127.0.0.1:5001/v1 even with a good 30B-4bit model the result is poor so far. It assumes zero shot python/json coding. Some model tailored prompt formatting improves results greatly. |
| ✅❌ | Auto-GPT | https://github.com/Significant-Gravitas/Auto-GPT | OPENAI_API_BASE=http://127.0.0.1:5001/v1 Same issues as langchain. Also assumes a 4k+ context |
| ❌ | chatbot-ui | https://github.com/mckaywrigley/chatbot-ui | OPENAI_API_HOST=http://127.0.0.1:5001 hits the api, but nothing happens, hangs |
| ❓ | babyagi | https://github.com/yoheinakajima/babyagi | unknown |
| ❓ | openai-node | https://github.com/openai/openai-node | unknown |

## Future plans
* better error handling
* model changing, esp. something for swapping loras or embedding models
* consider switching to FastAPI + starlette for SSE
* do something about rate limiting requests for completions, most systems will only be able handle a single request at a time before OOM
* the whole api, images (stable diffusion), edits, audio, files, fine-tunes, etc.