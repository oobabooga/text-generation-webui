# An OpenedAI API (openai like)

This extension creates an API that works kind of like openai (ie. api.openai.com).
It's incomplete so far but perhaps is functional enough for you.

## Setup & installation 

Doesn't require anything extra installed unless you want to run flask_cloudflared.

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

Other programs I've seen will work if you set some other environment variable before you start your program.

[shell_gpt](https://github.com/TheR1D/shell_gpt):

```
OPENAI_API_HOST=http://127.0.0.1:5001
```

or some other use:

```
API_URL=http://127.0.0.1:5001/v1/chat/completions
```

etc.

I may compile a list of these settings if it's worth while.

## Compatibility & not so compatibility

What's kinda working:
| API endpoint | tested with | notes |
| /v1/models | openai.Model.list() | returns the currently loaded model_name and 'gpt3.5-turbo' for some mock compatibility |
| /v1/text_completion | openai.Completion.create() | the most tested |
| /v1/chat/completions | openai.ChatCompletion.create() | may still have some issues with roles & stopping, something still isn't right. |

The model setting is ignored in completions, but you may need to adjust the maximum token length to fit the model (ie. set to <2048 tokens instead of 4096, 8k, etc).

Not all parameters are mapped correctly, but temperature, top_p, max_tokens and stream, all work as expected. Sort of? streaming was a ... difficult situation, it seems the protocol has changed? I've stuffed it with two protocols I found...

Still doesn't seem right at times and some clients I tested don't work at all yet (Ex. chatbot-ui).

I'm attempting to get Auto-GPT or babyagi working, but need to setup other stuff too.

Some hacky mappings:

frequency_penalty -> encoder_repetition_penalty # this seems to operate with a different scale and defaults, but I map it 1:1
presence_penalty -> repetition_penalty # this seems to operate with a different scale and defaults, but I map it 1:1
best_of -> top_k
stop -> custom_stopping_strings, this is also stuffed with ['\nsystem', 'system:', '\nuser', 'user:', '\n###', '###'] for good measure.
n -> 1 # it may be worth implementing this but I'm not sure how yet
1.0 -> typical_p
1 -> num_beams

defaults are mostly from openai, so are different.

## Future plans

* work with Auto-GPT/babyagi
* better error handling
* model changing, something for loras
* consider switching to FastAPI
* do something about rate limiting requests for completions, most systems will only be able handle a single request at a time before OOM
* the whole api, model changing, images (stable diffusion), embeddings, audio, files, training, etc.