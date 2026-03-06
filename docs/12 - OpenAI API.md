## OpenAI compatible API

The main API for this project is meant to be a drop-in replacement to the OpenAI API, including Chat and Completions endpoints.

* It is 100% offline and private.
* It doesn't create any logs.
* It doesn't connect to OpenAI.
* It doesn't use the openai-python library.

### Starting the API

Add `--api` to your command-line flags.

* To create a public Cloudflare URL, add the `--public-api` flag.
* To listen on your local network, add the `--listen` flag.
* To change the port, which is 5000 by default, use `--api-port 1234` (change 1234 to your desired port number).
* To use SSL, add `--ssl-keyfile key.pem --ssl-certfile cert.pem`. ⚠️ **Note**: this doesn't work with `--public-api` since Cloudflare already uses HTTPS by default.
* To use an API key for authentication, add `--api-key yourkey`.

### Examples

For the documentation with all the endpoints, parameters and their types, consult `http://127.0.0.1:5000/docs` or the [typing.py](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/openai/typing.py) file.

The official examples in the [OpenAI documentation](https://platform.openai.com/docs/api-reference) should also work, and the same parameters apply (although the API here has more optional parameters).

#### Completions

```shell
curl http://127.0.0.1:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "This is a cake recipe:\n\n1.",
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

#### Chat completions

Works best with instruction-following models. If the "instruction_template" variable is not provided, it will be guessed automatically based on the model name using the regex patterns in `user_data/models/config.yaml`.

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

#### Chat completions with characters

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello! Who are you?"
      }
    ],
    "mode": "chat-instruct",
    "character": "Example",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

#### Multimodal/vision (llama.cpp and ExLlamaV3)

##### With /v1/chat/completions (recommended!)

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Please describe what you see in this image."},
          {"type": "image_url", "image_url": {"url": "https://github.com/turboderp-org/exllamav3/blob/master/examples/media/cat.png?raw=true"}}
        ]
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

For base64-encoded images, just replace the inner "url" value with this format: `data:image/FORMAT;base64,BASE64_STRING` where FORMAT is the file type (png, jpeg, gif, etc.) and BASE64_STRING is your base64-encoded image data.

##### With /v1/completions

```shell
curl http://127.0.0.1:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "About image <__media__> and image <__media__>, what I can say is that the first one"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://github.com/turboderp-org/exllamav3/blob/master/examples/media/cat.png?raw=true"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://github.com/turboderp-org/exllamav3/blob/master/examples/media/strawberry.png?raw=true"
            }
          }
        ]
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

For base64-encoded images, just replace the inner "url" values with this format: `data:image/FORMAT;base64,BASE64_STRING` where FORMAT is the file type (png, jpeg, gif, etc.) and BASE64_STRING is your base64-encoded image data.

#### Image generation

```shell
curl http://127.0.0.1:5000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "an orange tree",
    "steps": 9,
    "cfg_scale": 0,
    "batch_size": 1,
    "batch_count": 1
  }'
```

You need to load an image model first. You can do this via the UI, or by adding `--image-model your_model_name` when launching the server.

The output is a JSON object containing a `data` array. Each element has a `b64_json` field with the base64-encoded PNG image:

```json
{
  "created": 1764791227,
  "data": [
    {
      "b64_json": "iVBORw0KGgo..."
    }
  ]
}
```

#### SSE streaming

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "stream": true
  }'
```

#### Logits

```shell
curl -k http://127.0.0.1:5000/v1/internal/logits \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Who is best, Asuka or Rei? Answer:",
    "use_samplers": false
  }'
```

#### Logits after sampling parameters

```shell
curl -k http://127.0.0.1:5000/v1/internal/logits \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Who is best, Asuka or Rei? Answer:",
    "use_samplers": true,
    "top_k": 3
  }'
```

#### List models

```shell
curl -k http://127.0.0.1:5000/v1/internal/model/list \
  -H "Content-Type: application/json"
```

#### Load model

```shell
curl -k http://127.0.0.1:5000/v1/internal/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
    "args": {
      "ctx_size": 32768,
      "flash_attn": true,
      "cache_type": "q8_0"
    }
  }'
```

#### Python chat example

```python
import requests

url = "http://127.0.0.1:5000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

history = []

while True:
    user_message = input("> ")
    history.append({"role": "user", "content": user_message})
    data = {
        "messages": history,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    }

    response = requests.post(url, headers=headers, json=data, verify=False)
    assistant_message = response.json()['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_message})
    print(assistant_message)
```

#### Python chat example with streaming

Start the script with `python -u` to see the output in real time.

```python
import requests
import sseclient  # pip install sseclient-py
import json

url = "http://127.0.0.1:5000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

history = []

while True:
    user_message = input("> ")
    history.append({"role": "user", "content": user_message})
    data = {
        "stream": True,
        "messages": history,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    }

    stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
    client = sseclient.SSEClient(stream_response)

    assistant_message = ''
    for event in client.events():
        payload = json.loads(event.data)
        chunk = payload['choices'][0]['delta']['content']
        assistant_message += chunk
        print(chunk, end='')

    print()
    history.append({"role": "assistant", "content": assistant_message})
```

#### Python completions example with streaming

Start the script with `python -u` to see the output in real time.

```python
import json
import requests
import sseclient  # pip install sseclient-py

url = "http://127.0.0.1:5000/v1/completions"

headers = {
    "Content-Type": "application/json"
}

data = {
    "prompt": "This is a cake recipe:\n\n1.",
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "stream": True,
}

stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
client = sseclient.SSEClient(stream_response)

print(data['prompt'], end='')
for event in client.events():
    payload = json.loads(event.data)
    print(payload['choices'][0]['text'], end='')

print()
```

#### Python parallel requests example

The API supports handling multiple requests in parallel. For ExLlamaV3, this works out of the box. For llama.cpp, you need to pass `--parallel N` to set the number of concurrent slots.

```python
import concurrent.futures
import requests

url = "http://127.0.0.1:5000/v1/chat/completions"
prompts = [
    "Write a haiku about the ocean.",
    "Explain quantum computing in simple terms.",
    "Tell me a joke about programmers.",
]

def send_request(prompt):
    response = requests.post(url, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    })
    return response.json()["choices"][0]["message"]["content"]

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(send_request, prompts))

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

#### Python example with API key

Replace

```python
headers = {
    "Content-Type": "application/json"
}
```

with

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer yourPassword123"
}
```

in any of the examples above.

#### Tool/Function calling

Use a model with tool calling support (Qwen, Mistral, GPT-OSS, etc). Tools are passed via the `tools` parameter and the prompt is automatically formatted using the model's Jinja2 template.

When the model decides to call a tool, the response will have `finish_reason: "tool_calls"` and a `tool_calls` array with structured function names and arguments. You then execute the tool, send the result back as a `role: "tool"` message, and continue until the model responds with `finish_reason: "stop"`.

Some models call multiple tools in parallel (Qwen, Mistral), while others call one at a time (GPT-OSS). The loop below handles both styles.

```python
import json
import requests

url = "http://127.0.0.1:5000/v1/chat/completions"

# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone string"},
                },
                "required": ["timezone"]
            }
        }
    },
]


def execute_tool(name, arguments):
    """Replace this with your actual tool implementations."""
    if name == "get_weather":
        return {"temperature": 22, "condition": "sunny", "humidity": 45}
    elif name == "get_time":
        return {"time": "2:30 PM", "timezone": "JST"}
    return {"error": f"Unknown tool: {name}"}


messages = [{"role": "user", "content": "What time is it in Tokyo and what's the weather like there?"}]

# Tool-calling loop: keep going until the model gives a final answer
for _ in range(10):
    response = requests.post(url, json={"messages": messages, "tools": tools}).json()
    choice = response["choices"][0]

    if choice["finish_reason"] == "tool_calls":
        # Add the assistant's response (with tool_calls) to history
        messages.append({
            "role": "assistant",
            "content": choice["message"]["content"],
            "tool_calls": choice["message"]["tool_calls"],
        })

        # Execute each tool and add results to history
        for tool_call in choice["message"]["tool_calls"]:
            name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            result = execute_tool(name, arguments)

            print(f"Tool call: {name}({arguments}) => {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })
    else:
        # Final answer
        print(f"\nAssistant: {choice['message']['content']}")
        break
```

### Environment variables

The following environment variables can be used (they take precedence over everything else):

| Variable Name          | Description                                                                                        | Example Value              |
|------------------------|------------------------------------|----------------------------|
| `OPENEDAI_PORT`           | Port number         |             5000               |
| `OPENEDAI_CERT_PATH`      | SSL certificate file path         |            cert.pem                |
| `OPENEDAI_KEY_PATH`       | SSL key file path                    |             key.pem               |
| `OPENEDAI_DEBUG`          | Enable debugging (set to 1)    | 1                          |
| `OPENEDAI_EMBEDDING_MODEL` | Embedding model (if applicable) |          sentence-transformers/all-mpnet-base-v2                  |
| `OPENEDAI_EMBEDDING_DEVICE` | Embedding device (if applicable) |           cuda                 |

#### Persistent settings with `settings.yaml`

You can also set the following variables in your `settings.yaml` file:

```
openai-embedding_device: cuda
openai-embedding_model: "sentence-transformers/all-mpnet-base-v2"
openai-debug: 1
```

### Third-party application setup

You can usually force an application that uses the OpenAI API to connect to the local API by using the following environment variables:

```shell
OPENAI_API_HOST=http://127.0.0.1:5000
```

or

```shell
OPENAI_API_KEY=sk-111111111111111111111111111111111111111111111111
OPENAI_API_BASE=http://127.0.0.1:5000/v1
```

With the [official python openai client](https://github.com/openai/openai-python) (v1.x), the address can be set like this:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-111111111111111111111111111111111111111111111111",
    base_url="http://127.0.0.1:5000/v1"
)

response = client.chat.completions.create(
    model="x",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

With the [official Node.js openai client](https://github.com/openai/openai-node) (v4.x):

```js
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://127.0.0.1:5000/v1",
});

const response = await client.chat.completions.create({
  model: "x",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);
```
### Embeddings (alpha)

Embeddings requires `sentence-transformers` installed, but chat and completions will function without it loaded. The embeddings endpoint is currently using the HuggingFace model: `sentence-transformers/all-mpnet-base-v2` for embeddings. This produces 768 dimensional embeddings. The model is small and fast. This model and embedding size may change in the future.

| model name             | dimensions | input max tokens | speed | size | Avg. performance |
| ---------------------- | ---------- | ---------------- | ----- | ---- | ---------------- |
| all-mpnet-base-v2      | 768        | 384              | 2800  | 420M | 63.3             |
| all-MiniLM-L6-v2       | 384        | 256              | 14200 | 80M  | 58.8             |

In short, the all-MiniLM-L6-v2 model is 5x faster, 5x smaller ram, 2x smaller storage, and still offers good quality. Stats from (https://www.sbert.net/docs/pretrained_models.html). To change the model from the default you can set the environment variable `OPENEDAI_EMBEDDING_MODEL`, ex. "OPENEDAI_EMBEDDING_MODEL=all-MiniLM-L6-v2".

Warning: You cannot mix embeddings from different models even if they have the same dimensions. They are not comparable.

### Compatibility

| API endpoint              | notes                                                                       |
| ------------------------- | --------------------------------------------------------------------------- |
| /v1/chat/completions      | Use with instruction-following models. Supports streaming, tool calls.      |
| /v1/completions           | Text completion endpoint.                                                   |
| /v1/embeddings            | Using SentenceTransformer embeddings.                                       |
| /v1/images/generations    | Image generation, response_format='b64_json' only.                         |
| /v1/moderations           | Basic support via embeddings.                                               |
| /v1/models                | Lists models. Currently loaded model first.                                 |
| /v1/models/{id}           | Returns model info.                                                         |
| /v1/audio/\*              | Supported.                                                                  |
| /v1/images/edits          | Not yet supported.                                                          |
| /v1/images/variations     | Not yet supported.                                                          |

#### Applications

Almost everything needs the `OPENAI_API_KEY` and `OPENAI_API_BASE` environment variables set, but there are some exceptions.

| Compatibility | Application/Library  | Website                                                                        | Notes                                                                                     |
| ------------- | -------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| ✅❌          | openai-python        | https://github.com/openai/openai-python                                        | Use `OpenAI(base_url="http://127.0.0.1:5000/v1")`. Only the endpoints from above work.   |
| ✅❌          | openai-node          | https://github.com/openai/openai-node                                          | Use `new OpenAI({baseURL: "http://127.0.0.1:5000/v1"})`. See example above.              |
| ✅            | anse                 | https://github.com/anse-app/anse                                               | API Key & URL configurable in UI, Images also work.                                       |
| ✅            | shell_gpt            | https://github.com/TheR1D/shell_gpt                                            | OPENAI_API_HOST=http://127.0.0.1:5000                                                    |
| ✅            | gpt-shell            | https://github.com/jla/gpt-shell                                               | OPENAI_API_BASE=http://127.0.0.1:5000/v1                                                 |
| ✅            | gpt-discord-bot      | https://github.com/openai/gpt-discord-bot                                      | OPENAI_API_BASE=http://127.0.0.1:5000/v1                                                 |
| ✅            | OpenAI for Notepad++ | https://github.com/Krazal/nppopenai                                            | api_url=http://127.0.0.1:5000 in the config file, or environment variables.               |
| ✅            | vscode-openai        | https://marketplace.visualstudio.com/items?itemName=AndrewButson.vscode-openai | OPENAI_API_BASE=http://127.0.0.1:5000/v1                                                 |
| ✅❌          | langchain            | https://github.com/hwchase17/langchain                                         | Use `base_url="http://127.0.0.1:5000/v1"`. Results depend on model and prompt formatting. |
