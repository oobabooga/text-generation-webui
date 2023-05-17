Based on https://github.com/tloen/alpaca-lora

## Instructions

1. Download a LoRA, for instance:

```
python download-model.py tloen/alpaca-lora-7b
```

2. Load the LoRA. 16-bit, 8-bit, and CPU modes work:

```
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --load-in-8bit
python server.py --model llama-7b-hf --lora tloen_alpaca-lora-7b --cpu
```

* For using LoRAs in 4-bit mode, follow [these special instructions](GPTQ-models-(4-bit-mode).md#using-loras-in-4-bit-mode).

* Instead of using the `--lora` command-line flag, you can also select the LoRA in the "Parameters" tab of the interface.

## Prompt
For the Alpaca LoRA in particular, the prompt must be formatted like this:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Write a Python script that generates text using the transformers library.
### Response:
```

Sample output:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Write a Python script that generates text using the transformers library.
### Response:

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
texts = ["Hello world", "How are you"]
for sentence in texts:
sentence = tokenizer(sentence)
print(f"Generated {len(sentence)} tokens from '{sentence}'")
output = model(sentences=sentence).predict()
print(f"Predicted {len(output)} tokens for '{sentence}':\n{output}")
```

## Training a LoRA

You can train your own LoRAs from the `Training` tab. See [Training LoRAs](Training-LoRAs.md) for details.
