Based on https://github.com/tloen/alpaca-lora

## Instructions
1. Download the LoRA

```
python download-model.py tloen/alpaca-lora-7b
```

2. Load llama-7b in 8-bit mode (it only seems to work in 8-bit mode, not sure if that's a bug in [LoRA.py](https://github.com/oobabooga/text-generation-webui/blob/main/modules/LoRA.py) or if it's just how LoRAs work.)

```
python server.py --model llama-7b --load-in-8bit
```

Alternatively, load the interface in CPU mode. It also works:

```
python server.py --model llama-7b --cpu
```


3. Select the LoRA in the Parameters tab.

## Prompt
For this particular LoRA, the prompt must be formatted like this:

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

For now there is no menu in the interface for training a new LoRA, but it's really easy to do with the `alpaca-lora` code.

All I had to do was

```
conda activate textgen
git clone 'https://github.com/tloen/alpaca-lora'
```

then I edited those two lines in `alpaca-lora/finetune.py` to use my existing `llama-7b` folder instead of downloading everything from decapoda:

```
model = LlamaForCausalLM.from_pretrained(
    "models/llama-7b",
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(
    "models/llama-7b", add_eos_token=True
)
```

and ran the script with 

```
python finetune.py
```

It just worked. It runs at 22.32s/it, with 1170 iterations in total, so about 7 hours and a half for training a LoRA. RTX 3090, 18153MiB VRAM used, drawing maximum power (350W, room heater mode).