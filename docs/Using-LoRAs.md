Based on https://github.com/tloen/alpaca-lora

## Instructions

1. Download a LoRA, for instance:

```
python download-model.py tloen/alpaca-lora-7b
```

2. Load the LoRA. 16-bit, 8-bit, and CPU modes work:

```
python server.py --model llama-7b-hf --lora alpaca-lora-7b
python server.py --model llama-7b-hf --lora alpaca-lora-7b --load-in-8bit
python server.py --model llama-7b-hf --lora alpaca-lora-7b --cpu
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

The Training tab in the interface can be used to train a LoRA. The parameters are self-documenting and good defaults are included.

You can interrupt and resume LoRA training in this tab. If the name and rank are the same, training will resume using the `adapter_model.bin` in your LoRA folder. You can resume from a past checkpoint by replacing this file using the contents of one of the checkpoint folders. Note that the learning rate and steps will be reset, and you may want to set the learning rate to the last reported rate in the console output.

LoRA training was contributed by [mcmonkey4eva](https://github.com/mcmonkey4eva) in PR [#570](https://github.com/oobabooga/text-generation-webui/pull/570).

#### Using the original alpaca-lora code

Kept here for reference. The Training tab has much more features than this method.

```
conda activate textgen
git clone https://github.com/tloen/alpaca-lora
```

Edit those two lines in `alpaca-lora/finetune.py` to use your existing model folder instead of downloading everything from decapoda:

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

Run the script with:

```
python finetune.py
```

It just works. It runs at 22.32s/it, with 1170 iterations in total, so about 7 hours and a half for training a LoRA. RTX 3090, 18153MiB VRAM used, drawing maximum power (350W, room heater mode).
