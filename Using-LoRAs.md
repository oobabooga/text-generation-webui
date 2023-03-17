Based on https://github.com/tloen/alpaca-lora

## Instructions
1. Re-install the requirements

```
pip install -r requirements.txt
```

2. Download the LoRA

```
python download-model.py tloen/alpaca-lora-7b
```

3. Load llama-7b in 8-bit mode (it only seems to work in 8-bit mode, probably a bug in [LoRA.py](https://github.com/oobabooga/text-generation-webui/blob/main/modules/LoRA.py))

```
python server.py --model llama-7b --load-in-8bit
```

Alternatively, load the interface in CPU mode. It also works:

```
python server.py --model llama-7b --cpu
```


4. Select the LoRA in the Parameters tab.

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