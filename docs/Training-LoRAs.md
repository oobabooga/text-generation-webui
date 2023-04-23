
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
