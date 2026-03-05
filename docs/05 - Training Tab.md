## Training Your Own LoRAs

The WebUI seeks to make training your own LoRAs as easy as possible. It comes down to just a few simple steps:

### **Step 1**: Make a plan.
- What base model do you want to use? The LoRA you make has to be matched up to a single architecture (eg LLaMA-13B) and cannot be transferred to others (eg LLaMA-7B, StableLM, etc. would all be different). Derivatives of the same model (eg Alpaca finetune of LLaMA-13B) might be transferrable, but even then it's best to train exactly on what you plan to use.
- What are you training it on? Do you want it to learn real information, a simple format, ...?

### **Step 2**: Gather a dataset.
- For instruction/chat training, prepare a JSON dataset in one of the [supported formats](#instruction-templates) (OpenAI messages or ShareGPT).
- For pretraining-style training on raw text, use the `Text Dataset` tab with a JSON file where each row has a `"text"` key.
- If you use a structured dataset not in this format, you may have to find an external way to convert it - or open an issue to request native support.

### **Step 3**: Do the training.
- **3.1**: Load the WebUI, and your model.
    - Make sure you don't have any LoRAs already loaded (unless you want to train for multi-LoRA usage).
- **3.2**: Open the `Training` tab at the top, `Train LoRA` sub-tab.
- **3.3**: Fill in the name of the LoRA, select your dataset in the dataset options.
- **3.4**: Select other parameters to your preference. See [parameters below](#parameters).
- **3.5**: click `Start LoRA Training`, and wait.
    - It can take a few hours for a large dataset, or just a few minute if doing a small run.
    - You may want to monitor your [loss value](#loss) while it goes.

### **Step 4**: Evaluate your results.
- Load the LoRA under the Models Tab.
- You can go test-drive it on the `Text generation` tab, or you can use the `Perplexity evaluation` sub-tab of the `Training` tab.
- If you used the `Save every n steps` option, you can grab prior copies of the model from sub-folders within the LoRA model's folder and try them instead.

### **Step 5**: Re-run if you're unhappy.
- Make sure to unload the LoRA before training it.
- You can simply resume a prior run - use `Copy parameters from` to select your LoRA, and edit parameters. Note that you cannot change the `Rank` of an already created LoRA.
    - If you want to resume from a checkpoint saved along the way, simply copy the contents of the checkpoint folder into the LoRA's folder.
    - (Note: `adapter_model.safetensors` or `adapter_model.bin` is the important file that holds the actual LoRA content).
    - This will start Learning Rate and Steps back to the start. If you want to resume as if you were midway through, you can adjust your Learning Rate to the last reported LR in logs and reduce your epochs.
- Or, you can start over entirely if you prefer.
- If your model is producing corrupted outputs, you probably need to start over and use a lower Learning Rate.
- If your model isn't learning detailed information but you want it to, you might need to just run more epochs, or you might need a higher Rank.
- If your model is enforcing a format you didn't want, you may need to tweak your dataset, or start over and not train as far.

## Instruction Templates

All instruction/chat training uses `apply_chat_template()` with Jinja2 templates. You have two options in the **Data Format** dropdown:

- **Chat Template**: Uses the model's built-in chat template from its tokenizer. Works with instruct/chat models that ship with a chat template (Llama 3, Qwen, Mistral, etc.).
- **Named template** (e.g. ChatML, Alpaca, Llama-v3, etc.): Loads a Jinja2 template from `user_data/instruction-templates/`. This is useful for base models that don't have a built-in template, or when you want to override the model's default template.

Both options are functionally identical — the only difference is where the Jinja2 template string comes from. In both cases:
- The dataset is tokenized via `apply_chat_template()`
- Labels are automatically masked so only assistant responses are trained on
- Multi-turn conversations are supported natively
- Special tokens are handled correctly by the template

The WebUI ships with 50+ templates in `user_data/instruction-templates/`. You can also add your own by creating a `.yaml` file with an `instruction_template` key containing a Jinja2 template string, or a plain `.jinja` file.

**Dataset formats:** Your JSON dataset can use either of these structures:

OpenAI messages format:
```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "A programming language."},
      {"role": "user", "content": "What's it used for?"},
      {"role": "assistant", "content": "Web dev, data science, scripting, and more."}
    ]
  }
]
```

ShareGPT format (`conversations` key with `from`/`value` fields):
```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful assistant."},
      {"from": "human", "value": "What is Python?"},
      {"from": "gpt", "value": "A programming language."},
      {"from": "human", "value": "What's it used for?"},
      {"from": "gpt", "value": "Web dev, data science, scripting, and more."}
    ]
  }
]
```

## Text Dataset

For pretraining-style training on raw text, use the **Text Dataset** tab. Your dataset should be a JSON file with one document per row, each with a `"text"` key:

```json
[
  {"text": "First document content..."},
  {"text": "Second document content..."}
]
```

This is the standard format used by most pretraining datasets (The Pile, RedPajama, etc.).

Each document is tokenized (with BOS token), concatenated into one long token sequence, and split into chunks of `Cutoff Length` tokens. The final chunk is padded if shorter than the cutoff length. When `Add EOS token` is enabled, an EOS token is appended after each document before concatenation, helping the model learn document boundaries.

- `Stride Length` controls the overlap between consecutive chunks in tokens. Set to 0 for non-overlapping chunks (the standard concatenate-and-split approach). Values like 256 or 512 create overlapping chunks that help the model learn context across chunk boundaries, at the cost of more training samples.

## Target Modules

By default, **Target all linear layers** is enabled. This uses peft's `all-linear` mode, which applies LoRA to every `nn.Linear` layer in the model except the output head (`lm_head`). It works for any model architecture.

If you uncheck it, you can manually select individual projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `down_proj`, `up_proj`). Targeting fewer modules reduces VRAM usage and adapter size, but also reduces how much the model can learn. The default selection of `q_proj` + `v_proj` is the minimum for basic style/format training.

## Parameters

The basic purpose and function of each parameter is documented on-page in the WebUI, so read through them in the UI to understand your options.

That said, here's a guide to the most important parameter choices you should consider:

### VRAM

- First, you must consider your VRAM availability.
    - Generally, under default settings, VRAM usage for training with default parameters is very close to when generating text (with 1000+ tokens of context) (ie, if you can generate text, you can train LoRAs).
        - Note: VRAM usage is higher when training 4-bit quantized models. Reduce `Micro Batch Size` to `1` to compensate.
    - If you have VRAM to spare, setting higher batch sizes will use more VRAM and get you better quality training in exchange.
    - If you have large data, setting a higher cutoff length may be beneficial, but will cost significant VRAM. If you can spare some, set your batch size to `1` and see how high you can push your cutoff length.
    - If you're low on VRAM, reducing batch size or cutoff length will of course improve that.
    - Don't be afraid to just try it and see what happens. If it's too much, it will just error out, and you can lower settings and try again.

### Rank

- Second, you want to consider the amount of learning you want.
    - For example, you may wish to just learn a dialogue format (as in the case of Alpaca) in which case setting a low `Rank` value (32 or lower) works great.
    - Or, you might be training on project documentation you want the bot to understand and be able to understand questions about, in which case the higher the rank, the better.
    - Generally, higher Rank = more precise learning = more total content learned = more VRAM usage while training.

### Learning Rate and Epochs

- Third, how carefully you want it to be learned.
    - In other words, how okay or not you are with the model losing unrelated understandings.
    - You can control this with 3 key settings: the Learning Rate, its scheduler, and your total epochs.
    - The learning rate controls how much change is made to the model by each token it sees.
        - It's in scientific notation normally, so for example `3e-4` means `3 * 10^-4` which is `0.0003`. The number after `e-` controls how many `0`s are in the number.
        - Higher values let training run faster, but also are more likely to corrupt prior data in the model.
    - You essentially have two variables to balance: the LR, and Epochs.
        - If you make LR higher, you can set Epochs equally lower to match. High LR + low epochs = very fast, low quality training.
        - If you make LR low, set epochs high. Low LR + high epochs = slow but high-quality training.
    - The scheduler controls change-over-time as you train - it starts high, and then goes low. This helps balance getting data in, and having decent quality, at the same time.
        - You can see graphs of the different scheduler options [in the HuggingFace docs here](https://moon-ci-docs.huggingface.co/docs/transformers/pr_1/en/main_classes/optimizer_schedules#transformers.SchedulerType)

## Loss

When you're running training, the WebUI's console window will log reports that include, among other things, a numeric value named `Loss`. It will start as a high number, and gradually get lower and lower as it goes.

"Loss" in the world of AI training theoretically means "how close is the model to perfect", with `0` meaning "absolutely perfect". This is calculated by measuring the difference between the model outputting exactly the text you're training it to output, and what it actually outputs.

In practice, a good LLM should have a very complex variable range of ideas running in its artificial head, so a loss of `0` would indicate that the model has broken and forgotten how to think about anything other than what you trained it on.

So, in effect, Loss is a balancing game: you want to get it low enough that it understands your data, but high enough that it isn't forgetting everything else. Generally, if it goes below `1.0`, it's going to start forgetting its prior memories, and you should stop training. In some cases you may prefer to take it as low as `0.5` (if you want it to be very very predictable). Different goals have different needs, so don't be afraid to experiment and see what works best for you.

Note: if you see Loss start at or suddenly jump to exactly `0`, it is likely something has gone wrong in your training process (eg model corruption).
