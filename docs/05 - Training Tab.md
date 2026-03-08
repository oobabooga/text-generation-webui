## Training Your Own LoRAs

A LoRA is tied to a specific model architecture — a LoRA trained on Llama 3 8B won't work on Mistral 7B. Train on the exact model you plan to use.

### Quick Start

1. Load your base model with the **Transformers** loader (no LoRAs loaded).
2. Open the **Training** tab > **Train LoRA**.
3. Pick a dataset and configure parameters (see [below](#parameters)).
4. Click **Start LoRA Training** and monitor the [loss](#loss).
5. When done, load the LoRA from the **Models** tab and test it.

### Resuming Training

To resume from a checkpoint, use the same LoRA name and uncheck `Override Existing Files`. If checkpoints exist (from `Save every n steps`), training will automatically resume from the latest one with full optimizer and scheduler state preserved. Note that you cannot change the `Rank` of an already created LoRA.

You should also use `Copy parameters from` to restore the UI settings (learning rate, epochs, etc.) from the previous run, so that training continues with the same configuration.

### Troubleshooting

- **Corrupted outputs**: Start over with a lower Learning Rate.
- **Not learning enough**: Run more epochs, or increase the Rank.
- **Unwanted formatting**: Tweak your dataset, or train for fewer steps.

## Instruction Templates

All instruction/chat training uses `apply_chat_template()` with Jinja2 templates. You have two options in the **Instruction Template** dropdown:

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

Each parameter has a description in the UI. Below is guidance on the most important choices.

### VRAM

VRAM usage during training is roughly similar to inference with ~1000 tokens of context. If you can run the model, you can probably train LoRAs with the default settings. If you run out of VRAM, reduce `Micro Batch Size` or `Cutoff Length`. Training 4-bit quantized models uses more VRAM — set `Micro Batch Size` to `1` to compensate.

### Rank

Higher rank = more learning capacity = larger adapter = more VRAM. Use 4–8 for style/format, 128–256 to teach factual knowledge.

### Learning Rate and Epochs

These control how aggressively the model learns and how many times it sees the data. Higher LR + fewer epochs = fast but rough. Lower LR + more epochs = slower but higher quality. The scheduler (default: cosine) decays the LR over the course of training — see [HuggingFace docs](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules) for graphs of each option.

## Loss

When you're running training, the WebUI's console window will log reports that include, among other things, a numeric value named `Loss`. It will start as a high number, and gradually get lower and lower as it goes.

Loss measures how far the model's predictions are from the training data, with `0` meaning a perfect match. It's calculated as the cross-entropy between the model's output distribution and the expected tokens.

In practice, a loss of `0` means the model has overfit — it memorized the training data at the expense of its general capabilities.

Loss is a balancing game: you want it low enough that the model learns your data, but not so low that it loses general knowledge. Generally, if it goes below `1.0`, overfitting is likely and you should stop training. In some cases you may want to go as low as `0.5` (if you need very predictable outputs). Different goals have different needs, so experiment and see what works best for you.

Note: if you see Loss start at or suddenly jump to exactly `0`, it is likely something has gone wrong in your training process (eg model corruption).
