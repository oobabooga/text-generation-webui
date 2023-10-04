## Training Your Own LoRAs

The WebUI seeks to make training your own LoRAs as easy as possible. It comes down to just a few simple steps:

### **Step 1**: Make a plan.
- What base model do you want to use? The LoRA you make has to be matched up to a single architecture (eg LLaMA-13B) and cannot be transferred to others (eg LLaMA-7B, StableLM, etc. would all be different). Derivatives of the same model (eg Alpaca finetune of LLaMA-13B) might be transferrable, but even then it's best to train exactly on what you plan to use.
- What model format do you want? At time of writing, 8-bit models are most stable, and 4-bit are supported but experimental. In the near future it is likely that 4-bit will be the best option for most users.
- What are you training it on? Do you want it to learn real information, a simple format, ...?

### **Step 2**: Gather a dataset.
- If you use a dataset similar to the [Alpaca](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json) format, that is natively supported by the `Formatted Dataset` input in the WebUI, with premade formatter options.
- If you use a dataset that isn't matched to Alpaca's format, but uses the same basic JSON structure, you can make your own format file by copying `training/formats/alpaca-format.json` to a new file and [editing its content](#format-files).
- If you can get the dataset into a simple text file, that works too! You can train using the `Raw text file` input option.
    - This means you can for example just copy/paste a chatlog/documentation page/whatever you want, shove it in a plain text file, and train on it.
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
    - (Note: `adapter_model.bin` is the important file that holds the actual LoRA content).
    - This will start Learning Rate and Steps back to the start. If you want to resume as if you were midway through, you can adjust your Learning Rate to the last reported LR in logs and reduce your epochs.
- Or, you can start over entirely if you prefer.
- If your model is producing corrupted outputs, you probably need to start over and use a lower Learning Rate.
- If your model isn't learning detailed information but you want it to, you might need to just run more epochs, or you might need a higher Rank.
- If your model is enforcing a format you didn't want, you may need to tweak your dataset, or start over and not train as far.

## Format Files

If using JSON formatted datasets, they are presumed to be in the following approximate format:

```json
[
    {
        "somekey": "somevalue",
        "key2": "value2"
    },
    {
        // etc
    }
]
```

Where the keys (eg `somekey`, `key2` above) are standardized, and relatively consistent across the dataset, and the values (eg `somevalue`, `value2`) contain the content actually intended to be trained.

For Alpaca, the keys are `instruction`, `input`, and `output`, wherein `input` is sometimes blank.

A simple format file for Alpaca to be used as a chat bot is:

```json
{
    "instruction,output": "User: %instruction%\nAssistant: %output%",
    "instruction,input,output": "User: %instruction%: %input%\nAssistant: %output%"
}
```

Note that the keys (eg `instruction,output`) are a comma-separated list of dataset keys, and the values are a simple string that use those keys with `%%`.

So for example if a dataset has `"instruction": "answer my question"`, then the format file's `User: %instruction%\n` will be automatically filled in as `User: answer my question\n`.

If you have different sets of key inputs, you can make your own format file to match it. This format-file is designed to be as simple as possible to enable easy editing to match your needs.

## Raw Text File Settings

When using raw text files as your dataset, the text is automatically split into chunks based on your `Cutoff Length` you get a few basic options to configure them.
- `Overlap Length` is how much to overlap chunks by. Overlapping chunks helps prevent the model from learning strange mid-sentence cuts, and instead learn continual sentences that flow from earlier text.
- `Prefer Newline Cut Length` sets a maximum distance in characters to shift the chunk cut towards newlines. Doing this helps prevent lines from starting or ending mid-sentence, preventing the model from learning to cut off sentences randomly.
- `Hard Cut String` sets a string that indicates there must be a hard cut without overlap. This defaults to `\n\n\n`, meaning 3 newlines. No trained chunk will ever contain this string. This allows you to insert unrelated sections of text in the same text file, but still ensure the model won't be taught to randomly change the subject.

## Parameters

The basic purpose and function of each parameter is documented on-page in the WebUI, so read through them in the UI to understand your options.

That said, here's a guide to the most important parameter choices you should consider:

### VRAM

- First, you must consider your VRAM availability.
    - Generally, under default settings, VRAM usage for training with default parameters is very close to when generating text (with 1000+ tokens of context) (ie, if you can generate text, you can train LoRAs).
        - Note: worse by default in the 4-bit monkeypatch currently. Reduce `Micro Batch Size` to `1` to restore this to expectations.
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

In practice, a good LLM should have a very complex variable range of ideas running in its artificial head, so a loss of `0` would indicate that the model has broken and forgotten to how think about anything other than what you trained it.

So, in effect, Loss is a balancing game: you want to get it low enough that it understands your data, but high enough that it isn't forgetting everything else. Generally, if it goes below `1.0`, it's going to start forgetting its prior memories, and you should stop training. In some cases you may prefer to take it as low as `0.5` (if you want it to be very very predictable). Different goals have different needs, so don't be afraid to experiment and see what works best for you.

Note: if you see Loss start at or suddenly jump to exactly `0`, it is likely something has gone wrong in your training process (eg model corruption).

## Note: 4-Bit Monkeypatch

The [4-bit LoRA monkeypatch](GPTQ-models-(4-bit-mode).md#using-loras-in-4-bit-mode) works for training, but has side effects:
- VRAM usage is higher currently. You can reduce the `Micro Batch Size` to `1` to compensate.
- Models do funky things. LoRAs apply themselves, or refuse to apply, or spontaneously error out, or etc. It can be helpful to reload base model or restart the WebUI between training/usage to minimize chances of anything going haywire.
- Loading or working with multiple LoRAs at the same time doesn't currently work.
- Generally, recognize and treat the monkeypatch as the dirty temporary hack it is - it works, but isn't very stable. It will get better in time when everything is merged upstream for full official support.

## Legacy notes

LoRA training was contributed by [mcmonkey4eva](https://github.com/mcmonkey4eva) in PR [#570](https://github.com/oobabooga/text-generation-webui/pull/570).

### Using the original alpaca-lora code

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
