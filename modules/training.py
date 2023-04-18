import json
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import (LoraConfig, PeftModel, get_peft_model,
                  get_peft_model_state_dict, prepare_model_for_int8_training)

from modules import shared, ui

# This mapping is from a very recent commit, not yet released.
# If not available, default to a backup map for the 3 safe model types.
try:
    from peft.utils.other import \
        TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as \
        model_to_lora_modules
except:
    standard_modules = ["q_proj", "v_proj"]
    model_to_lora_modules = {"llama": standard_modules, "opt": standard_modules, "gptj": standard_modules}

WANT_INTERRUPT = False

PARAMETERS = ["lora_name", "always_override", "save_steps", "micro_batch_size", "batch_size", "epochs", "learning_rate", "lora_rank", "lora_alpha", "lora_dropout", "cutoff_len", "dataset", "eval_dataset", "format", "eval_steps", "raw_text_file", "overlap_len", "newline_favor_len", "do_shuffle", "higher_rank_limit"]

# Mapping of Python class names to peft IDs
MODEL_CLASSES = {
    "LlamaForCausalLM": "llama",
    "OPTForCausalLM": "opt",
    "GPTJForCausalLM": "gptj"
}


def get_datasets(path: str, ext: str):
    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=str.lower)


def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)


def create_train_interface():
    with gr.Tab('Train LoRA', elem_id='lora-train-tab'):
        with gr.Row():
            lora_name = gr.Textbox(label='Name', info='The name of your new LoRA file')
            always_override = gr.Checkbox(label='Override Existing Files', value=True, info='If the name given is the same as an existing file, checking this will replace that file. Leaving unchecked will load that file and continue from it (will use the original rank/alpha/dropout) (NOTE: Currently broken).')
            save_steps = gr.Number(label='Save every n steps', value=0, info='If above 0, a checkpoint of the LoRA will be saved every time this many steps pass.')

        with gr.Row():
            copy_from = gr.Dropdown(label='Copy parameters from', value='None', choices=get_available_loras())
            ui.create_refresh_button(copy_from, lambda: None, lambda: {'choices': get_available_loras()}, 'refresh-button')

        with gr.Row():
            # TODO: Implement multi-device support.
            micro_batch_size = gr.Slider(label='Micro Batch Size', value=4, minimum=1, maximum=128, step=1, info='Per-device batch size (NOTE: multiple devices not yet implemented). Increasing this will increase VRAM usage.')
            batch_size = gr.Slider(label='Batch Size', value=128, minimum=0, maximum=1024, step=4, info='Global batch size. The two batch sizes together determine gradient accumulation (gradientAccum = batch / microBatch). Higher gradient accum values lead to better quality training.')

        with gr.Row():
            epochs = gr.Number(label='Epochs', value=3, info='Number of times every entry in the dataset should be fed into training. So 1 means feed each item in once, 5 means feed it in five times, etc.')
            learning_rate = gr.Textbox(label='Learning Rate', value='3e-4', info='Learning rate, in scientific notation. 3e-4 is a good starting base point. 1e-2 is extremely high, 1e-6 is extremely low.')

        # TODO: What is the actual maximum rank? Likely distinct per model. This might be better to somehow be on a log scale.
        lora_rank = gr.Slider(label='LoRA Rank', value=32, minimum=0, maximum=1024, step=4, info='LoRA Rank, or dimension count. Higher values produce a larger file with better control over the model\'s content. Smaller values produce a smaller file with less overall control. Small values like 4 or 8 are great for stylistic guidance, higher values like 128 or 256 are good for teaching content upgrades, extremely high values (1024+) are difficult to train but may improve fine-detail learning for large datasets. Higher ranks also require higher VRAM.')
        lora_alpha = gr.Slider(label='LoRA Alpha', value=64, minimum=0, maximum=2048, step=4, info='LoRA Alpha. This divided by the rank becomes the scaling of the LoRA. Higher means stronger. A good standard value is twice your Rank.')

        # TODO: Better explain what this does, in terms of real world effect especially.
        lora_dropout = gr.Slider(label='LoRA Dropout', minimum=0.0, maximum=1.0, step=0.025, value=0.05, info='Percentage probability for dropout of LoRA layers. This can help reduce overfitting. Most users should leave at default.')
        cutoff_len = gr.Slider(label='Cutoff Length', minimum=0, maximum=2048, value=256, step=32, info='Cutoff length for text input. Essentially, how long of a line of text to feed in at a time. Higher values require drastically more VRAM.')
        with gr.Row():
            do_shuffle = gr.Checkbox(label='Shuffle Dataset', value=True, info='If checked, the dataset will be randomly shuffled. This can help reduce overfitting.')
            higher_rank_limit = gr.Checkbox(label='Enable higher ranks', value=False, info='If checked, changes Rank/Alpha slider above to go much higher. This will not work without a datacenter-class GPU.')

        with gr.Tab(label="Formatted Dataset"):
            with gr.Row():
                dataset = gr.Dropdown(choices=get_datasets('training/datasets', 'json'), value='None', label='Dataset', info='The dataset file to use for training.')
                ui.create_refresh_button(dataset, lambda: None, lambda: {'choices': get_datasets('training/datasets', 'json')}, 'refresh-button')
                eval_dataset = gr.Dropdown(choices=get_datasets('training/datasets', 'json'), value='None', label='Evaluation Dataset', info='The (optional) dataset file used to evaluate the model after training.')
                ui.create_refresh_button(eval_dataset, lambda: None, lambda: {'choices': get_datasets('training/datasets', 'json')}, 'refresh-button')
                format = gr.Dropdown(choices=get_datasets('training/formats', 'json'), value='None', label='Data Format', info='The format file used to decide how to format the dataset input.')
                ui.create_refresh_button(format, lambda: None, lambda: {'choices': get_datasets('training/formats', 'json')}, 'refresh-button')

            eval_steps = gr.Number(label='Evaluate every n steps', value=100, info='If an evaluation dataset is given, test it every time this many steps pass.')

        with gr.Tab(label="Raw Text File"):
            with gr.Row():
                raw_text_file = gr.Dropdown(choices=get_datasets('training/datasets', 'txt'), value='None', label='Text File', info='The raw text file to use for training.')
                ui.create_refresh_button(raw_text_file, lambda: None, lambda: {'choices': get_datasets('training/datasets', 'txt')}, 'refresh-button')

            with gr.Row():
                overlap_len = gr.Slider(label='Overlap Length', minimum=0, maximum=512, value=128, step=16, info='Overlap length - ie how many tokens from the prior chunk of text to include into the next chunk. (The chunks themselves will be of a size determined by Cutoff Length below). Setting overlap to exactly half the cutoff length may be ideal.')
                newline_favor_len = gr.Slider(label='Prefer Newline Cut Length', minimum=0, maximum=512, value=128, step=16, info='Length (in characters, not tokens) of the maximum distance to shift an overlap cut by to ensure chunks cut at newlines. If too low, cuts may occur in the middle of lines.')

        with gr.Row():
            start_button = gr.Button("Start LoRA Training")
            stop_button = gr.Button("Interrupt")

        output = gr.Markdown(value="Ready")

        def do_copy_params(lora_name: str):
            with open(f"{shared.args.lora_dir}/{clean_path(None, lora_name)}/training_parameters.json", 'r', encoding='utf-8') as formatFile:
                params: dict[str, str] = json.load(formatFile)

            return [params[x] for x in PARAMETERS]

        def change_rank_limit(use_higher_ranks: bool):
            mult = 2 if use_higher_ranks else 1
            return {"maximum": 1024 * mult, "__type__": "update"}, {"maximum": 2048 * mult, "__type__": "update"}

        all_params = [lora_name, always_override, save_steps, micro_batch_size, batch_size, epochs, learning_rate, lora_rank, lora_alpha, lora_dropout, cutoff_len, dataset, eval_dataset, format, eval_steps, raw_text_file, overlap_len, newline_favor_len, do_shuffle, higher_rank_limit]
        copy_from.change(do_copy_params, copy_from, all_params)
        start_button.click(do_train, all_params, output)
        stop_button.click(do_interrupt, None, None, queue=False)
        higher_rank_limit.change(change_rank_limit, [higher_rank_limit], [lora_rank, lora_alpha])


def do_interrupt():
    global WANT_INTERRUPT
    WANT_INTERRUPT = True


def clean_path(base_path: str, path: str):
    """"Strips unusual symbols and forcibly builds a path as relative to the intended directory."""
    # TODO: Probably could do with a security audit to guarantee there's no ways this can be bypassed to target an unwanted path.
    # Or swap it to a strict whitelist of [a-zA-Z_0-9]
    path = path.replace('\\', '/').replace('..', '_')
    if base_path is None:
        return path

    return f'{Path(base_path).absolute()}/{path}'


def do_train(lora_name: str, always_override: bool, save_steps: int, micro_batch_size: int, batch_size: int, epochs: int, learning_rate: str, lora_rank: int, lora_alpha: int, lora_dropout: float,
             cutoff_len: int, dataset: str, eval_dataset: str, format: str, eval_steps: int, raw_text_file: str, overlap_len: int, newline_favor_len: int, do_shuffle: bool, higher_rank_limit: bool):
    global WANT_INTERRUPT
    WANT_INTERRUPT = False

    # == Input validation / processing ==
    yield "Prepping..."
    lora_file_path = f"{shared.args.lora_dir}/{clean_path(None, lora_name)}"
    actual_lr = float(learning_rate)
    model_type = type(shared.model).__name__

    if model_type in MODEL_CLASSES:
        model_id = MODEL_CLASSES[model_type]
    else:
        model_id = "llama"
        if model_type == "PeftModelForCausalLM":
            yield "You are trying to train a LoRA while you already have another LoRA loaded. This will work, but may have unexpected effects. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
            print("Warning: Training LoRA over top of another LoRA. May have unexpected effects.")
        else:
            yield "LoRA training has only currently been validated for LLaMA, OPT, and GPT-J models. Unexpected errors may follow. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
            print(f"Warning: LoRA training has only currently been validated for LLaMA, OPT, and GPT-J models. (Found model type: {model_type})")
        time.sleep(5)

    if shared.args.wbits > 0:
        yield "LoRA training does not yet support 4bit. Please use `--load-in-8bit` for now."
        return

    elif not shared.args.load_in_8bit:
        yield "It is highly recommended you use `--load-in-8bit` for LoRA training. *(Will continue anyway in 2 seconds, press `Interrupt` to stop.)*"
        print("Warning: It is highly recommended you use `--load-in-8bit` for LoRA training.")
        time.sleep(2)  # Give it a moment for the message to show in UI before continuing

    if cutoff_len <= 0 or micro_batch_size <= 0 or batch_size <= 0 or actual_lr <= 0 or lora_rank <= 0 or lora_alpha <= 0:
        yield "Cannot input zeroes."
        return

    gradient_accumulation_steps = batch_size // micro_batch_size
    CURRENT_GRADIENT_ACCUM = gradient_accumulation_steps
    shared.tokenizer.pad_token = 0
    shared.tokenizer.padding_side = "left"

    def tokenize(prompt):
        result = shared.tokenizer(prompt, truncation=True, max_length=cutoff_len + 1, padding="max_length")
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    # == Prep the dataset, format, etc ==
    if raw_text_file not in ['None', '']:
        print("Loading raw text file dataset...")
        with open(clean_path('training/datasets', f'{raw_text_file}.txt'), 'r', encoding='utf-8') as file:
            raw_text = file.read()

        tokens = shared.tokenizer.encode(raw_text)
        del raw_text  # Note: could be a gig for a large dataset, so delete redundant data as we go to be safe on RAM
        tokens = list(split_chunks(tokens, cutoff_len - overlap_len))
        for i in range(1, len(tokens)):
            tokens[i] = tokens[i - 1][-overlap_len:] + tokens[i]

        text_chunks = [shared.tokenizer.decode(x) for x in tokens]
        del tokens
        if newline_favor_len > 0:
            text_chunks = [cut_chunk_for_newline(x, newline_favor_len) for x in text_chunks]

        train_data = Dataset.from_list([tokenize(x) for x in text_chunks])
        del text_chunks
        eval_data = None

    else:
        if dataset in ['None', '']:
            yield "**Missing dataset choice input, cannot continue.**"
            return

        if format in ['None', '']:
            yield "**Missing format choice input, cannot continue.**"
            return

        with open(clean_path('training/formats', f'{format}.json'), 'r', encoding='utf-8') as formatFile:
            format_data: dict[str, str] = json.load(formatFile)

        def generate_prompt(data_point: dict[str, str]):
            for options, data in format_data.items():
                if set(options.split(',')) == set(x[0] for x in data_point.items() if (x[1] is not None and len(x[1].strip()) > 0)):
                    for key, val in data_point.items():
                        if val is not None:
                            data = data.replace(f'%{key}%', val)
                    return data
            raise RuntimeError(f'Data-point "{data_point}" has no keyset match within format "{list(format_data.keys())}"')

        def generate_and_tokenize_prompt(data_point):
            prompt = generate_prompt(data_point)
            return tokenize(prompt)

        print("Loading JSON datasets...")
        data = load_dataset("json", data_files=clean_path('training/datasets', f'{dataset}.json'))
        train_data = data['train'].map(generate_and_tokenize_prompt)

        if eval_dataset == 'None':
            eval_data = None
        else:
            eval_data = load_dataset("json", data_files=clean_path('training/datasets', f'{eval_dataset}.json'))
            eval_data = eval_data['train'].map(generate_and_tokenize_prompt)
            if do_shuffle:
                eval_data = eval_data.shuffle()

    if do_shuffle:
        train_data = train_data.shuffle()

    # == Start prepping the model itself ==
    if not hasattr(shared.model, 'lm_head') or hasattr(shared.model.lm_head, 'weight'):
        print("Getting model ready...")
        prepare_model_for_int8_training(shared.model)

    print("Prepping for training...")
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=model_to_lora_modules[model_id],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    try:
        if not always_override and Path(f"{lora_file_path}/adapter_model.bin").is_file():
            print("Loading existing LoRA file...")
            lora_model = PeftModel.from_pretrained(shared.model, lora_file_path)
        else:
            print("Creating new LoRA model...")
            lora_model = get_peft_model(shared.model, config)
    except:
        yield traceback.format_exc()
        return

    class Tracked():
        def __init__(self):
            self.current_steps = 0
            self.max_steps = 0

    tracked = Tracked()

    class Callbacks(transformers.TrainerCallback):
        def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps = state.global_step * CURRENT_GRADIENT_ACCUM
            tracked.max_steps = state.max_steps * CURRENT_GRADIENT_ACCUM
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

        def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps += 1
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

    trainer = transformers.Trainer(
        model=lora_model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=actual_lr,
            fp16=False if shared.args.cpu else True,
            logging_steps=5,
            evaluation_strategy="steps" if eval_data is not None else "no",
            eval_steps=math.ceil(eval_steps / gradient_accumulation_steps) if eval_data is not None else None,
            save_strategy="steps",
            save_steps=math.ceil(save_steps / gradient_accumulation_steps),
            output_dir=lora_file_path,
            load_best_model_at_end=True if eval_data is not None else False,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None,
            no_cuda=shared.args.cpu
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(shared.tokenizer, mlm=False),
        callbacks=list([Callbacks()])
    )

    lora_model.config.use_cache = False
    old_state_dict = lora_model.state_dict
    lora_model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(lora_model, type(lora_model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        lora_model = torch.compile(lora_model)

    # == Save parameters for reuse ==
    with open(f"{lora_file_path}/training_parameters.json", 'w', encoding='utf-8') as file:
        vars = locals()
        json.dump({x: vars[x] for x in PARAMETERS}, file)

    # == Main run and monitor loop ==
    print("Starting training...")
    yield "Starting..."
    if WANT_INTERRUPT:
        yield "Interrupted before start."
        return

    def threaded_run():
        trainer.train()

    thread = threading.Thread(target=threaded_run)
    thread.start()
    last_step = 0
    start_time = time.perf_counter()

    while thread.is_alive():
        time.sleep(0.5)
        if WANT_INTERRUPT:
            yield "Interrupting, please wait... *(Run will stop after the current training step completes.)*"

        elif tracked.current_steps != last_step:
            last_step = tracked.current_steps
            time_elapsed = time.perf_counter() - start_time
            if time_elapsed <= 0:
                timer_info = ""
                total_time_estimate = 999
            else:
                its = tracked.current_steps / time_elapsed
                if its > 1:
                    timer_info = f"`{its:.2f}` it/s"
                else:
                    timer_info = f"`{1.0/its:.2f}` s/it"

                total_time_estimate = (1.0 / its) * (tracked.max_steps)

            yield f"Running... **{tracked.current_steps}** / **{tracked.max_steps}** ... {timer_info}, {format_time(time_elapsed)} / {format_time(total_time_estimate)} ... {format_time(total_time_estimate - time_elapsed)} remaining"

    print("Training complete, saving...")
    lora_model.save_pretrained(lora_file_path)

    if WANT_INTERRUPT:
        print("Training interrupted.")
        yield f"Interrupted. Incomplete LoRA saved to `{lora_file_path}`"
    else:
        print("Training complete!")
        yield f"Done! LoRA saved to `{lora_file_path}`"


def split_chunks(arr, step):
    for i in range(0, len(arr), step):
        yield arr[i:i + step]


def cut_chunk_for_newline(chunk: str, max_length: int):
    if '\n' not in chunk:
        return chunk

    first_newline = chunk.index('\n')
    if first_newline < max_length:
        chunk = chunk[first_newline + 1:]

    if '\n' not in chunk:
        return chunk

    last_newline = chunk.rindex('\n')
    if len(chunk) - last_newline < max_length:
        chunk = chunk[:last_newline]

    return chunk


def format_time(seconds: float):
    if seconds < 120:
        return f"`{seconds:.0f}` seconds"

    minutes = seconds / 60
    if minutes < 120:
        return f"`{minutes:.0f}` minutes"

    hours = minutes / 60
    return f"`{hours:.0f}` hours"
