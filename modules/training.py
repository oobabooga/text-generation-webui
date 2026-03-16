import os

os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_DISABLED"] = "true"

import json
import math
import random
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml
import gradio as gr

from modules import shared, ui, utils
from modules.evaluate import (
    calculate_perplexity,
    generate_markdown_table,
    save_past_evaluations
)
from modules.logging_colors import logger
from modules.models import reload_model

PARAMETERS = ["lora_name", "always_override", "all_linear", "q_proj_en", "v_proj_en", "k_proj_en", "o_proj_en", "gate_proj_en", "down_proj_en", "up_proj_en", "save_steps", "micro_batch_size", "batch_size", "epochs", "learning_rate", "lr_scheduler_type", "lora_rank", "lora_alpha", "lora_dropout", "cutoff_len", "dataset", "eval_dataset", "format", "eval_steps", "text_dataset", "higher_rank_limit", "warmup_steps", "optimizer", "stride_length", "stop_at_loss", "add_eos_token", "excess_length", "report_to"]
WANT_INTERRUPT = False

train_log = {}
train_template = {}


def create_ui():
    mu = shared.args.multi_user
    with gr.Tab("Training", elem_id="training-tab"):
        with gr.Tab('Train LoRA', elem_id='lora-train-tab'):
            tmp = gr.State('')
            with gr.Row():
                with gr.Column():
                    gr.Markdown("[Tutorial](https://github.com/oobabooga/text-generation-webui/wiki/05-%E2%80%90-Training-Tab)")

                    with gr.Row():
                        copy_from = gr.Dropdown(label='Copy parameters from', value='None', choices=utils.get_available_loras(), elem_classes=['slim-dropdown'], interactive=not mu)
                        ui.create_refresh_button(copy_from, lambda: None, lambda: {'choices': utils.get_available_loras()}, 'refresh-button', interactive=not mu)

                    with gr.Row():
                        with gr.Column(scale=5):
                            lora_name = gr.Textbox(label='Name', info='The name of your new LoRA file')
                        with gr.Column():
                            always_override = gr.Checkbox(label='Override Existing Files', value=False, info='If the name is the same, checking will replace the existing file, and unchecking will load and continue from it (the rank must be the same).', elem_classes=['no-background'])

                    with gr.Accordion(label='Target Modules', open=False, elem_classes='tgw-accordion'):
                        gr.Markdown("Selects which modules to target in training. Targeting more modules is closer to a full fine-tune at the cost of increased VRAM and adapter size.")
                        all_linear = gr.Checkbox(label='Target all linear layers', value=True, info='Targets every nn.Linear layer except lm_head. Works for any model architecture. When checked, the individual module checkboxes below are ignored.', elem_classes=['no-background'])
                        with gr.Row():
                            with gr.Column():
                                q_proj_en = gr.Checkbox(label='Enable q_proj', value=True)
                            with gr.Column():
                                v_proj_en = gr.Checkbox(label='Enable v_proj', value=True)
                            with gr.Column():
                                k_proj_en = gr.Checkbox(label='Enable k_proj', value=False)
                            with gr.Column():
                                o_proj_en = gr.Checkbox(label='Enable o_proj', value=False)
                            with gr.Column():
                                gate_proj_en = gr.Checkbox(label='Enable gate_proj', value=False)
                            with gr.Column():
                                down_proj_en = gr.Checkbox(label='Enable down_proj', value=False)
                            with gr.Column():
                                up_proj_en = gr.Checkbox(label='Enable up_proj', value=False)

                    with gr.Row():
                        with gr.Column():
                            lora_rank = gr.Slider(label='LoRA Rank', value=8, minimum=0, maximum=1024, step=4, info='Also called dimension count. Higher values = larger file, more content control. Smaller values = smaller file, less control. Use 4 or 8 for style, 128 or 256 to teach, 1024+ for fine-detail on big data. More VRAM is needed for higher ranks.')
                            lora_alpha = gr.Slider(label='LoRA Alpha', value=16, minimum=0, maximum=2048, step=4, info='This divided by the rank becomes the scaling of the LoRA. Higher means stronger. A good standard value is twice your Rank.')
                            batch_size = gr.Slider(label='Batch Size', value=32, minimum=0, maximum=1024, step=4, info='Global batch size. The two batch sizes together determine gradient accumulation (gradientAccum = batch / microBatch). Higher gradient accum values lead to better quality training.')
                            micro_batch_size = gr.Slider(label='Micro Batch Size', value=4, minimum=1, maximum=128, step=1, info='Per-device batch size (NOTE: multiple devices not yet implemented). Increasing this will increase VRAM usage.')
                            cutoff_len = gr.Slider(label='Cutoff Length', minimum=0, maximum=4096, value=512, step=32, info='Maximum sequence length in tokens. For instruction datasets, conversations longer than this are dropped. For text datasets, documents are split into chunks of this size. Higher values require more VRAM.')

                        with gr.Column():
                            save_steps = gr.Number(label='Save every n steps', value=0, info='If above 0, a full training checkpoint (adapter weights, optimizer, scheduler) will be saved every time this many steps pass. Training can be resumed from these checkpoints.')

                            epochs = gr.Number(label='Epochs', value=3, info='Number of times every entry in the dataset should be fed into training. So 1 means feed each item in once, 5 means feed it in five times, etc.')
                            learning_rate = gr.Textbox(label='Learning Rate', value='3e-4', info='In scientific notation. 3e-4 is a good starting base point. 1e-2 is extremely high, 1e-6 is extremely low.')
                            with gr.Row():
                                lr_scheduler_type = gr.Dropdown(label='LR Scheduler', value='cosine', choices=['linear', 'constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt'], info='Learning rate scheduler - defines how the learning rate changes over time. "Constant" means never change, "linear" means to go in a straight line from the learning rate down to 0, cosine follows a curve, etc.', elem_classes=['slim-dropdown'])

                    with gr.Accordion(label='Advanced Options', open=False, elem_classes='tgw-accordion'):
                        with gr.Row():
                            with gr.Column():
                                lora_dropout = gr.Slider(label='LoRA Dropout', minimum=0.0, maximum=1.0, step=0.025, value=0.0, info='Percentage probability for dropout of LoRA layers. This can help reduce overfitting. Most users should leave at default.')
                                stop_at_loss = gr.Slider(label='Stop at loss', minimum=0.0, maximum=3.0, step=0.1, value=0.00, info='The process will automatically stop once the desired loss value is reached. (reasonable numbers are 1.5-1.8)')
                                with gr.Row():
                                    optimizer = gr.Dropdown(label='Optimizer', value='adamw_torch', choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad'], info='Optimizer algorithm. adamw_torch is the standard choice. adamw_bnb_8bit uses less VRAM. adafactor is memory-efficient for large models.', elem_classes=['slim-dropdown'])

                            with gr.Column():
                                warmup_steps = gr.Number(label='Warmup Steps', value=100, info='For this many steps at the start, the learning rate is gradually ramped up from 0 to the target value. This prevents unstable updates early in training.')

                                add_eos_token = gr.Checkbox(label='Add EOS token', value=True, info="Adds EOS token for each document in text datasets.")
                                excess_length = gr.Dropdown(label='Excess length', value='drop', choices=['drop', 'truncate'], info='What to do with conversations that exceed the cutoff length. "Drop" removes them entirely (recommended). "Truncate" cuts from the right, which may produce incomplete responses.', elem_classes=['slim-dropdown'])

                                higher_rank_limit = gr.Checkbox(label='Enable higher ranks', value=False, info='If checked, changes Rank/Alpha slider above to go much higher. This will not work without a datacenter-class GPU.')
                                report_to = gr.Radio(label="Save detailed logs with", value="None", choices=["None", "wandb", "tensorboard"], interactive=True)

                with gr.Column():
                    with gr.Tab(label='Chat Dataset'):
                        with gr.Row():
                            dataset = gr.Dropdown(choices=utils.get_chat_datasets(str(shared.user_data_dir / 'training/datasets')), value='None', label='Dataset File', info='A JSON file with chat conversations (messages or ShareGPT format). Each row is one conversation.', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(dataset, lambda: None, lambda: {'choices': utils.get_chat_datasets(str(shared.user_data_dir / 'training/datasets'))}, 'refresh-button', interactive=not mu)

                        with gr.Row():
                            format = gr.Dropdown(choices=get_instruction_templates(), value='None', label='Instruction Template', info='Select an instruction template for formatting the dataset, or "Chat Template" to use the model\'s built-in chat template.', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(format, lambda: None, lambda: {'choices': get_instruction_templates()}, 'refresh-button', interactive=not mu)

                    with gr.Tab(label="Text Dataset"):
                        with gr.Row():
                            text_dataset = gr.Dropdown(choices=utils.get_text_datasets(str(shared.user_data_dir / 'training/datasets')), value='None', label='Dataset File', info='A JSON file with a "text" key per row, for pretraining-style training. Each row is one document.', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(text_dataset, lambda: None, lambda: {'choices': utils.get_text_datasets(str(shared.user_data_dir / 'training/datasets'))}, 'refresh-button', interactive=not mu)

                        stride_length = gr.Slider(label='Stride Length', minimum=0, maximum=2048, value=256, step=32, info='Overlap between chunks in tokens. 0 = no overlap. Values like 256 or 512 help preserve context across chunk boundaries.')

                    with gr.Row():
                        eval_dataset = gr.Dropdown(choices=utils.get_datasets(str(shared.user_data_dir / 'training/datasets'), 'json'), value='None', label='Evaluation Dataset', info='The (optional) dataset file used to evaluate the model after training.', elem_classes=['slim-dropdown'], interactive=not mu)
                        ui.create_refresh_button(eval_dataset, lambda: None, lambda: {'choices': utils.get_datasets(str(shared.user_data_dir / 'training/datasets'), 'json')}, 'refresh-button', interactive=not mu)

                    eval_steps = gr.Number(label='Evaluate every n steps', value=100, info='If an evaluation dataset is given, test it every time this many steps pass.')

                    with gr.Row():
                        start_button = gr.Button("Start LoRA Training", variant='primary', interactive=not mu)
                        stop_button = gr.Button("Interrupt", interactive=not mu)

                    output = gr.Markdown(value="Ready")

        with gr.Tab('Perplexity evaluation', elem_id='evaluate-tab'):
            with gr.Row():
                with gr.Column():
                    models = gr.Dropdown(utils.get_available_models(), label='Models', multiselect=True, interactive=not mu)
                    evaluate_text_file = gr.Dropdown(choices=['wikitext', 'ptb', 'ptb_new'] + utils.get_datasets(str(shared.user_data_dir / 'training/datasets'), 'txt')[1:], value='wikitext', label='Input dataset', info=f'The raw text file on which the model will be evaluated. The first options are automatically downloaded: wikitext, ptb, and ptb_new. The next options are your local text files under {shared.user_data_dir}/training/datasets.', interactive=not mu)
                    with gr.Row():
                        with gr.Column():
                            stride_length = gr.Slider(label='Stride', minimum=0, maximum=32768, value=512, step=256, info='Used to make the evaluation faster at the cost of accuracy. 1 = slowest but most accurate. 512 is a common value.')

                        with gr.Column():
                            max_length = gr.Number(label='max_length', precision=0, step=256, value=0, info='The context for each evaluation. If set to 0, the maximum context length for the model will be used.')

                    with gr.Row():
                        start_current_evaluation = gr.Button("Evaluate loaded model", interactive=not mu)
                        start_evaluation = gr.Button("Evaluate selected models", interactive=not mu)
                        stop_evaluation = gr.Button("Interrupt", interactive=not mu)

                with gr.Column():
                    evaluation_log = gr.Markdown(value='')

            evaluation_table = gr.Dataframe(value=generate_markdown_table(), interactive=True)
            with gr.Row():
                save_comments = gr.Button('Save comments', elem_classes="small-button", interactive=not mu)
                refresh_table = gr.Button('Refresh the table', elem_classes="small-button", interactive=not mu)

    # Training events
    all_params = [lora_name, always_override, all_linear, q_proj_en, v_proj_en, k_proj_en, o_proj_en, gate_proj_en, down_proj_en, up_proj_en, save_steps, micro_batch_size, batch_size, epochs, learning_rate, lr_scheduler_type, lora_rank, lora_alpha, lora_dropout, cutoff_len, dataset, eval_dataset, format, eval_steps, text_dataset, higher_rank_limit, warmup_steps, optimizer, stride_length, stop_at_loss, add_eos_token, excess_length, report_to]

    copy_from.change(do_copy_params, [copy_from] + all_params, all_params)
    start_button.click(do_train, all_params, output)
    stop_button.click(do_interrupt, None, None, queue=False)
    higher_rank_limit.change(change_rank_limit, [higher_rank_limit], [lora_rank, lora_alpha])

    # Evaluation events. For some reason, the interrupt event
    # doesn't work with the .then() syntax, so I write them one
    # by one in this ugly but functional way.
    ev = start_evaluation.click(calculate_perplexity, [models, evaluate_text_file, stride_length, max_length], evaluation_log, show_progress=False)
    ev.then(generate_markdown_table, None, evaluation_table, show_progress=False)

    ev_cur = start_current_evaluation.click(
        lambda: ['current model'], None, tmp).then(
        calculate_perplexity, [tmp, evaluate_text_file, stride_length, max_length], evaluation_log, show_progress=False)

    ev_cur.then(generate_markdown_table, None, evaluation_table, show_progress=False)

    stop_evaluation.click(None, None, None, cancels=[ev, ev_cur], queue=False)
    refresh_table.click(generate_markdown_table, None, evaluation_table, show_progress=True)
    save_comments.click(
        save_past_evaluations, evaluation_table, None).then(
        lambda: "Comments saved.", None, evaluation_log, show_progress=False)


def do_interrupt():
    global WANT_INTERRUPT
    WANT_INTERRUPT = True


def do_copy_params(lora_name: str, *args):
    f_name = f"{shared.args.lora_dir}/{clean_path(None, lora_name)}/training_parameters.json"
    if Path(f_name).is_file():
        with open(f_name, 'r', encoding='utf-8') as format_file:
            params: dict[str, str] = json.load(format_file)
    else:
        params = {}

    result = list()
    for i in range(0, len(PARAMETERS)):
        key = PARAMETERS[i]
        if key in params:
            result.append(params[key])
        else:
            result.append(args[i])

    return result


def change_rank_limit(use_higher_ranks: bool):
    mult = 2 if use_higher_ranks else 1
    return {"maximum": 1024 * mult, "__type__": "update"}, {"maximum": 2048 * mult, "__type__": "update"}


def clean_path(base_path: str, path: str):
    """Strips unusual symbols and forcibly builds a path as relative to the intended directory."""
    path = path.replace('\\', '/').replace('..', '_')
    if base_path is None:
        return path

    return f'{Path(base_path).absolute()}/{path}'


def get_instruction_templates():
    path = shared.user_data_dir / 'instruction-templates'
    names = set()
    for ext in ['yaml', 'yml', 'jinja', 'jinja2']:
        for f in path.glob(f'*.{ext}'):
            names.add(f.stem)
    return ['None', 'Chat Template'] + sorted(names, key=utils.natural_keys)


def load_template(name):
    """Load a Jinja2 template string from {user_data_dir}/instruction-templates/."""
    path = shared.user_data_dir / 'instruction-templates'
    for ext in ['jinja', 'jinja2', 'yaml', 'yml']:
        filepath = path / f'{name}.{ext}'
        if filepath.exists():
            if ext in ['jinja', 'jinja2']:
                return filepath.read_text(encoding='utf-8')
            else:
                data = yaml.safe_load(filepath.read_text(encoding='utf-8'))
                return data.get('instruction_template', '')
    return ''


def backup_adapter(input_folder):
    # Get the creation date of the adapter file (safetensors or bin)
    try:
        adapter_file = Path(f"{input_folder}/adapter_model.safetensors")
        if not adapter_file.is_file():
            adapter_file = Path(f"{input_folder}/adapter_model.bin")
        if adapter_file.is_file():

            logger.info("Backing up existing LoRA adapter")
            creation_date = datetime.fromtimestamp(adapter_file.stat().st_ctime)
            creation_date_str = creation_date.strftime("Backup-%Y-%m-%d")

            # Create the new subfolder
            subfolder_path = Path(f"{input_folder}/{creation_date_str}")
            subfolder_path.mkdir(parents=True, exist_ok=True)

            # Check if the file already exists in the subfolder
            backup_adapter_file = subfolder_path / adapter_file.name
            if backup_adapter_file.is_file():
                print(" - Backup already exists. Skipping backup process.")
                return

            # Copy existing files to the new subfolder
            existing_files = Path(input_folder).iterdir()
            for file in existing_files:
                if file.is_file():
                    shutil.copy2(file, subfolder_path)
    except Exception as e:
        print("An error occurred in backup_adapter:", str(e))


def calc_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def do_train(lora_name: str, always_override: bool, all_linear: bool, q_proj_en: bool, v_proj_en: bool, k_proj_en: bool, o_proj_en: bool, gate_proj_en: bool, down_proj_en: bool, up_proj_en: bool, save_steps: int, micro_batch_size: int, batch_size: int, epochs: int, learning_rate: str, lr_scheduler_type: str, lora_rank: int, lora_alpha: int, lora_dropout: float, cutoff_len: int, dataset: str, eval_dataset: str, format: str, eval_steps: int, text_dataset: str, higher_rank_limit: bool, warmup_steps: int, optimizer: str, stride_length: int, stop_at_loss: float, add_eos_token: bool, excess_length: str, report_to: str):

    import torch
    import transformers
    from datasets import Dataset, load_dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict
    )

    global WANT_INTERRUPT
    WANT_INTERRUPT = False

    # == Input validation / processing ==
    yield "Preparing the input..."

    if shared.args.loader == 'llama.cpp':
        yield "Error: LoRA training requires a model loaded with the Transformers loader. GGUF models are not supported for training."
        return

    lora_file_path = clean_path(None, lora_name)
    if lora_file_path.strip() == '':
        yield "Missing or invalid LoRA file name input."
        return

    lora_file_path = f"{Path(shared.args.lora_dir)}/{lora_file_path}"
    actual_lr = float(learning_rate)
    model_type = type(shared.model).__name__

    if model_type == "PeftModelForCausalLM":
        if len(shared.lora_names) > 0:
            yield "You are trying to train a LoRA while you already have another LoRA loaded. This will work, but may have unexpected effects. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
            logger.warning("Training LoRA over top of another LoRA. May have unexpected effects.")
        else:
            yield "Model ID not matched due to LoRA loading. Consider reloading base model. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
            logger.warning("Model ID not matched due to LoRA loading. Consider reloading base model.")

        time.sleep(5)

    if cutoff_len <= 0 or micro_batch_size <= 0 or batch_size <= 0 or actual_lr <= 0 or lora_rank <= 0 or lora_alpha <= 0:
        yield "Cannot input zeroes."
        return

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
    original_chat_template = getattr(shared.tokenizer, 'chat_template', None)
    if shared.tokenizer.pad_token_id is None:
        shared.tokenizer.pad_token_id = shared.tokenizer.eos_token_id
    shared.tokenizer.padding_side = "right"

    def list_target_modules():
        if all_linear:
            return "all-linear"

        target_mods = [f"{name}_proj" for name, enabled in {
            "q": q_proj_en, "k": k_proj_en, "v": v_proj_en, "o": o_proj_en,
            "gate": gate_proj_en, "down": down_proj_en, "up": up_proj_en,
        }.items() if enabled]
        return target_mods

    def normalize_messages(data_point):
        """Convert a dataset row to OpenAI messages format for apply_chat_template()."""
        if "messages" in data_point:
            return data_point["messages"]

        if "conversations" in data_point:
            role_map = {"human": "user", "gpt": "assistant"}
            return [
                {"role": role_map.get(turn.get("from", ""), turn.get("from", "")), "content": turn["value"]}
                for turn in data_point["conversations"]
            ]

        raise RuntimeError(
            f'Dataset row must contain "messages" or "conversations" key. '
            f'Found: {list(data_point.keys())}'
        )

    def tokenize_conversation(data_point):
        """Tokenize using apply_chat_template() with assistant-only label masking."""
        messages = normalize_messages(data_point)
        full_ids = list(shared.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False))

        # Build labels: -100 for everything, then unmask assistant turns.
        # This assumes apply_chat_template(messages[:i]) is a token-for-token
        # prefix of apply_chat_template(messages[:i+1]), which holds for all
        # standard chat templates (Llama, ChatML, Mistral, etc.).
        labels = [-100] * len(full_ids)
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # Tokens up to where this assistant turn starts
                header_ids = shared.tokenizer.apply_chat_template(
                    messages[:i], tokenize=True, return_dict=False, add_generation_prompt=True
                )
                # Tokens through end of this assistant turn
                through_ids = shared.tokenizer.apply_chat_template(
                    messages[:i + 1], tokenize=True, return_dict=False
                )
                # Unmask assistant tokens
                start = len(header_ids)
                end = min(len(through_ids), len(full_ids))
                labels[start:end] = full_ids[start:end]

        if len(full_ids) > cutoff_len:
            if excess_length == 'truncate':
                full_ids = full_ids[:cutoff_len]
                labels = labels[:cutoff_len]
            else:
                return {"input_ids": [], "labels": [], "attention_mask": []}

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    train_template.clear()

    # == Prep the dataset, format, etc ==
    has_text_dataset = text_dataset not in ['None', '']
    has_chat_dataset = dataset not in ['None', '']
    if has_text_dataset and has_chat_dataset:
        yield "Error: select either a Chat Dataset or a Text Dataset, not both."
        return

    def tokenize_text_data(data):
        """Tokenize text dataset rows, concatenate, and split into chunks."""
        all_tokens = []
        for row in data:
            tokens = shared.tokenizer.encode(row['text'])
            if add_eos_token:
                tokens.append(shared.tokenizer.eos_token_id)
            all_tokens.extend(tokens)

        stride = int(stride_length)
        step = cutoff_len - stride if stride > 0 else cutoff_len

        if step <= 0:
            return None, "Error: stride length must be smaller than cutoff length."
        if len(all_tokens) < cutoff_len:
            return None, "Error: dataset is too short to fill even one chunk of the given cutoff length."

        chunks = []
        for start in range(0, len(all_tokens), step):
            chunk = all_tokens[start:start + cutoff_len]
            if len(chunk) == 0:
                break
            if len(chunk) < cutoff_len:
                pad_len = cutoff_len - len(chunk)
                chunks.append({
                    "input_ids": chunk + [shared.tokenizer.pad_token_id] * pad_len,
                    "labels": list(chunk) + [-100] * pad_len,
                    "attention_mask": [1] * len(chunk) + [0] * pad_len,
                })
            else:
                chunks.append({
                    "input_ids": chunk,
                    "labels": list(chunk),
                    "attention_mask": [1] * cutoff_len,
                })

        return Dataset.from_list(chunks), None

    if has_text_dataset:
        train_template["template_type"] = "text_dataset"
        logger.info("Loading text dataset")
        data = load_dataset("json", data_files=clean_path(str(shared.user_data_dir / 'training/datasets'), f'{text_dataset}.json'))

        if "text" not in data['train'].column_names:
            yield "Error: text dataset must have a \"text\" key per row."
            return

        train_data, err = tokenize_text_data(data['train'])
        if err:
            yield err
            return

        if eval_dataset == 'None':
            eval_data = None
        else:
            eval_raw = load_dataset("json", data_files=clean_path(str(shared.user_data_dir / 'training/datasets'), f'{eval_dataset}.json'))
            if "text" not in eval_raw['train'].column_names:
                yield "Error: evaluation dataset must have a \"text\" key per row."
                return
            eval_data, err = tokenize_text_data(eval_raw['train'])
            if err:
                yield err
                return
    elif has_chat_dataset:
        if format in ['None', '']:
            yield "Missing format choice input, cannot continue."
            return

        if format == 'Chat Template':
            if not getattr(shared.tokenizer, 'chat_template', None):
                yield "Error: this model's tokenizer does not have a chat template. Select an instruction template instead, or load an instruct/chat model."
                return
        else:
            # Load custom instruction template and set on tokenizer
            template_str = load_template(format)
            if not template_str:
                yield f"Error: could not load instruction template '{format}'."
                return
            shared.tokenizer.chat_template = template_str

        # Unified path — both cases use tokenize_conversation()
        train_template["template_type"] = "chat_template"

        logger.info("Loading JSON dataset with chat template format")
        data = load_dataset("json", data_files=clean_path(str(shared.user_data_dir / 'training/datasets'), f'{dataset}.json'))

        # Validate the first row
        try:
            normalize_messages(data['train'][0])
        except (RuntimeError, KeyError, IndexError) as e:
            yield f"Error: {e}"
            return

        total = len(data['train'])
        train_data = data['train'].map(
            tokenize_conversation,
            remove_columns=data['train'].column_names,
            new_fingerprint='%030x' % random.randrange(16**30)
        )
        train_data = train_data.filter(lambda x: len(x['input_ids']) > 0)
        dropped = total - len(train_data)
        if dropped > 0:
            logger.warning(f"Dropped {dropped}/{total} conversations exceeding cutoff length of {cutoff_len} tokens.")
        if len(train_data) == 0:
            yield f"Error: all {total} conversations exceed the cutoff length of {cutoff_len} tokens. Increase the cutoff length or shorten your data."
            return

        if eval_dataset == 'None':
            eval_data = None
        else:
            eval_data = load_dataset("json", data_files=clean_path(str(shared.user_data_dir / 'training/datasets'), f'{eval_dataset}.json'))
            eval_data = eval_data['train'].map(
                tokenize_conversation,
                remove_columns=eval_data['train'].column_names,
                new_fingerprint='%030x' % random.randrange(16**30)
            )
            eval_data = eval_data.filter(lambda x: len(x['input_ids']) > 0)
    else:
        yield "No dataset selected. Choose a Chat Dataset or a Text Dataset."
        return

    # == We MUST reload model if it went through any previous training, even failed one ==
    if shared.model_dirty_from_training:
        selected_model = shared.model_name
        if selected_model:
            print("\033[1;31;1m(Model has been modified by previous training, it needs to be reloaded...)\033[0;37;0m")
            try:
                yield f"Reloading {selected_model}..."
                reload_model()
                if shared.model is not None:
                    print("Model reloaded OK, continue with training.")
                else:
                    yield f"Failed to load {selected_model}."
                    return
            except Exception:
                exc = traceback.format_exc()
                logger.error('Failed to reload the model.')
                print(exc)
                yield exc.replace('\n', '\n\n')
                return

    # == Start prepping the model itself ==
    if not hasattr(shared.model, 'lm_head') or hasattr(shared.model.lm_head, 'weight'):
        logger.info("Getting model ready")
        if 'quantization_config' in shared.model.config.to_dict():
            prepare_model_for_kbit_training(shared.model)

    # base model is now frozen and should not be reused for any other LoRA training than this one
    shared.model_dirty_from_training = True

    logger.info("Preparing for training")
    target_modules = list_target_modules()
    if not target_modules:
        yield "No target modules selected. Enable at least one module or check 'Target all linear layers'."
        return

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # == Backup the existing adapter ==
    if not always_override:
        backup_adapter(lora_file_path)

    # == get model trainable params
    model_trainable_params, model_all_params = calc_trainable_parameters(shared.model)

    # == Determine if we can resume from a checkpoint ==
    resume_checkpoint = None
    try:
        logger.info("Creating LoRA model")
        lora_model = get_peft_model(shared.model, config)
        if not always_override and Path(lora_file_path).exists():
            # Look for HF Trainer checkpoint dirs (full resumption)
            checkpoints = sorted(Path(lora_file_path).glob("checkpoint-*"), key=os.path.getmtime)
            if checkpoints:
                resume_checkpoint = str(checkpoints[-1])
                logger.info(f"Will resume from checkpoint: {resume_checkpoint}")
            else:
                # Legacy fallback: load bare adapter weights only
                safetensors_path = Path(f"{lora_file_path}/adapter_model.safetensors")
                bin_path = Path(f"{lora_file_path}/adapter_model.bin")
                if safetensors_path.is_file():
                    logger.info("Loading existing LoRA data (safetensors)")
                    from safetensors.torch import load_file
                    state_dict_peft = load_file(str(safetensors_path))
                    set_peft_model_state_dict(lora_model, state_dict_peft)
                elif bin_path.is_file():
                    logger.info("Loading existing LoRA data (bin)")
                    state_dict_peft = torch.load(str(bin_path), weights_only=True)
                    set_peft_model_state_dict(lora_model, state_dict_peft)
    except Exception:
        yield traceback.format_exc().replace('\n', '\n\n')
        return

    class Tracked():
        def __init__(self):
            self.current_steps = 0
            self.max_steps = 0
            self.did_save = False

    tracked = Tracked()
    actual_save_steps = math.ceil(save_steps / gradient_accumulation_steps)

    class Callbacks(transformers.TrainerCallback):
        def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps = state.global_step * gradient_accumulation_steps
            tracked.max_steps = state.max_steps * gradient_accumulation_steps
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

        def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps += 1
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

        def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, logs, **kwargs):
            train_log.update(logs)
            train_log.update({"current_steps": tracked.current_steps})
            if WANT_INTERRUPT:
                print("\033[1;31;1mInterrupted by user\033[0;37;0m")

            print(f"\033[1;30;40mStep: {tracked.current_steps} \033[0;37;0m", end='')
            if 'loss' in logs:
                loss = float(logs['loss'])
                if stop_at_loss > 0 and loss <= stop_at_loss:
                    control.should_epoch_stop = True
                    control.should_training_stop = True
                    print(f"\033[1;31;1mStop Loss {stop_at_loss} reached.\033[0;37;0m")

        def on_save(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint_dir.exists():
                with open(checkpoint_dir / "training_log.json", 'w', encoding='utf-8') as file:
                    json.dump(train_log, file, indent=2)
                with open(checkpoint_dir / "training_prompt.json", 'w', encoding='utf-8') as file:
                    json.dump(train_template, file, indent=2)

    # Fix training for mixed precision models
    for param in shared.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    lora_model.config.use_cache = False

    def collate_fn(batch):
        max_len = max(len(item['input_ids']) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            pad_len = max_len - len(item['input_ids'])
            input_ids.append(item['input_ids'] + [shared.tokenizer.pad_token_id] * pad_len)
            labels.append(item['labels'] + [-100] * pad_len)
            attention_mask.append(item['attention_mask'] + [0] * pad_len)
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(attention_mask),
        }

    trainer = transformers.Trainer(
        model=lora_model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            report_to=report_to if report_to != "None" else "none",
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=math.ceil(warmup_steps / gradient_accumulation_steps),
            num_train_epochs=epochs,
            learning_rate=actual_lr,
            fp16=False if shared.args.cpu or shared.args.bf16 else True,
            bf16=shared.args.bf16,
            optim=optimizer,
            logging_steps=1,
            eval_strategy="steps" if eval_data is not None else "no",
            eval_steps=math.ceil(eval_steps / gradient_accumulation_steps) if eval_data is not None else None,
            save_strategy="steps" if save_steps > 0 or eval_data is not None else "no",
            save_steps=actual_save_steps if save_steps > 0 else None,
            output_dir=lora_file_path,
            lr_scheduler_type=lr_scheduler_type,
            load_best_model_at_end=eval_data is not None,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None,
            use_cpu=shared.args.cpu,
            remove_unused_columns=False,
        ),
        data_collator=collate_fn,
        callbacks=[Callbacks()]
    )

    # == Save parameters for reuse ==
    with open(f"{lora_file_path}/training_parameters.json", 'w', encoding='utf-8') as file:
        local_vars = locals()
        json.dump({x: local_vars[x] for x in PARAMETERS}, file, indent=2)

    # == Save training prompt ==
    with open(f"{lora_file_path}/training_prompt.json", 'w', encoding='utf-8') as file:
        json.dump(train_template, file, indent=2)

    # == Main run and monitor loop ==
    logger.info("Starting training")
    yield "Starting..."

    lora_trainable_param, lora_all_param = calc_trainable_parameters(lora_model)

    if target_modules == "all-linear":
        projections_string = "all-linear"
    else:
        projections_string = ", ".join([projection.replace("_proj", "") for projection in target_modules])

    print(f"Training '{model_type}' model using ({projections_string}) projections")

    if lora_all_param > 0:
        print(f"Trainable params: {lora_trainable_param:,d} ({100 * lora_trainable_param / lora_all_param:.4f} %), All params: {lora_all_param:,d} (Model: {model_all_params:,d})")

    train_log.update({"base_model_name": shared.model_name})
    train_log.update({"base_model_class": shared.model.__class__.__name__})
    train_log.update({"base_loaded_in_4bit": getattr(lora_model, "is_loaded_in_4bit", False)})
    train_log.update({"base_loaded_in_8bit": getattr(lora_model, "is_loaded_in_8bit", False)})
    train_log.update({"projections": projections_string})

    if stop_at_loss > 0:
        print(f"Monitoring loss \033[1;31;1m(Auto-Stop at: {stop_at_loss})\033[0;37;0m")

    if WANT_INTERRUPT:
        yield "Interrupted before start."
        return

    def log_train_dataset(trainer):
        decoded_entries = []
        # Try to decode the entries and write the log file
        try:
            # Iterate over the first 10 elements in the dataset (or fewer if there are less than 10)
            for i in range(min(10, len(trainer.train_dataset))):
                decoded_text = shared.tokenizer.decode(trainer.train_dataset[i]['input_ids'])
                decoded_entries.append({"value": decoded_text})

            # Write the log file
            (shared.user_data_dir / 'logs').mkdir(exist_ok=True)
            with open(shared.user_data_dir / 'logs' / 'train_dataset_sample.json', 'w') as json_file:
                json.dump(decoded_entries, json_file, indent=4)

            logger.info(f"Log file 'train_dataset_sample.json' created in the '{shared.user_data_dir}/logs' directory.")
        except Exception as e:
            logger.error(f"Failed to create log file due to error: {e}")

    thread_error = None

    def threaded_run():
        nonlocal thread_error
        try:
            log_train_dataset(trainer)
            trainer.train(resume_from_checkpoint=resume_checkpoint)
            # Note: save in the thread in case the gradio thread breaks (eg browser closed)
            lora_model.save_pretrained(lora_file_path)
            tracked.did_save = True
            logger.info("LoRA training run is completed and saved.")
            # Save log
            with open(f"{lora_file_path}/training_log.json", 'w', encoding='utf-8') as file:
                json.dump(train_log, file, indent=2)
        except Exception as e:
            thread_error = e
            logger.error(f"Training error: {e}")

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

    # Check for errors from the training thread
    if thread_error is not None:
        yield f"Training failed: {thread_error}"
        return

    # Saving in the train thread might fail if an error occurs, so save here if so.
    if not tracked.did_save:
        logger.info("Training complete, saving")
        lora_model.save_pretrained(lora_file_path)

    # Restore the original chat_template if we changed it for training
    if shared.tokenizer is not None and hasattr(shared.tokenizer, 'chat_template'):
        shared.tokenizer.chat_template = original_chat_template

    if WANT_INTERRUPT:
        logger.info("Training interrupted.")
        yield f"Interrupted. Incomplete LoRA saved to `{lora_file_path}`."
    else:
        logger.info("Training complete!")
        yield f"Done! LoRA saved to `{lora_file_path}`.\n\nBefore testing your new LoRA, make sure to first reload the model, as it is currently dirty from training."


def format_time(seconds: float):
    if seconds < 120:
        return f"`{seconds:.0f}` seconds"

    minutes = seconds / 60
    if minutes < 120:
        return f"`{minutes:.0f}` minutes"

    hours = minutes / 60
    return f"`{hours:.0f}` hours"
