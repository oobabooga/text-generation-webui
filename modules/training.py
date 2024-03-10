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

import gradio as gr
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from peft.utils.other import \
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as model_to_lora_modules
from transformers import is_torch_xpu_available
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)

from modules import shared, ui, utils
from modules.evaluate import (
    calculate_perplexity,
    generate_markdown_table,
    save_past_evaluations
)
from modules.logging_colors import logger
from modules.models import reload_model
from modules.utils import natural_keys

MODEL_CLASSES = {v[1]: v[0] for v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()}
PARAMETERS = ["lora_name", "always_override", "q_proj_en", "v_proj_en", "k_proj_en", "o_proj_en", "gate_proj_en", "down_proj_en", "up_proj_en", "save_steps", "micro_batch_size", "batch_size", "epochs", "learning_rate", "lr_scheduler_type", "lora_rank", "lora_alpha", "lora_dropout", "cutoff_len", "dataset", "eval_dataset", "format", "eval_steps", "raw_text_file", "overlap_len", "newline_favor_len", "higher_rank_limit", "warmup_steps", "optimizer", "hard_cut_string", "train_only_after", "stop_at_loss", "add_eos_token", "min_chars", "report_to"]
WANT_INTERRUPT = False

train_log = {}
train_template = {}


def create_ui():
    mu = shared.args.multi_user
    with gr.Tab("训练", elem_id="training-tab"):
        with gr.Tab('训练LoRA', elem_id='lora-train-tab'):
            tmp = gr.State('')
            with gr.Row():
                with gr.Column():
                    gr.Markdown("[教程](https://github.com/Touch-Night/text-generation-webui/wiki/05-%E2%80%90-Training-Tab)")

                    with gr.Row():
                        copy_from = gr.Dropdown(label='从以下参数复制', value='None', choices=utils.get_available_loras(), elem_classes=['slim-dropdown'], interactive=not mu)
                        ui.create_refresh_button(copy_from, lambda: None, lambda: {'choices': utils.get_available_loras()}, '刷新按钮', interactive=not mu)

                    with gr.Row():
                        with gr.Column(scale=5):
                            lora_name = gr.Textbox(label='名称', info='新LoRA文件的名称')
                        with gr.Column():
                            always_override = gr.Checkbox(label='覆盖现有文件', value=False, info='如果名称相同，选中将替换现有文件，未选中将加载并继续（排名必须相同）。', elem_classes=['no-background'])

                    with gr.Accordion(label='目标模块', open=False):
                        gr.Markdown("选择在训练中要针对的模块。针对更多模块更接近完整的微调，但会增加VRAM需求和适配器大小。\n注意：仅对model_id='llama'有效，其他类型将保留默认训练行为，不使用这些设置。")
                        with gr.Row():
                            with gr.Column():
                                q_proj_en = gr.Checkbox(label='启用q_proj', value=True)
                            with gr.Column():
                                v_proj_en = gr.Checkbox(label='启用v_proj', value=True)
                            with gr.Column():
                                k_proj_en = gr.Checkbox(label='启用k_proj', value=False)
                            with gr.Column():
                                o_proj_en = gr.Checkbox(label='启用o_proj', value=False)
                            with gr.Column():
                                gate_proj_en = gr.Checkbox(label='启用gate_proj', value=False)
                            with gr.Column():
                                down_proj_en = gr.Checkbox(label='启用down_proj', value=False)
                            with gr.Column():
                                up_proj_en = gr.Checkbox(label='启用up_proj', value=False)

                    with gr.Row():
                        with gr.Column():
                            lora_rank = gr.Slider(label='LoRA秩', value=32, minimum=0, maximum=1024, step=4, info='也称为维度计数。较高的值=更大的文件，更多的内容控制。较小的值=更小的文件，控制力较差。用4或8来表示风格，用128或256来教学，用1024+来细节处理大数据。更高的排名需要更多的VRAM。')
                            lora_alpha = gr.Slider(label='LoRA Alpha', value=64, minimum=0, maximum=2048, step=4, info='这个除以排名成为LoRA的缩放。较高意味着更强。一个好的标准值是你排名的两倍。')
                            batch_size = gr.Slider(label='批量大小', value=128, minimum=0, maximum=1024, step=4, info='全局批量大小。这两个批量大小共同决定了梯度累积（gradientAccum = batch / microBatch）。较高的梯度累积值会带来更好的训练质量。')
                            micro_batch_size = gr.Slider(label='微批量大小', value=4, minimum=1, maximum=128, step=1, info='每个设备的批量大小（注意：多设备尚未实现）。增加这个将增加VRAM使用。')
                            cutoff_len = gr.Slider(label='截断长度', minimum=0, maximum=4096, value=256, step=32, info='文本输入的截断长度。本质上，一次输入多长的文本行。较高的值需要大量的VRAM。')

                        with gr.Column():
                            save_steps = gr.Number(label='每n步保存一次', value=0, info='如果大于0，每当这么多步过去时，就会保存LoRA的一个检查点。')

                            epochs = gr.Number(label='周期', value=3, info='数据集中的每个条目应该输入训练的次数。所以1意味着每个项目输入一次，5意味着输入五次，等等。')
                            learning_rate = gr.Textbox(label='学习率', value='3e-4', info='用科学记数法表示。3e-4是一个很好的起点。1e-2非常高，1e-6非常低。')
                            with gr.Row():
                                lr_scheduler_type = gr.Dropdown(label='学习率调度器', value='linear', choices=['linear', 'constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt'], info='学习率调度器 - 定义学习率随时间的变化方式。"Constant"意味着永不改变，"linear"意味着从学习率直线下降到0，cosine遵循曲线等等。', elem_classes=['slim-dropdown'])

                    with gr.Accordion(label='高级选项', open=False):
                        with gr.Row():
                            with gr.Column():
                                lora_dropout = gr.Slider(label='LoRA Dropout', minimum=0.0, maximum=1.0, step=0.025, value=0.05, info='LoRA层的dropout概率百分比。这可以帮助减少过拟合。大多数用户应保持默认值。')
                                stop_at_loss = gr.Slider(label='停止损失', minimum=0.0, maximum=3.0, step=0.1, value=0.00, info='一旦达到期望的损失值，过程将自动停止。（合理的数字是1.5-1.8）')
                                with gr.Row():
                                    optimizer = gr.Dropdown(label='优化器', value='adamw_torch', choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad'], info='不同优化器实现选项，供高级用户使用。不同选项的效果尚未得到很好的记录。', elem_classes=['slim-dropdown'])

                            with gr.Column():
                                warmup_steps = gr.Number(label='热身步数', value=100, info='在开始时的这么多步骤中，学习率将低于正常水平。这有助于训练器准备模型并预先计算统计数据，以提高开始后的训练质量。')
                                train_only_after = gr.Textbox(label='仅在此之后训练', value='', info='在任何给定的文本块中，只考虑*在此字符串之后*的文本进行训练。对于Alpaca数据集，使用"### Response:"仅训练响应并忽略输入。')

                                add_eos_token = gr.Checkbox(label='添加EOS令牌', value=False, info="为每个数据集项目添加EOS令牌。如果是原始文本，则EOS将添加在硬切割处")

                                higher_rank_limit = gr.Checkbox(label='启用更高秩', value=False, info='如果选中，将更改上面的秩/Alpha滑块，使其更高。如果没有数据中心级GPU，这将不起作用。')
                                report_to = gr.Radio(label="保存详细日志至", value="None", choices=["None", "wandb", "tensorboard"], interactive=True)

                with gr.Column():
                    with gr.Tab(label='格式化数据集'):
                        with gr.Row():
                            format = gr.Dropdown(choices=utils.get_datasets('training/formats', 'json'), value='None', label='数据格式', info='用于决定如何格式化数据集输入的格式文件。', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(format, lambda: None, lambda: {'choices': utils.get_datasets('training/formats', 'json')}, '刷新按钮', interactive=not mu)

                        with gr.Row():
                            dataset = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'json'), value='None', label='数据集', info='用于训练的数据集文件。', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(dataset, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'json')}, '刷新按钮', interactive=not mu)

                        with gr.Row():
                            eval_dataset = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'json'), value='None', label='评估数据集', info='用于在训练后评估模型的（可选）数据集文件。', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(eval_dataset, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'json')}, '刷新按钮', interactive=not mu)

                        eval_steps = gr.Number(label='每n步评估一次', value=100, info='如果给出评估数据集，每次通过这么多步骤时测试它。')

                    with gr.Tab(label="原始文本文件"):
                        with gr.Row():
                            raw_text_file = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'txt'), value='None', label='文本文件', info='用于训练的原始文本文件。', elem_classes=['slim-dropdown'], interactive=not mu)
                            ui.create_refresh_button(raw_text_file, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'txt')}, '刷新按钮', interactive=not mu)

                        with gr.Row():
                            with gr.Column():
                                overlap_len = gr.Slider(label='重叠长度', minimum=0, maximum=512, value=128, step=16, info='在下一个文本块中包含多少个来自前一个文本块的tokens。（文本块本身的大小由截断长度决定）。将重叠设置为截断长度的恰好一半可能是理想的。')
                                newline_favor_len = gr.Slider(label='优先换行剪切长度', minimum=0, maximum=512, value=128, step=16, info='为了确保文本块在换行处剪切，可移动重叠剪切的最大距离的长度（以字符而非tokens计算）。如果设置得太低，剪切可能会发生在行中间。')

                            with gr.Column():
                                hard_cut_string = gr.Textbox(label='硬剪切字符串', value='\\n\\n\\n', info='表示文本部分之间硬剪切的字符串。有助于防止不想要的重叠。')
                                min_chars = gr.Number(label='忽略小块', value=0, info='忽略小于或等于该数字字符的硬剪切块。')

                        with gr.Row():
                            start_button = gr.Button("开始LoRA训练", variant='primary', interactive=not mu)
                            stop_button = gr.Button("中断", interactive=not mu)

                        output = gr.Markdown(value="准备就绪")

        with gr.Tab('困惑度评估', elem_id='evaluate-tab'):
            with gr.Row():
                with gr.Column():
                    models = gr.Dropdown(utils.get_available_models(), label='模型', multiselect=True, interactive=not mu)
                    evaluate_text_file = gr.Dropdown(choices=['wikitext', 'ptb', 'ptb_new'] + utils.get_datasets('training/datasets', 'txt')[1:], value='wikitext', label='输入数据集', info='模型将在其上进行评估的原始文本文件。前几个选项会自动下载：wikitext, ptb, 和 ptb_new。接下来的选项是您在training/datasets下的本地文本文件。', interactive=not mu)
                    with gr.Row():
                        with gr.Column():
                            stride_length = gr.Slider(label='步长', minimum=0, maximum=32768, value=512, step=256, info='以牺牲准确性为代价来加快评估速度。1 = 最慢但最准确。512是一个常见的值。')

                        with gr.Column():
                            max_length = gr.Slider(label='最大长度', minimum=0, maximum=shared.settings['truncation_length_max'], value=0, step=256, info='每次评估的上下文长度。如果设置为0，将使用模型的最大上下文长度。')

                    with gr.Row():
                        start_current_evaluation = gr.Button("评估已加载模型", interactive=not mu)
                        start_evaluation = gr.Button("评估所选模型", interactive=not mu)
                        stop_evaluation = gr.Button("中断", interactive=not mu)

                with gr.Column():
                    evaluation_log = gr.Markdown(value='')

            evaluation_table = gr.Dataframe(value=generate_markdown_table(), interactive=True)
            with gr.Row():
                save_comments = gr.Button('保存评论', elem_classes="small-button", interactive=not mu)
                refresh_table = gr.Button('刷新表格', elem_classes="small-button", interactive=not mu)

    # Training events
    all_params = [lora_name, always_override, q_proj_en, v_proj_en, k_proj_en, o_proj_en, gate_proj_en, down_proj_en, up_proj_en, save_steps, micro_batch_size, batch_size, epochs, learning_rate, lr_scheduler_type, lora_rank, lora_alpha, lora_dropout, cutoff_len, dataset, eval_dataset, format, eval_steps, raw_text_file, overlap_len, newline_favor_len, higher_rank_limit, warmup_steps, optimizer, hard_cut_string, train_only_after, stop_at_loss, add_eos_token, min_chars, report_to]

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
        lambda: "评论已保存。", None, evaluation_log, show_progress=False)


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


def backup_adapter(input_folder):
    # Get the creation date of the file adapter_model.bin
    try:
        adapter_file = Path(f"{input_folder}/adapter_model.bin")
        if adapter_file.is_file():

            logger.info("正在备份现有的LoRA适配器")
            creation_date = datetime.fromtimestamp(adapter_file.stat().st_ctime)
            creation_date_str = creation_date.strftime("Backup-%Y-%m-%d")

            # Create the new subfolder
            subfolder_path = Path(f"{input_folder}/{creation_date_str}")
            subfolder_path.mkdir(parents=True, exist_ok=True)

            # Check if the file already exists in the subfolder
            backup_adapter_file = Path(f"{input_folder}/{creation_date_str}/adapter_model.bin")
            if backup_adapter_file.is_file():
                print(" - 备份已存在。跳过备份过程。")
                return

            # Copy existing files to the new subfolder
            existing_files = Path(input_folder).iterdir()
            for file in existing_files:
                if file.is_file():
                    shutil.copy2(file, subfolder_path)
    except Exception as e:
        print("备份适配器时发生错误：", str(e))


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


def do_train(lora_name: str, always_override: bool, q_proj_en: bool, v_proj_en: bool, k_proj_en: bool, o_proj_en: bool, gate_proj_en: bool, down_proj_en: bool, up_proj_en: bool, save_steps: int, micro_batch_size: int, batch_size: int, epochs: int, learning_rate: str, lr_scheduler_type: str, lora_rank: int, lora_alpha: int, lora_dropout: float, cutoff_len: int, dataset: str, eval_dataset: str, format: str, eval_steps: int, raw_text_file: str, overlap_len: int, newline_favor_len: int, higher_rank_limit: bool, warmup_steps: int, optimizer: str, hard_cut_string: str, train_only_after: str, stop_at_loss: float, add_eos_token: bool, min_chars: int, report_to: str):

    if shared.args.monkey_patch:
        from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import (
            replace_peft_model_with_int4_lora_model
        )
        replace_peft_model_with_int4_lora_model()

    global WANT_INTERRUPT
    WANT_INTERRUPT = False

    # == Input validation / processing ==
    yield "准备输入..."
    lora_file_path = clean_path(None, lora_name)
    if lora_file_path.strip() == '':
        yield "缺少或无效的LoRA文件名输入。"
        return

    lora_file_path = f"{Path(shared.args.lora_dir)}/{lora_file_path}"
    actual_lr = float(learning_rate)
    model_type = type(shared.model).__name__

    if model_type in MODEL_CLASSES:
        model_id = MODEL_CLASSES[model_type]
    else:
        model_id = "llama"
        if model_type == "PeftModelForCausalLM":
            if len(shared.lora_names) > 0:
                yield "您正在尝试在已加载另一个LoRA的情况下训练LoRA。这可以工作，但可能会有意想不到的效果。*(将在5秒后继续，按`中断`停止。)*"
                logger.warning("在另一个LoRA上训练LoRA。可能会有意想不到的效果。")
            else:
                yield "由于LoRA加载，模型ID未匹配。考虑重新加载基础模型。*(将在5秒后继续，按`中断`停止。)*"
                logger.warning("由于LoRA加载，模型ID未匹配。考虑重新加载基础模型。")
        else:
            yield "LoRA训练目前仅对LLaMA、OPT、GPT-J和GPT-NeoX模型进行了验证。可能会出现意外错误。*(将在5秒后继续，按`中断`停止。)*"
            logger.warning(f"LoRA训练目前仅对LLaMA、OPT、GPT-J和GPT-NeoX模型进行了验证。（发现模型类型：{model_type}）")

        time.sleep(5)

    if shared.args.loader == 'GPTQ-for-LLaMa' and not shared.args.monkey_patch:
        yield "使用GPTQ-for-LLaMa进行LoRA训练需要启用`--monkey-patch`"
        return

    if cutoff_len <= 0 or micro_batch_size <= 0 or batch_size <= 0 or actual_lr <= 0 or lora_rank <= 0 or lora_alpha <= 0:
        yield "不能输入零。"
        return

    gradient_accumulation_steps = batch_size // micro_batch_size
    shared.tokenizer.pad_token_id = 0
    shared.tokenizer.padding_side = "left"

    # Populate target_modules list with chosen X_proj modules. Llama-based models only atm, non-llama will revert to default behavior.
    def list_target_modules(model_id):
        if model_id != "llama" and model_id != "mistral":
            return model_to_lora_modules[model_id]

        available_modules = {
            "gate": gate_proj_en,
            "down": down_proj_en,
            "up": up_proj_en,
            "q": q_proj_en,
            "v": v_proj_en,
            "k": k_proj_en,
            "o": o_proj_en,
        }
        target_mods = [f"{name}_proj" for name, enabled in available_modules.items() if enabled]
        return target_mods

    def encode(text, add_bos_token):
        result = shared.tokenizer.encode(text, truncation=True, max_length=cutoff_len)
        # Check if the first two tokens are BOS
        if len(result) >= 2 and result[:2] == [shared.tokenizer.bos_token_id, shared.tokenizer.bos_token_id]:
            result = result[1:]

        if not add_bos_token and result[0] == shared.tokenizer.bos_token_id:
            result = result[1:]
        return result

    def tokenize(prompt, append_eos_token=False):

        if train_only_after == '' or train_only_after not in prompt:
            input_ids = encode(prompt, True)

            if append_eos_token and input_ids[-1] != shared.tokenizer.eos_token_id and len(input_ids) < cutoff_len:
                input_ids.append(shared.tokenizer.eos_token_id)

            input_ids = [shared.tokenizer.pad_token_id] * (cutoff_len - len(input_ids)) + input_ids
            labels = [1] * len(input_ids)

        else:
            ind = prompt.index(train_only_after) + len(train_only_after)
            before_tokens = encode(prompt[:ind], True)
            after_tokens = encode(prompt[ind:], False)

            if append_eos_token and after_tokens[-1] != shared.tokenizer.eos_token_id:
                after_tokens.append(shared.tokenizer.eos_token_id)

            full_length = len(after_tokens) + len(before_tokens)
            if full_length > cutoff_len:
                after_tokens = after_tokens[:cutoff_len - len(before_tokens)]
            else:
                before_tokens = [shared.tokenizer.pad_token_id] * (cutoff_len - full_length) + before_tokens

            input_ids = before_tokens + after_tokens
            labels = [-100] * len(before_tokens) + [1] * len(after_tokens)

        input_ids = torch.tensor(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(shared.tokenizer.pad_token_id),
        }

    train_template.clear()

    # == Prep the dataset, format, etc ==
    if raw_text_file not in ['None', '']:
        train_template["template_type"] = "raw_text"
        logger.info("正在加载原始文本文件数据集")
        fullpath = clean_path('training/datasets', f'{raw_text_file}')
        fullpath = Path(fullpath)
        if fullpath.is_dir():
            logger.info('训练路径目录 {}'.format(raw_text_file))
            raw_text = ""
            file_paths = sorted(fullpath.glob('*.txt'), key=lambda path: natural_keys(path.name))
            for file_path in file_paths:
                if file_path.is_file():
                    with file_path.open('r', encoding='utf-8') as file:
                        raw_text += file.read().replace('\r', '')

                    logger.info(f"已加载训练文件：{file_path.name}")
        else:
            with open(clean_path('training/datasets', f'{raw_text_file}.txt'), 'r', encoding='utf-8') as file:
                raw_text = file.read().replace('\r', '')

        cut_string = hard_cut_string.replace('\\n', '\n')
        eos_added = 0
        out_tokens = []
        for text_part in raw_text.split(cut_string):
            if len(text_part.strip()) <= min_chars:
                continue

            tokens = shared.tokenizer.encode(text_part)
            if add_eos_token:
                tokens.append(shared.tokenizer.eos_token_id)
                eos_added += 1

            step = cutoff_len - overlap_len
            if step <= 0:
                yield f"错误：overlap_len（{overlap_len}）不能大于或等于cutoff_len（{cutoff_len}）"
                return

            out_tokens.extend(split_chunks(tokens, cutoff_len, step))

        if eos_added > 0:
            print(f"已向{eos_added}个文本块添加EOS")

        del raw_text  # Note: could be a gig for a large dataset, so delete redundant data as we go to be safe on RAM
        text_chunks = [shared.tokenizer.decode(x) for x in out_tokens]
        del out_tokens
        if newline_favor_len > 0:
            text_chunks = [cut_chunk_for_newline(x, newline_favor_len) for x in text_chunks]

        train_data = Dataset.from_list([tokenize(x) for x in text_chunks])
        del text_chunks
        eval_data = None
    else:
        if dataset in ['None', '']:
            yield "缺少数据集选择输入，无法继续。"
            return

        if format in ['None', '']:
            yield "缺少格式选择输入，无法继续。"
            return

        train_template["template_type"] = "dataset"

        with open(clean_path('training/formats', f'{format}.json'), 'r', encoding='utf-8-sig') as formatFile:
            format_data: dict[str, str] = json.load(formatFile)

        # == store training prompt ==
        for _, value in format_data.items():
            prompt_key = f"template_{len(train_template)}"
            train_template[prompt_key] = value

        def generate_prompt(data_point: dict[str, str]):
            for options, data in format_data.items():
                if set(options.split(',')) == set(x[0] for x in data_point.items() if (type(x[1]) is str and len(x[1].strip()) > 0)):
                    for key, val in data_point.items():
                        if type(val) is str:
                            data = data.replace(f'%{key}%', val)
                    return data
            raise RuntimeError(f'数据点 "{data_point}" 在格式 "{list(format_data.keys())}" 中没有匹配的键集')

        def generate_and_tokenize_prompt(data_point):
            prompt = generate_prompt(data_point)
            return tokenize(prompt, add_eos_token)

        logger.info("正在加载JSON数据集")
        data = load_dataset("json", data_files=clean_path('training/datasets', f'{dataset}.json'))
        train_data = data['train'].map(generate_and_tokenize_prompt, new_fingerprint='%030x' % random.randrange(16**30))

        if eval_dataset == 'None':
            eval_data = None
        else:
            eval_data = load_dataset("json", data_files=clean_path('training/datasets', f'{eval_dataset}.json'))
            eval_data = eval_data['train'].map(generate_and_tokenize_prompt, new_fingerprint='%030x' % random.randrange(16**30))

    # == We MUST reload model if it went through any previous training, even failed one ==
    if shared.model_dirty_from_training:
        selected_model = shared.model_name
        if selected_model:
            print("\033[1;31;1m(模型已被之前的训练修改，需要重新加载...)\033[0;37;0m")
            try:
                yield f"正在重新加载 {selected_model}..."
                reload_model()
                if shared.model is not None:
                    print("模型重新加载成功，继续训练。")
                else:
                    return f"加载 {selected_model} 失败。"
            except:
                exc = traceback.format_exc()
                logger.error('重新加载模型失败。')
                print(exc)
                return exc.replace('\n', '\n\n')

    # == Start prepping the model itself ==
    if not hasattr(shared.model, 'lm_head') or hasattr(shared.model.lm_head, 'weight'):
        logger.info("正在准备模型")
        if 'quantization_config' in shared.model.config.to_dict():
            prepare_model_for_kbit_training(shared.model)

    # base model is now frozen and should not be reused for any other LoRA training than this one
    shared.model_dirty_from_training = True

    logger.info("正在准备训练")
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=list_target_modules(model_id),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # == Backup the existing adapter ==
    if not always_override:
        backup_adapter(lora_file_path)

    # == get model trainable params
    model_trainable_params, model_all_params = calc_trainable_parameters(shared.model)

    try:
        logger.info("正在创建LoRA模型")
        lora_model = get_peft_model(shared.model, config)
        if not always_override and Path(f"{lora_file_path}/adapter_model.bin").is_file():
            logger.info("正在加载现有的LoRA数据")
            state_dict_peft = torch.load(f"{lora_file_path}/adapter_model.bin", weights_only=True)
            set_peft_model_state_dict(lora_model, state_dict_peft)
    except:
        yield traceback.format_exc().replace('\n', '\n\n')
        return

    if shared.args.monkey_patch:
        from alpaca_lora_4bit.autograd_4bit import Autograd4bitQuantLinear
        from alpaca_lora_4bit.models import Linear4bitLt
        for _, m in lora_model.named_modules():
            if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
                if m.is_v1_model:
                    m.zeros = m.zeros.half()
                m.scales = m.scales.half()

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
            elif state.global_step > 0 and actual_save_steps > 0 and state.global_step % actual_save_steps == 0:
                lora_model.save_pretrained(f"{lora_file_path}/checkpoint-{tracked.current_steps}/")
                # Save log
                with open(f"{lora_file_path}/checkpoint-{tracked.current_steps}/training_log.json", 'w', encoding='utf-8') as file:
                    json.dump(train_log, file, indent=2)
                # == Save training prompt ==
                with open(f"{lora_file_path}/checkpoint-{tracked.current_steps}/training_prompt.json", 'w', encoding='utf-8') as file:
                    json.dump(train_template, file, indent=2)

        def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps += 1
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

        def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, logs, **kwargs):
            train_log.update(logs)
            train_log.update({"current_steps": tracked.current_steps})
            if WANT_INTERRUPT:
                print("\033[1;31;1m用户中断\033[0;37;0m")

            print(f"\033[1;30;40m步数：{tracked.current_steps} \033[0;37;0m", end='')
            if 'loss' in logs:
                loss = float(logs['loss'])
                if loss <= stop_at_loss:
                    control.should_epoch_stop = True
                    control.should_training_stop = True
                    print(f"\033[1;31;1m达到停止损失 {stop_at_loss}。\033[0;37;0m")

    # Fix training for mixed precision models
    for param in shared.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainer = transformers.Trainer(
        model=lora_model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            report_to=report_to if report_to != "None" else None,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=math.ceil(warmup_steps / gradient_accumulation_steps),
            num_train_epochs=epochs,
            learning_rate=actual_lr,
            fp16=False if shared.args.cpu or shared.args.bf16 else True,
            bf16=shared.args.bf16,
            optim=optimizer,
            logging_steps=2 if stop_at_loss > 0 else 5,
            evaluation_strategy="steps" if eval_data is not None else "no",
            eval_steps=math.ceil(eval_steps / gradient_accumulation_steps) if eval_data is not None else None,
            save_strategy="steps" if eval_data is not None else "no",
            output_dir=lora_file_path,
            lr_scheduler_type=lr_scheduler_type,
            load_best_model_at_end=eval_data is not None,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None,
            no_cuda=shared.args.cpu,
            use_ipex=True if is_torch_xpu_available() and not shared.args.cpu else False
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(shared.tokenizer, mlm=False),
        callbacks=list([Callbacks()])
    )

    lora_model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        lora_model = torch.compile(lora_model)

    # == Save parameters for reuse ==
    with open(f"{lora_file_path}/training_parameters.json", 'w', encoding='utf-8') as file:
        vars = locals()
        json.dump({x: vars[x] for x in PARAMETERS}, file, indent=2)

    # == Save training prompt ==
    with open(f"{lora_file_path}/training_prompt.json", 'w', encoding='utf-8') as file:
        json.dump(train_template, file, indent=2)

    # == Main run and monitor loop ==
    logger.info("正在开始训练")
    yield "正在开始..."

    lora_trainable_param, lora_all_param = calc_trainable_parameters(lora_model)

    projections_string = ", ".join([projection.replace("_proj", "") for projection in list_target_modules(model_id)])

    print(f"使用（{projections_string}）投影来训练 '{model_id}' 模型")

    if lora_all_param > 0:
        print(f"可训练参数：{lora_trainable_param:,d} ({100 * lora_trainable_param / lora_all_param:.4f} %), 所有参数：{lora_all_param:,d} (模型：{model_all_params:,d})")

    train_log.update({"base_model_name": shared.model_name})
    train_log.update({"base_model_class": shared.model.__class__.__name__})
    train_log.update({"base_loaded_in_4bit": getattr(lora_model, "is_loaded_in_4bit", False)})
    train_log.update({"base_loaded_in_8bit": getattr(lora_model, "is_loaded_in_8bit", False)})
    train_log.update({"projections": projections_string})

    if stop_at_loss > 0:
        print(f"正在监控损失 \033[1;31;1m(自动停止于：{stop_at_loss})\033[0;37;0m")

    if WANT_INTERRUPT:
        yield "在开始之前被中断。"
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
            Path('logs').mkdir(exist_ok=True)
            with open(Path('logs/train_dataset_sample.json'), 'w') as json_file:
                json.dump(decoded_entries, json_file, indent=4)

            logger.info("日志文件 'train_dataset_sample.json' 已在 'logs' 目录中创建。")
        except Exception as e:
            logger.error(f"由于错误 {e} 创建日志文件失败")

    def threaded_run():
        log_train_dataset(trainer)
        trainer.train()
        # Note: save in the thread in case the gradio thread breaks (eg browser closed)
        lora_model.save_pretrained(lora_file_path)
        logger.info("LoRA训练运行已完成并保存。")
        # Save log
        with open(f"{lora_file_path}/training_log.json", 'w', encoding='utf-8') as file:
            json.dump(train_log, file, indent=2)

    thread = threading.Thread(target=threaded_run)
    thread.start()
    last_step = 0
    start_time = time.perf_counter()

    while thread.is_alive():
        time.sleep(0.5)
        if WANT_INTERRUPT:
            yield "正在中断，请等待... *(运行将在当前训练步骤完成后停止。)*"

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

            yield f"运行中... **{tracked.current_steps}** / **{tracked.max_steps}** ... {timer_info}, {format_time(time_elapsed)} / {format_time(total_time_estimate)} ... {format_time(total_time_estimate - time_elapsed)} 剩余"

    # Saving in the train thread might fail if an error occurs, so save here if so.
    if not tracked.did_save:
        logger.info("训练完成，正在保存")
        lora_model.save_pretrained(lora_file_path)

    if WANT_INTERRUPT:
        logger.info("训练被中断。")
        yield f"已中断。未完成的LoRA已保存到 `{lora_file_path}`。"
    else:
        logger.info("训练完成！")
        yield f"完成！LoRA已保存到 `{lora_file_path}`。\n\n在测试您的新LoRA之前，请确保首先重新加载模型，因为它目前正因训练而处于脏乱状态。"


def split_chunks(arr, size, step):
    for i in range(0, len(arr), step):
        yield arr[i:i + size]


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
        return f"`{seconds:.0f}` 秒"

    minutes = seconds / 60
    if minutes < 120:
        return f"`{minutes:.0f}` 分钟"

    hours = minutes / 60
    return f"`{hours:.0f}` 小时"
