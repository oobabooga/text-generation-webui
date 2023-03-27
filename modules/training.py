import sys, torch, json, threading, time
from pathlib import Path
import gradio as gr
from datasets import load_dataset
import transformers
from modules import ui, shared
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict

CURRENT_STEPS = 0
MAX_STEPS = 0

def get_json_dataset(path: str):
    def get_set():
        return ['None'] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path(path).glob('*.json'))), key=str.lower)
    return get_set

def create_train_interface():
    with gr.Tab('Train LoRA', elem_id='lora-train-tab'):
        loraName = gr.Textbox(label="Name", info="The name of your new LoRA file")
        with gr.Row():
            # TODO: Implement multi-device support.
            microBatchSize = gr.Slider(label='Micro Batch Size', value=4, minimum=1, maximum=128, step=1, info='Per-device batch size (NOTE: multiple devices not yet implemented). Increasing this will increase VRAM usage.')
            batchSize = gr.Slider(label='Batch Size', value=128, minimum=1, maximum=1024, step=4, info='Global batch size. The two batch sizes together determine gradient accumulation (gradientAccum = batch / microBatch). Higher gradient accum values lead to better quality training.')
        with gr.Row():
            epochs = gr.Number(label='Epochs', value=1, info='Number of times every entry in the dataset should be fed into training. So 1 means feed each item in once, 5 means feed it in five times, etc.')
            learningRate = gr.Textbox(label='Learning Rate', value='3e-4', info='Learning rate, in scientific notation. 3e-4 is a good starting base point. 1e-2 is extremely high, 1e-6 is extremely low.')
        # TODO: What is the actual maximum rank? Likely distinct per model. This might be better to somehow be on a log scale.
        loraRank = gr.Slider(label='LoRA Rank', value=8, minimum=1, maximum=1024, step=4, info='LoRA Rank, or dimension count. Higher values produce a larger file with better control over the model\'s content. Smaller values produce a smaller file with less overall control. Small values like 4 or 8 are great for stylistic guidance, high values like 128 or 256 are good for teaching content upgrades. Higher ranks also require higher VRAM.')
        loraAlpha = gr.Slider(label='LoRA Alpha', value=16, minimum=1, maximum=2048, step=4, info='LoRA Alpha. This divided by the rank becomes the scaling of the LoRA. Higher means stronger. A good standard value is twice your Rank.')
        # TODO: Better explain what this does, in terms of real world effect especially.
        loraDropout = gr.Slider(label='LoRA Dropout', minimum=0.0, maximum=1.0, step=0.025, value=0.05, info='Percentage probability for dropout of LoRA layers.')
        cutoffLen = gr.Slider(label='Cutoff Length', minimum=1,maximum=2048, value=256, step=32, info='Cutoff length for text input. Essentially, how long of a line of text to feed in at a time. Higher values require drastically more VRAM.')
        with gr.Row():
            datasetFunction = get_json_dataset('training/datasets')
            dataset = gr.Dropdown(choices=datasetFunction(), value='None', label='Dataset', info='The dataset file to use for training.')
            ui.create_refresh_button(dataset, lambda : None, lambda : {'choices': datasetFunction()}, 'refresh-button')
            evalDataset = gr.Dropdown(choices=datasetFunction(), value='None', label='Evaluation Dataset', info='The dataset file used to evaluate the model after training.')
            ui.create_refresh_button(evalDataset, lambda : None, lambda : {'choices': datasetFunction()}, 'refresh-button')
            formatsFunction = get_json_dataset('training/formats')
            format = gr.Dropdown(choices=formatsFunction(), value='None', label='Data Format', info='The format file used to decide how to format the dataset input.')
            ui.create_refresh_button(format, lambda : None, lambda : {'choices': formatsFunction()}, 'refresh-button')
        startButton = gr.Button("Start LoRA Training")
        output = gr.Markdown(value="(...)")
        startButton.click(do_train, [loraName, microBatchSize, batchSize, epochs, learningRate, loraRank, loraAlpha, loraDropout, cutoffLen, dataset, evalDataset, format], [output])

class Callbacks(transformers.TrainerCallback):
    def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        global CURRENT_STEPS, MAX_STEPS
        CURRENT_STEPS = state.global_step
        MAX_STEPS = state.max_steps

def cleanPath(basePath: str, path: str):
    """"Strips unusual symbols and forcibly builds a path as relative to the intended directory."""
    # TODO: Probably could do with a security audit to guarantee there's no ways this can be bypassed to target an unwanted path.
    # Or swap it to a strict whitelist of [a-zA-Z_0-9]
    path = path.replace('\\', '/').replace('..', '_')
    if basePath is None:
        return path
    return f'{Path(basePath).absolute()}/{path}'

def do_train(loraName: str, microBatchSize: int, batchSize: int, epochs: int, learningRate: float, loraRank: int, loraAlpha: int, loraDropout: float, cutoffLen: int, dataset: str, evalDataset: str, format: str):
    global CURRENT_STEPS, MAX_STEPS
    CURRENT_STEPS = 0
    MAX_STEPS = 0
    yield "Prepping..."
    # == Input validation / processing ==
    # TODO: --lora-dir PR once pulled will need to be applied here
    loraName = f"loras/{cleanPath(None, loraName)}"
    if dataset is None:
        return "**Missing dataset choice input, cannot continue.**"
    if format is None:
        return "**Missing format choice input, cannot continue.**"
    gradientAccumulationSteps = batchSize // microBatchSize
    actualLR = float(learningRate)
    shared.tokenizer.pad_token = 0
    shared.tokenizer.padding_side = "left"
    # == Prep the dataset, format, etc ==
    with open(cleanPath('training/formats', f'{format}.json'), 'r') as formatFile:
        formatData: dict[str, str] = json.load(formatFile)
    def tokenize(prompt):
        result = shared.tokenizer(prompt, truncation=True, max_length=cutoffLen + 1, padding="max_length")
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }
    def generate_prompt(data_point: dict[str, str]):
        for options, data in formatData.items():
            if set(options.split(',')) == set(x[0] for x in data_point.items() if len(x[1].strip()) > 0):
                for key, val in data_point.items():
                    data = data.replace(f'%{key}%', val)
            return data
        raise RuntimeError(f'Data-point "{data_point}" has no keyset match within format "{list(formatData.keys())}"')
    def generate_and_tokenize_prompt(data_point):
        prompt = generate_prompt(data_point)
        return tokenize(prompt)
    print("Loading datasets...")
    data = load_dataset("json", data_files=cleanPath('training/datasets', f'{dataset}.json'))
    train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
    if evalDataset == 'None':
        evalData = None
    else:
        evalData = load_dataset("json", data_files=cleanPath('training/datasets', f'{evalDataset}.json'))
        evalData = evalData['train'].shuffle().map(generate_and_tokenize_prompt)
    # == Start prepping the model itself ==
    if not hasattr(shared.model, 'lm_head') or hasattr(shared.model.lm_head, 'weight'):
        print("Getting model ready...")
        prepare_model_for_int8_training(shared.model)
    print("Prepping for training...")
    config = LoraConfig(
        r=loraRank,
        lora_alpha=loraAlpha,
        # TODO: Should target_modules be configurable?
        target_modules=[ "q_proj", "v_proj" ],
        lora_dropout=loraDropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    loraModel = get_peft_model(shared.model, config)
    trainer = transformers.Trainer(
        model=loraModel,
        train_dataset=train_data,
        eval_dataset=evalData,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=microBatchSize,
            gradient_accumulation_steps=gradientAccumulationSteps,
            # TODO: Should more of these be configurable? Probably.
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=actualLR,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if evalData is not None else "no",
            save_strategy="steps",
            eval_steps=200 if evalData is not None else None,
            save_steps=200,
            output_dir=loraName,
            save_total_limit=3,
            load_best_model_at_end=True if evalData is not None else False,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(shared.tokenizer, mlm=False),
        callbacks=list([Callbacks()])
    )
    loraModel.config.use_cache = False
    old_state_dict = loraModel.state_dict
    loraModel.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(loraModel, type(loraModel))
    if torch.__version__ >= "2" and sys.platform != "win32":
        loraModel = torch.compile(loraModel)
    # == Main run and monitor loop ==
    # TODO: save/load checkpoints to resume from?
    print("Starting training...")
    yield "Starting..."
    def threadedRun():
        trainer.train()
    thread = threading.Thread(target=threadedRun)
    thread.start()
    lastStep = 0
    startTime = time.perf_counter()
    while thread.is_alive():
        time.sleep(0.5)
        if CURRENT_STEPS != lastStep:
            lastStep = CURRENT_STEPS
            timeElapsed = time.perf_counter() - startTime
            if timeElapsed <= 0:
                timerInfo = ""
            else:
                its = CURRENT_STEPS / timeElapsed
                if its > 1:
                    timerInfo = f"`{its:.2f}` it/s"
                else:
                    timerInfo = f"`{1.0/its:.2f}` s/it"
            yield f"Running... **{CURRENT_STEPS}** / **{MAX_STEPS}** ... {timerInfo}, `{timeElapsed:.1f}` seconds"
    print("Training complete, saving...")
    loraModel.save_pretrained(loraName)
    print("Training complete!")
    yield f"Done! LoRA saved to `{loraName}`"
