import importlib.util
if not importlib.util.find_spec("mamba_ssm"):
    class MambaTrainer:
        pass

    class MambaSsmConfig:
        pass

    class MambaSsmModel:
        def __init__(self):
            raise NotImplementedError("MambaSsmModel is currently only supported on Linux with CUDA.")
        pass
else:
    import torch
    from transformers import AutoTokenizer, Trainer
    from modules.logging_colors import logger
    from modules.callbacks import Iteratorize
    from typing import Any, Dict
    import json
    import os
    import mamba_ssm.models.config_mamba
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    class MambaTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            input_ids = inputs.pop("input_ids")
            lm_logits = model(input_ids).logits

            labels = input_ids.to(lm_logits.device)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

            return lm_loss

        def save_model(self, output_dir, _internal_call: bool = False):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            open(f"{output_dir}/config.json", "w").write(self.model.config.to_json_string())
            torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
            self.tokenizer.save_pretrained(output_dir)

    class MambaSsmConfig(mamba_ssm.models.config_mamba.MambaConfig):

        def to_json_string(self, use_diff: bool = True) -> str:
            """
            Serializes this instance to a JSON string.

            Args:
                use_diff (`bool`, *optional*, defaults to `True`):
                    If set to `True`, only the difference between the config instance and the default `MambaSsmConfig()`
                    is serialized to JSON string.

            Returns:
                `str`: String containing all the attributes that make up this configuration instance in JSON format.
            """
            data = {
                'd_model': self.d_model,
                'n_layer': self.n_layer,
                'vocab_size': self.vocab_size,
                'ssm_cfg': self.ssm_cfg,
                'rms_norm': self.rms_norm,
                'residual_in_fp32': self.residual_in_fp32,
                'fused_add_norm': self.fused_add_norm,
                'pad_vocab_size_multiple': self.pad_vocab_size_multiple,
            }
            if use_diff:
                default_mamba_config = mamba_ssm.models.config_mamba.MambaConfig()
                default_mamba_ssm_config = MambaSsmConfig().from_mamba_config(default_mamba_config).to_dict()
                for key in data:
                    if data[key] == default_mamba_ssm_config[key]:
                        del data[key]
            return json.dumps(data)

        def to_dict(self) -> Dict[str, Any]:
            """
            Serializes this instance to a Python dictionary.

            Returns:
                `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
            """
            data = {
                'd_model': self.d_model,
                'n_layer': self.n_layer,
                'vocab_size': self.vocab_size,
                'ssm_cfg': self.ssm_cfg,
                'rms_norm': self.rms_norm,
                'residual_in_fp32': self.residual_in_fp32,
                'fused_add_norm': self.fused_add_norm,
                'pad_vocab_size_multiple': self.pad_vocab_size_multiple,
            }
            return data

        def from_mamba_config(self, mamba_config):
            self.d_model = mamba_config.d_model
            self.n_layer = mamba_config.n_layer
            self.vocab_size = mamba_config.vocab_size
            self.ssm_cfg = mamba_config.ssm_cfg
            self.rms_norm = mamba_config.rms_norm
            self.residual_in_fp32 = mamba_config.residual_in_fp32
            self.fused_add_norm = mamba_config.fused_add_norm
            self.pad_vocab_size_multiple = mamba_config.pad_vocab_size_multiple

    class MambaSsmModel:
        __module__ = 'torch'

        def __init__(self):
            pass

        def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
            pass

        @classmethod
        def from_pretrained(self, path_to_model):

            model = MambaLMHeadModel.from_pretrained(path_to_model, dtype=torch.bfloat16, device="cuda")
            mamba_ssm_config = MambaSsmConfig()
            mamba_ssm_config.from_mamba_config(model.config)
            model.config = mamba_ssm_config

            tokenizer_file = os.path.join(path_to_model, 'tokenizer.json')
            if os.path.isfile(tokenizer_file):
                tokenizer = AutoTokenizer.from_pretrained(path_to_model)
            else:
                tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

            result = self()
            result.model = model
            result.config = model.config
            result.cache = None
            result.tokenizer = tokenizer
            result.generator = None
            result.loras = None
            return result, tokenizer

        def encode(self, string, **kwargs):
            return self.tokenizer.encode(string, return_tensors='pt')

        def decode(self, ids, **kwargs):
            return self.tokenizer.decode(ids, decode_special_tokens=True)

        def get_logits(self, token_ids, **kwargs):
            logger.debug("mamba: token_ids %s", token_ids)
            output = self.model.generate(
                input_ids=token_ids.cuda(),
                max_length=8,
                cg=True,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
            )
            logger.debug("mamba: output %s", output)
            logger.debug("mamba: output.scores %s", output.scores)
            scores = output.scores[0]
            logger.debug("scores %s", scores)
            logger.debug("scores[-1] %s", scores[-1])
            logger.debug("scores[-1][-1] %s", scores[-1][-1])

            raise NotImplementedError("logit results look wrong right now")
            return scores

        def generate(self, prompt, state, callback=None):
            input_ids = self.encode(prompt)
            initial_len = len(input_ids[0])

            output = self.model.generate(
                input_ids=input_ids.cuda(),
                max_length=state['max_new_tokens'] + initial_len,
                cg=True,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                temperature=state['temperature'],
                top_k=state['top_k'],
                top_p=state['top_p'],
            )
            decoded = self.decode(output.sequences.cpu()[0][initial_len:])
            logger.debug("decoded %s", decoded)
            callback(decoded)
            return decoded

        def generate_with_streaming(self, *args, **kwargs):
            with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
                reply = ''
                for token in generator:
                    reply += token
                    yield reply

        def to(self, *args, **kwargs):
            self.model.to(*args, **kwargs)
