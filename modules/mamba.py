import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
from modules.logging_colors import logger
from modules.callbacks import Iteratorize

from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import json
import os
import re
import warnings
import json
import mamba_ssm.models.config_mamba
from transformers.configuration_utils import PretrainedConfig
__version__ = "0.0.1"

class MambaSsmConfig(mamba_ssm.models.config_mamba.MambaConfig):

    def recursive_diff_dict(dict_a, dict_b, config_obj=None):
        """
        Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
        values from `dict_a` that are different from values in `dict_b`.
        """
        diff = {}
        default = config_obj.__class__().to_dict() if config_obj is not None else {}
        for key, value in dict_a.items():
            obj_value = getattr(config_obj, str(key), None)
            if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
                diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
                if len(diff_value) > 0:
                    diff[key] = diff_value
            elif key not in dict_b or value != dict_b[key] or key not in default or value != default[key]:
                diff[key] = value
        return diff

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        # if use_diff is True:
        #     config_dict = self.to_diff_dict()
        # else:
        #     config_dict = self.to_dict()
        # return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
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
        return json.dumps(data)

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # For nested configs we need to clean the diff recursively
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)

        self.dict_torch_dtype_to_str(serializable_config_dict)

        if "_attn_implementation_internal" in serializable_config_dict:
            del serializable_config_dict["_attn_implementation_internal"]

        return serializable_config_dict

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
        # output = copy.deepcopy(self.__dict__)
        # if hasattr(self.__class__, "model_type"):
        #     output["model_type"] = self.__class__.model_type
        # if "_auto_class" in output:
        #     del output["_auto_class"]
        # if "_commit_hash" in output:
        #     del output["_commit_hash"]
        # if "_attn_implementation_internal" in output:
        #     del output["_attn_implementation_internal"]

        # # Transformers version when serializing the model
        # output["transformers_version"] = __version__

        # for key, value in output.items():
        #     # Deal with nested configs like CLIP
        #     if isinstance(value, PretrainedConfig):
        #         value = value.to_dict()
        #         del value["transformers_version"]

        #     output[key] = value

        # if hasattr(self, "quantization_config"):
        #     output["quantization_config"] = (
        #         self.quantization_config.to_dict()
        #         if not isinstance(self.quantization_config, dict)
        #         else self.quantization_config
        #     )

        #     # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
        #     _ = output.pop("_pre_quantization_dtype", None)

        # self.dict_torch_dtype_to_str(output)

        # return output



class MambaSsmModel:

    def __init__(self):        
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        model = MambaLMHeadModel.from_pretrained(path_to_model, dtype=torch.bfloat16, device="cuda")
        model.config = MambaSsmConfig()
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

        result = self()
        result.model = model
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
        with Iteratorize(self.generate, args, kwargs,
                         callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)