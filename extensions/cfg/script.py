# This is taken from @Vermeille's code from this GitHub Issue: https://github.com/huggingface/transformers/issues/24536
# The included processor can be replaced with the `CFGLogits` processor once it's merged to `transformers`.
import gradio
import torch
from transformers import (LogitsProcessor, LogitsProcessorList,
                          MinLengthLogitsProcessor, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper,
                          TypicalLogitsWarper)
from transformers.generation import LogitNormalization
import torch.nn.functional as F

from modules import shared

params = {
    'cfg': 1.5,
}

class CFGLogits(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `uncond` branch. Finally, according to CFG Rescale, the reweighted logits are interpolated back with weight
    `rescale_factor` the conditional ones to smooth the effect and increase output quality.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        uncond (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary for the unconditional branch.
        model:
            The LM computing the unconditional scores. Supposedly the same as the one computing the conditional scores.
            Both models must use the same tokenizer.
        smooth_factor (float):
            The interpolation weight for CFG Rescale. 1 means no rescaling, 0 reduces to the conditional scores without
            CFG. Turn it lower if the output degenerates. Lower values allow for higher guidance scale.
    """

    def __init__(self, guidance_scale, uncond, model, rescale_factor=1.0):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.model = model
        self.out = None
        self.rescale_factor = rescale_factor

    def __call__(self, input_ids, scores):
        if self.guidance_scale == 1:
            return scores
        scores = F.log_softmax(scores, dim=-1)

        if self.out is None:
            self.out = self.model(self.uncond, attention_mask=torch.ones_like(self.uncond))
        else:
            single_input = input_ids[:, -1:]
            self.out = self.model(single_input, past_key_values=self.out.past_key_values)

        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        out = F.log_softmax(out, dim=-1)
        if self.rescale_factor == 1:
            return out
        return self.rescale_factor * out + (1 - self.rescale_factor) * scores


def logits_processor_modifier(logits_processor_list, input_ids):
    process_ids = input_ids[:, -1:]

    if 'custom string' in params and params['custom string'] != '':
        process_ids = shared.tokenizer.encode(params['custom string'], return_tensors='pt').to(shared.model.device)
    logits_processor_list.append(CFGLogits(params['cfg'], process_ids, shared.model))


def ui():
    negative_prompt = gradio.Textbox(value="", placeholder="Enter what you'd like to bias against", label="Negative Prompt", info='This prompt will be used to bias the output against the concepts expressed herein.')
    cfg_value = gradio.Slider(minimum=0.5, maximum=3.0, value=1.5, step=0.1, label="CFG Value", info='Classifier-Free Guidance value (how much to emphasize the negative prompt)')
    def update_negative_prompt(x):
        params.update({"custom string": x})

    negative_prompt.change(update_negative_prompt, negative_prompt, None)

    def update_cfg_value(x):
        params.update({'cfg': float(x)})

    cfg_value.change(update_cfg_value, cfg_value, None)
