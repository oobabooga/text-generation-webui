import torch.nn.functional as F

from transformers import LogitsWarper, LogitsProcessor

from modules import shared


class CFGLogits(LogitsWarper):
    def __init__(self, cfg, inputs, model, verbose=True):
        self.cfg = cfg
        self.inputs = inputs
        self.model = model
        self.out = None
        self.verbose = verbose

    def __call__(self, input_ids, scores):
        if self.cfg == 1:
            return F.log_softmax(scores, dim=-1)
        scores = F.log_softmax(scores, dim=-1)
        if self.out is None:
            self.out = self.model(self.inputs, use_cache=True)
        else:
            self.out = self.model(input_ids[:, -1:],
                                  use_cache=True,
                                  past_key_values=self.out.past_key_values)
        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        out = self.cfg * (scores - unconditional_logits) + unconditional_logits
        out = F.log_softmax(out, dim=-1)
        return 0.7 * out + 0.3 * scores


def logits_modifier(processor_list, input_ids):
    processor_list.append(CFGLogits(1.5, input_ids[:, -1:], shared.model))
