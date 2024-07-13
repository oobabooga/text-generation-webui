import json
import math
import pprint

import torch
import transformers
from transformers import LogitsWarper, is_torch_xpu_available
from transformers.generation.logits_process import (
    LogitNormalization,
    LogitsProcessor,
    LogitsProcessorList
)

from modules import shared
from modules.logging_colors import logger

global_scores = None


class TemperatureLogitsWarperCustom(LogitsWarper):
    '''
    A copy of the original Transformers temperature logits warper.
    '''

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."

            raise ValueError(except_msg)

        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class DynamicTemperatureLogitsWarper(LogitsWarper):
    '''
    Dynamic temperature.
    '''

    def __init__(self, dynatemp_low: float, dynatemp_high: float, dynatemp_exponent: float):
        self.dynatemp_low = dynatemp_low
        self.dynatemp_high = dynatemp_high
        self.dynatemp_exponent = dynatemp_exponent

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        min_temp = self.dynatemp_low
        max_temp = self.dynatemp_high
        exponent_val = self.dynatemp_exponent

        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)

        # Calculate entropy of the softmax probabilities
        entropy = -1.0 * torch.where(probs > 0, probs * torch.log(probs), torch.zeros_like(probs)).sum()

        # Guard against future possible division by zero
        entropy = max(entropy, torch.tensor(1e-10))  # Ensures entropy is slightly greater than 0

        # Any logits which are not -Infinity will be considered for calculating max entropy.
        num_valid_tokens = torch.sum(scores > -float('inf')).item()

        # Now, calculate the max entropy by using only the valid tokens' count
        max_entropy = math.log(num_valid_tokens)

        # Guard against future possible division by zero
        max_entropy = max_entropy if max_entropy > 0.0 else 1e-10

        # Normalize the entropy
        normalized_entropy = entropy / max_entropy

        # Map the normalized entropy to the desired temperature range using the power function
        dyn_temp = min_temp + (max_temp - min_temp) * (normalized_entropy.pow(exponent_val))

        # Apply the dynamically calculated temperature scaling
        scores = scores / dyn_temp

        # print("----------------------\nTemperature from generation_config:", self.temperature)
        # print("min_temp:", min_temp)
        # print("max_temp:", max_temp)
        # print("Entropy:", entropy.item())
        # print("Max Possible Entropy considering valid tokens only:", max_entropy)
        # print("Normalized Entropy:", normalized_entropy.item())
        # print("Dynamic Temperature (dyn_temp):", dyn_temp.item())
        # print("----------------------")

        # max_prob_token_id = torch.argmax(scores, dim=-1)  # Get the token ID with the highest probability
        # max_prob_token = shared.tokenizer.convert_ids_to_tokens(int(max_prob_token_id))  # Convert ID to token
        # print("--- T=", float(dyn_temp), "token=", max_prob_token, "min=", min_temp, "max=", max_temp, "exponent=", exponent_val)

        return scores


class QuadraticSamplingLogitsWarper(LogitsWarper):
    '''
    Quadratic sampling with smoothing factor and smoothing curve parameters.
    '''

    def __init__(self, smoothing_factor, smoothing_curve):
        self.smoothing_factor = smoothing_factor
        self.smoothing_curve = smoothing_curve

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # Compute necessary values
        max_logit = scores.max()
        diff = scores - max_logit
        k = (3 - self.smoothing_curve) / 2
        s = (self.smoothing_curve - 1) / 2

        # Apply transformation to non-negative infinity values
        transformed_logits = torch.where(
            scores != float('-inf'),
            -(k * self.smoothing_factor * diff**2) + (s * self.smoothing_factor * diff**3) + max_logit,
            scores
        )

        return transformed_logits


class TailFreeLogitsWarper(LogitsWarper):
    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(self, top_a: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class DRYLogitsProcessor(LogitsProcessor):
    def __init__(self, multiplier: float, base: float, allowed_length: int, sequence_breakers: set[int], _range: int):
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.sequence_breakers = sequence_breakers
        self._range = _range

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._range > 0:
            input_ids = input_ids[:, -self._range:]

        for input_ids_row, scores_row in zip(input_ids, scores):
            # Use normal Python data types for improved performance
            input_ids = input_ids_row.tolist()

            last_token = input_ids[-1]
            if last_token in self.sequence_breakers:
                continue

            # Exclude the last token as it always matches.
            match_indices = []
            for idx, val in enumerate(input_ids[:-1]):
                if val == last_token:
                    match_indices.append(idx)

            # Stores the maximum matching sequence length
            # for each token immediately following the sequence in the input.
            match_lengths = {}

            for i in match_indices:
                next_token = input_ids[i + 1]

                if next_token in self.sequence_breakers:
                    continue

                # We have already found that `last_token` matches at this index,
                # so the match is at least of length 1.
                match_length = 1

                # Extend the match backwards (at most to 50 to prevent exponent overflow at penalty calculation) (this cap also improves performance on worst case)
                while match_length < 50:
                    j = i - match_length
                    if j < 0:
                        # Start of input reached.
                        break

                    previous_token = input_ids[-(match_length + 1)]
                    if input_ids[j] != previous_token:
                        # Start of match reached.
                        break

                    if previous_token in self.sequence_breakers:
                        # Sequence-breaking token reached.
                        break

                    match_length += 1

                if next_token in match_lengths:
                    match_lengths[next_token] = max(match_length, match_lengths[next_token])
                else:
                    match_lengths[next_token] = match_length

            # Apply penalties.
            for token, match_length in match_lengths.items():
                if match_length >= self.allowed_length:
                    penalty = self.multiplier * self.base ** (match_length - self.allowed_length)
                    scores_row[token] -= penalty

        return scores


class MirostatLogitsWarper(LogitsWarper):
    def __init__(self, mirostat_mode: int, mirostat_tau: float, mirostat_eta: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if mirostat_mode not in [2]:
            raise ValueError(f"`mirostat` has to be a an integer 2, but is {mirostat_mode}")

        self.mirostat_mode = mirostat_mode
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mu = 2 * self.mirostat_tau
        self.e = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates

        # Truncate the words with surprise values greater than mu
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if (i == 0):
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break

        # Normalize the probabilities of the remaining words
        if is_torch_xpu_available():
            prob_topk = torch.softmax(sorted_logits, dim=0).to("xpu")
            prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True).to("xpu")
        else:
            prob_topk = torch.softmax(sorted_logits, dim=0).to('cuda')
            prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True).to('cuda')

        observed_surprise = -math.log2(prob_topk[prev_i])
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0))
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class SpyLogitsWarper(LogitsWarper):
    def __init__(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        global global_scores
        global_scores = scores
        return scores


class RepetitionPenaltyLogitsProcessorWithRange(LogitsProcessor):
    '''
    Copied from the transformers library
    '''

    def __init__(self, penalty: float, presence_penalty: float, frequency_penalty: float, _range: int):
        if not (penalty > 0):
            raise ValueError(f"`penalty` has to be strictly positive, but is {penalty}")

        self.penalty = penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self._range = _range

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids[:, -self._range:]

        # We loop here because torch.unique() needs to process each row separately in the
        # case that batch_size > 1.
        for input_ids_row, scores_row in zip(input_ids, scores):
            unique_ids, counts = torch.unique(input_ids_row, return_counts=True)
            score = torch.gather(scores_row, 0, unique_ids)

            # multiplicative repetition penalty
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(score < 0, score * self.penalty, score / self.penalty)
            scores_row.scatter_(0, unique_ids, score)

            # presence_penalty and frequency_penalty
            raw_presence_penalty = (counts > 0).to(scores.dtype)
            raw_frequency_penalty = counts.to(scores.dtype)
            additive_penalty = raw_presence_penalty * self.presence_penalty + raw_frequency_penalty * self.frequency_penalty
            scores_row.scatter_add_(0, unique_ids, -additive_penalty)

        return scores


def get_logits_warper_patch(self, generation_config, **kwargs):

    # Parameter sanitization
    if isinstance(generation_config.temperature, int):
        generation_config.temperature = float(generation_config.temperature)  # Must be float

    # Get the original warpers
    warpers = self._get_logits_warper_old(generation_config, **kwargs)

    # Replace temperature with our modified class.
    # Currently, it behaves identically to the original.
    for i in range(len(warpers)):
        if warpers[i].__class__.__name__ == 'TemperatureLogitsWarper':
            warpers[i] = TemperatureLogitsWarperCustom(
                generation_config.temperature,
            )

    # Add custom warpers
    warpers_to_add = LogitsProcessorList()
    min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1
    if generation_config.tfs is not None and 0.0 <= generation_config.tfs < 1.0:
        warpers_to_add.append(
            TailFreeLogitsWarper(
                tfs=generation_config.tfs,
                min_tokens_to_keep=min_tokens_to_keep
            )
        )

    if generation_config.top_a is not None and 0.0 < generation_config.top_a <= 1.0:
        warpers_to_add.append(
            TopALogitsWarper(
                top_a=generation_config.top_a,
                min_tokens_to_keep=min_tokens_to_keep
            )
        )

    if generation_config.dynamic_temperature:
        warpers_to_add.append(
            DynamicTemperatureLogitsWarper(
                dynatemp_low=generation_config.dynatemp_low,
                dynatemp_high=generation_config.dynatemp_high,
                dynatemp_exponent=generation_config.dynatemp_exponent,
            )
        )

    if generation_config.smoothing_factor > 0:
        warpers_to_add.append(
            QuadraticSamplingLogitsWarper(
                smoothing_factor=generation_config.smoothing_factor,
                smoothing_curve=generation_config.smoothing_curve
            )
        )

    if generation_config.mirostat_mode is not None and generation_config.mirostat_mode == 2:
        warpers_to_add.append(
            MirostatLogitsWarper(
                mirostat_mode=generation_config.mirostat_mode,
                mirostat_eta=generation_config.mirostat_eta,
                mirostat_tau=generation_config.mirostat_tau,
                min_tokens_to_keep=min_tokens_to_keep
            )
        )

    if len(warpers) > 0 and isinstance(warpers[-1], LogitNormalization):
        normalize = warpers.pop(-1)
    else:
        normalize = None

    warpers += warpers_to_add

    # Sort the samplers.
    sampler_priority = generation_config.sampler_priority

    # Handle temperature_last
    if generation_config.temperature_last:
        for param_name in ['temperature', 'dynamic_temperature', 'quadratic_sampling']:
            if param_name in sampler_priority:
                if param_name in sampler_priority:
                    index = sampler_priority.index(param_name)
                    sampler_priority.append(sampler_priority.pop(index))
                else:
                    sampler_priority.append(param_name)

    class_name_to_nickname = {
        'DynamicTemperatureLogitsWarper': 'dynamic_temperature',
        'EpsilonLogitsWarper': 'epsilon_cutoff',
        'EtaLogitsWarper': 'eta_cutoff',
        'MinPLogitsWarper': 'min_p',
        'MirostatLogitsWarper': 'mirostat',
        'QuadraticSamplingLogitsWarper': 'quadratic_sampling',
        'TailFreeLogitsWarper': 'tfs',
        'TemperatureLogitsWarperCustom': 'temperature',
        'TopALogitsWarper': 'top_a',
        'TopKLogitsWarper': 'top_k',
        'TopPLogitsWarper': 'top_p',
        'TypicalLogitsWarper': 'typical_p'
    }

    def custom_sort_key(obj):
        class_name = obj.__class__.__name__

        # Return a large value if class name is not mapped or if the mapped nickname is not in priority
        if class_name not in class_name_to_nickname or class_name_to_nickname[class_name] not in sampler_priority:
            return float('inf')

        # Return the index of the nickname in the priority list for sorting
        return sampler_priority.index(class_name_to_nickname[class_name])

    # Sort the list using the custom key function
    warpers = sorted(warpers, key=custom_sort_key)
    if shared.args.verbose:
        logger.info("WARPERS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint([x.__class__.__name__ for x in warpers])
        print()

    if normalize is not None:
        warpers.append(normalize)

    warpers.append(SpyLogitsWarper())
    warpers = LogitsProcessorList(warpers)
    return warpers


def get_logits_processor_patch(self, **kwargs):
    generation_config = kwargs['generation_config']

    do_rep_pen_hijack = (generation_config.repetition_penalty > 1) or (generation_config.presence_penalty != 0) or (generation_config.frequency_penalty != 0)
    if do_rep_pen_hijack:
        generation_config.repetition_penalty = 1.1  # Set to value > 1 to ensure RepetitionPenaltyLogitsProcessor is created

    result = self._get_logits_processor_old(**kwargs)

    if do_rep_pen_hijack:
        for i in range(len(result)):
            if result[i].__class__.__name__ == 'RepetitionPenaltyLogitsProcessor':
                result[i] = RepetitionPenaltyLogitsProcessorWithRange(
                    generation_config.repetition_penalty,
                    generation_config.presence_penalty,
                    generation_config.frequency_penalty,
                    generation_config.repetition_penalty_range
                )

    if generation_config.dry_multiplier is not None and generation_config.dry_multiplier > 0.0:
        dry_sequence_breakers = generation_config.dry_sequence_breakers

        # Support both JSON array notation and comma-separated strings.
        if not dry_sequence_breakers.startswith("["):
            dry_sequence_breakers = "[" + dry_sequence_breakers + "]"

        sequence_breaker_strings = json.loads(dry_sequence_breakers)
        # Prefix with 'a' to get the correct encoding of the token at the end of a text.
        sequence_breakers = {shared.tokenizer.encode(f'a{s}')[-1] for s in sequence_breaker_strings}

        result.append(
            DRYLogitsProcessor(
                multiplier=generation_config.dry_multiplier,
                base=generation_config.dry_base,
                allowed_length=generation_config.dry_allowed_length,
                sequence_breakers=sequence_breakers,
                _range=generation_config.repetition_penalty_range,
            )
        )

    return result


def generation_config_init_patch(self, **kwargs):
    self.__init___old(**kwargs)
    self.min_p = kwargs.pop("min_p", 0.0)
    self.dynamic_temperature = kwargs.pop("dynamic_temperature", False)
    self.dynatemp_low = kwargs.pop("dynatemp_low", 1)
    self.dynatemp_high = kwargs.pop("dynatemp_high", 1)
    self.dynatemp_exponent = kwargs.pop("dynatemp_exponent", 1)
    self.smoothing_factor = kwargs.pop("smoothing_factor", 0.0)
    self.smoothing_curve = kwargs.pop("smoothing_curve", 1.0)
    self.tfs = kwargs.pop("tfs", 1.0)
    self.top_a = kwargs.pop("top_a", 0.0)
    self.mirostat_mode = kwargs.pop("mirostat_mode", 0)
    self.mirostat_eta = kwargs.pop("mirostat_eta", 0.1)
    self.mirostat_tau = kwargs.pop("mirostat_tau", 5)
    self.repetition_penalty_range = kwargs.pop("repetition_penalty_range", 0)
    self.presence_penalty = kwargs.pop("presence_penalty", 0)
    self.frequency_penalty = kwargs.pop("frequency_penalty", 0)
    self.dry_multiplier = kwargs.pop("dry_multiplier", 0.0)
    self.dry_base = kwargs.pop("dry_base", 1.75)
    self.dry_allowed_length = kwargs.pop("dry_allowed_length", 2)
    self.dry_sequence_breakers = kwargs.pop("dry_sequence_breakers", '"\\n", ":", "\\"", "*"')
    self.temperature_last = kwargs.pop("temperature_last", False)
    self.sampler_priority = kwargs.pop("sampler_priority", ['temperature', 'dynamic_temperature', 'quadratic_sampling', 'top_k', 'top_p', 'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'tfs', 'top_a', 'min_p', 'mirostat'])


def hijack_samplers():
    transformers.GenerationMixin._get_logits_warper_old = transformers.GenerationMixin._get_logits_warper
    transformers.GenerationMixin._get_logits_warper = get_logits_warper_patch

    transformers.GenerationMixin._get_logits_processor_old = transformers.GenerationMixin._get_logits_processor
    transformers.GenerationMixin._get_logits_processor = get_logits_processor_patch

    transformers.GenerationConfig.__init___old = transformers.GenerationConfig.__init__
    transformers.GenerationConfig.__init__ = generation_config_init_patch
