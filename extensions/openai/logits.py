from extensions.openai.completions import process_parameters
from modules.logits import get_next_logits


def _get_next_logits(body):
    # Pre-process the input payload to simulate a real generation
    use_samplers = body['use_samplers']
    state = process_parameters(body) if use_samplers else {}
    state['stream'] = True

    return get_next_logits(body['prompt'], state, use_samplers, "", return_dict=True)
