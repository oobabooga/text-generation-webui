from pathlib import Path

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui(default_preset):
    generate_params = presets.load_preset(default_preset)
    with gr.Tab("Parameters", elem_id="parameters"):
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset, label='Preset', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button')
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button')
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button')

                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(label="Filter by loader", choices=["All"] + list(loaders.loaders_and_params.keys()), value="All", elem_classes='slim-dropdown')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                            shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature')
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p')
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k')
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range')
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p')
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff')

                        with gr.Column():
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='For CFG. 1.5 is a good value.')
                            shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label='Negative prompt', lines=3, elem_classes=['add_scrollbar'])
                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='For Contrastive Search. do_sample must be unchecked.')
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='mode=1 is for llama.cpp only.')
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')
                            shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
                            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)')
                            with gr.Accordion('Other parameters', open=False):
                                shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
                                shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                                shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'], label='min_length')
                                shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams', info='For Beam Search, along with length_penalty and early_stopping.')
                                shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
                                shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')

                    with gr.Accordion("Learn more", open=False):
                        gr.Markdown("""

            For a technical description of the parameters, the [transformers documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) is a good reference.

            The best presets, according to the [Preset Arena](https://github.com/oobabooga/oobabooga.github.io/blob/main/arena/results.md) experiment, are:

            * Instruction following:
                1) Divine Intellect
                2) Big O
                3) simple-1
                4) Space Alien
                5) StarChat
                6) Titanic
                7) tfs-with-top-a
                8) Asterism
                9) Contrastive Search

            * Chat:
                1) Midnight Enigma
                2) Yara
                3) Shortwave

            ### Temperature
            Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.
            ### top_p
            If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.
            ### top_k
            Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.
            ### typical_p
            If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.
            ### epsilon_cutoff
            In units of 1e-4; a reasonable value is 3. This sets a probability floor below which tokens are excluded from being sampled. Should be used with top_p, top_k, and eta_cutoff set to 0.
            ### eta_cutoff
            In units of 1e-4; a reasonable value is 3. Should be used with top_p, top_k, and epsilon_cutoff set to 0.
            ### repetition_penalty
            Exponential penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.
            ### repetition_penalty_range
            The number of most recent tokens to consider for repetition penalty. 0 makes all tokens be used.
            ### encoder_repetition_penalty
            Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.
            ### no_repeat_ngram_size
            If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.
            ### min_length
            Minimum generation length in tokens.
            ### penalty_alpha
            Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.

                        """, elem_classes="markdown")

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Slider(value=get_truncation_length(), minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=256, label='Truncate the prompt up to this length', info='The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048.')
                            shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='Maximum number of tokens/second', info='To make text readable in real time.')
                            shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=1, value=shared.settings["custom_stopping_strings"] or None, label='Custom stopping strings', info='In addition to the defaults. Written between "" and separated by commas.', placeholder='"\\n", "\\nYou:"')
                            shared.gradio['custom_token_bans'] = gr.Textbox(value=shared.settings['custom_token_bans'] or None, label='Custom token bans', info='Specific token IDs to ban from generating, comma-separated. The IDs can be found in the Default or Notebook tab.')

                        with gr.Column():
                            shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='Expand max_new_tokens to the available context length.')
                            shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.')
                            shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Add the bos_token to the beginning of prompts', info='Disabling this can make the replies more creative.')
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.')
                            shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label='Activate text streaming')

                            with gr.Row() as shared.gradio['grammar_file_row']:
                                shared.gradio['grammar_file'] = gr.Dropdown(value='None', choices=utils.get_available_grammars(), label='Load grammar from file (.gbnf)', elem_classes='slim-dropdown')
                                ui.create_refresh_button(shared.gradio['grammar_file'], lambda: None, lambda: {'choices': utils.get_available_grammars()}, 'refresh-button')
                                shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button')
                                shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button')

                    shared.gradio['grammar_string'] = gr.Textbox(value='', label='Grammar', lines=16, elem_classes=['add_scrollbar', 'monospace'])

        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(loaders.blacklist_samplers, gradio('filter_by_loader'), gradio(loaders.list_all_samplers()), show_progress=False)
    shared.gradio['preset_menu'].change(presets.load_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()))
    shared.gradio['grammar_file'].change(load_grammar, gradio('grammar_file'), gradio('grammar_string'))


def get_truncation_length():
    if shared.args.max_seq_len != shared.args_defaults.max_seq_len:
        return shared.args.max_seq_len
    if shared.args.n_ctx != shared.args_defaults.n_ctx:
        return shared.args.n_ctx
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r').read()
    else:
        return ''
