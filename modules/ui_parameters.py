from pathlib import Path

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui(default_preset):
    mu = shared.args.multi_user
    generate_params = presets.load_preset(default_preset)
    with gr.Tab("Parameters", elem_id="parameters"):
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset, label='Preset', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['random_preset'] = gr.Button('🎲', elem_classes='refresh-button')

                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(label="Filter by loader", choices=["All"] + list(loaders.loaders_and_params.keys()), value="All", elem_classes='slim-dropdown')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                            shared.gradio['temperature'] = gr.Slider(0.01, 5, value=generate_params['temperature'], step=0.01, label='temperature')
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p')
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k')
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p')
                            shared.gradio['min_p'] = gr.Slider(0.0, 1.0, value=generate_params['min_p'], step=0.01, label='min_p')
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty')
                            shared.gradio['frequency_penalty'] = gr.Slider(0, 2, value=generate_params['frequency_penalty'], step=0.05, label='frequency_penalty')
                            shared.gradio['presence_penalty'] = gr.Slider(0, 2, value=generate_params['presence_penalty'], step=0.05, label='presence_penalty')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range')
                            shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')

                            with gr.Blocks():
                                shared.gradio['dry_multiplier'] = gr.Slider(0, 5, value=generate_params['dry_multiplier'], step=0.01, label='dry_multiplier', info='Set to value > 0 to enable DRY. Controls the magnitude of the penalty for the shortest penalized sequences.')
                                shared.gradio['dry_base'] = gr.Slider(1, 4, value=generate_params['dry_base'], step=0.01, label='dry_base', info='Controls how fast the penalty grows with increasing sequence length.')
                                shared.gradio['dry_allowed_length'] = gr.Slider(1, 20, value=generate_params['dry_allowed_length'], step=1, label='dry_allowed_length', info='Longest sequence that can be repeated without being penalized.')
                                shared.gradio['dry_sequence_breakers'] = gr.Textbox(value=generate_params['dry_sequence_breakers'], label='dry_sequence_breakers', info='Tokens across which sequence matching is not continued. Specified as a comma-separated list of quoted strings.')

                            gr.Markdown("[Learn more](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab)")

                        with gr.Column():
                            with gr.Group():
                                shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='Expand max_new_tokens to the available context length.')
                                shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.')
                                shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Add the bos_token to the beginning of prompts', info='Disabling this can make the replies more creative.')
                                shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=2, value=shared.settings["custom_stopping_strings"] or None, label='Custom stopping strings', info='Written between "" and separated by commas.', placeholder='"\\n", "\\nYou:"')
                                shared.gradio['custom_token_bans'] = gr.Textbox(value=shared.settings['custom_token_bans'] or None, label='Token bans', info='Token IDs to ban, separated by commas. The IDs can be found in the Default or Notebook tab.')

                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='For Contrastive Search. do_sample must be unchecked.')
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='For CFG. 1.5 is a good value.')
                            shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label='Negative prompt', lines=3, elem_classes=['add_scrollbar'])
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='mode=1 is for llama.cpp only.')
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff')
                            shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
                            shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')

                with gr.Column():
                    with gr.Row() as shared.gradio['grammar_file_row']:
                        shared.gradio['grammar_file'] = gr.Dropdown(value='None', choices=utils.get_available_grammars(), label='Load grammar from file (.gbnf)', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['grammar_file'], lambda: None, lambda: {'choices': utils.get_available_grammars()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

                    shared.gradio['grammar_string'] = gr.Textbox(value='', label='Grammar', lines=16, elem_classes=['add_scrollbar', 'monospace'])

                    with gr.Row():
                        with gr.Column():
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')
                            shared.gradio['smoothing_factor'] = gr.Slider(0.0, 10.0, value=generate_params['smoothing_factor'], step=0.01, label='smoothing_factor', info='Activates Quadratic Sampling.')
                            shared.gradio['smoothing_curve'] = gr.Slider(1.0, 10.0, value=generate_params['smoothing_curve'], step=0.01, label='smoothing_curve', info='Adjusts the dropoff curve of Quadratic Sampling.')
                            shared.gradio['dynamic_temperature'] = gr.Checkbox(value=generate_params['dynamic_temperature'], label='dynamic_temperature')
                            shared.gradio['dynatemp_low'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_low'], step=0.01, label='dynatemp_low', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_high'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_high'], step=0.01, label='dynatemp_high', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_exponent'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_exponent'], step=0.01, label='dynatemp_exponent', visible=generate_params['dynamic_temperature'])
                            shared.gradio['temperature_last'] = gr.Checkbox(value=generate_params['temperature_last'], label='temperature_last', info='Moves temperature/dynamic temperature/quadratic sampling to the end of the sampler stack, ignoring their positions in "Sampler priority".')
                            shared.gradio['sampler_priority'] = gr.Textbox(value=generate_params['sampler_priority'], lines=12, label='Sampler priority', info='Parameter names separated by new lines or commas.')

                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Number(precision=0, step=256, value=get_truncation_length(), label='Truncate the prompt up to this length', info='The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048.')
                            shared.gradio['prompt_lookup_num_tokens'] = gr.Slider(value=shared.settings['prompt_lookup_num_tokens'], minimum=0, maximum=10, step=1, label='prompt_lookup_num_tokens', info='Activates Prompt Lookup Decoding.')
                            shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='Maximum tokens/second', info='To make text readable in real time.')
                            shared.gradio['max_updates_second'] = gr.Slider(value=shared.settings['max_updates_second'], minimum=0, maximum=24, step=1, label='Maximum UI updates/second', info='Set this if you experience lag in the UI during streaming.')
                            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)')
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.')
                            shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label='Activate text streaming')

        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(loaders.blacklist_samplers, gradio('filter_by_loader', 'dynamic_temperature'), gradio(loaders.list_all_samplers()), show_progress=False)
    shared.gradio['preset_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.load_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()), show_progress=False)

    shared.gradio['random_preset'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.random_preset, gradio('interface_state'), gradio('interface_state') + gradio(presets.presets_params()), show_progress=False)

    shared.gradio['grammar_file'].change(load_grammar, gradio('grammar_file'), gradio('grammar_string'), show_progress=False)
    shared.gradio['dynamic_temperature'].change(lambda x: [gr.update(visible=x)] * 3, gradio('dynamic_temperature'), gradio('dynatemp_low', 'dynatemp_high', 'dynatemp_exponent'), show_progress=False)


def get_truncation_length():
    if 'max_seq_len' in shared.provided_arguments or shared.args.max_seq_len != shared.args_defaults.max_seq_len:
        return shared.args.max_seq_len
    elif 'n_ctx' in shared.provided_arguments or shared.args.n_ctx != shared.args_defaults.n_ctx:
        return shared.args.n_ctx
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r', encoding='utf-8').read()
    else:
        return ''
