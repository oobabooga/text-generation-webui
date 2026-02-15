from pathlib import Path

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio
from modules.i18n import t


def create_ui():
    mu = shared.args.multi_user
    with gr.Tab(t("Parameters"), elem_id="parameters"):
        with gr.Tab(t("Generation")):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=shared.settings['preset'], label=t('Preset'), elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['reset_preset'] = gr.Button(t('Restore preset'), elem_classes='refresh-button', interactive=True)
                        shared.gradio['neutralize_samplers'] = gr.Button(t('Neutralize samplers'), elem_classes='refresh-button', interactive=True)

                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(label=t("Filter by loader"), choices=["All"] + list(loaders.loaders_and_params.keys()) if not shared.args.portable else ['llama.cpp'], value="All", elem_classes='slim-dropdown')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(t('## Curve shape'))
                            shared.gradio['temperature'] = gr.Slider(0.01, 5, value=shared.settings['temperature'], step=0.01, label=t('temperature'))
                            shared.gradio['dynatemp_low'] = gr.Slider(0.01, 5, value=shared.settings['dynatemp_low'], step=0.01, label=t('dynatemp_low'), visible=shared.settings['dynamic_temperature'])
                            shared.gradio['dynatemp_high'] = gr.Slider(0.01, 5, value=shared.settings['dynatemp_high'], step=0.01, label=t('dynatemp_high'), visible=shared.settings['dynamic_temperature'])
                            shared.gradio['dynatemp_exponent'] = gr.Slider(0.01, 5, value=shared.settings['dynatemp_exponent'], step=0.01, label=t('dynatemp_exponent'), visible=shared.settings['dynamic_temperature'])
                            shared.gradio['smoothing_factor'] = gr.Slider(0.0, 10.0, value=shared.settings['smoothing_factor'], step=0.01, label='smoothing_factor', info=t('Activates Quadratic Sampling.'))
                            shared.gradio['smoothing_curve'] = gr.Slider(1.0, 10.0, value=shared.settings['smoothing_curve'], step=0.01, label='smoothing_curve', info=t('Adjusts the dropoff curve of Quadratic Sampling.'))
                            shared.gradio['dynamic_temperature'] = gr.Checkbox(value=shared.settings['dynamic_temperature'], label=t('dynamic_temperature'))

                            gr.Markdown(t('## Curve cutoff'))
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=shared.settings['top_p'], step=0.01, label=t('top_p'))
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=shared.settings['top_k'], step=1, label=t('top_k'))
                            shared.gradio['min_p'] = gr.Slider(0.0, 1.0, value=shared.settings['min_p'], step=0.01, label=t('min_p'))
                            shared.gradio['top_n_sigma'] = gr.Slider(0.0, 5.0, value=shared.settings['top_n_sigma'], step=0.01, label=t('top_n_sigma'))
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=shared.settings['typical_p'], step=0.01, label=t('typical_p'))
                            shared.gradio['xtc_threshold'] = gr.Slider(0, 0.5, value=shared.settings['xtc_threshold'], step=0.01, label=t('xtc_threshold'), info=t('If 2 or more tokens have probability above this threshold, consider removing all but the last one.'))
                            shared.gradio['xtc_probability'] = gr.Slider(0, 1, value=shared.settings['xtc_probability'], step=0.01, label=t('xtc_probability'), info=t('Probability that the removal will actually happen. 0 disables the sampler. 1 makes it always happen.'))
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=shared.settings['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=shared.settings['eta_cutoff'], step=0.01, label='eta_cutoff')
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=shared.settings['tfs'], step=0.01, label='tfs')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=shared.settings['top_a'], step=0.01, label='top_a')

                            gr.Markdown(t('## Repetition suppression'))
                            shared.gradio['dry_multiplier'] = gr.Slider(0, 5, value=shared.settings['dry_multiplier'], step=0.01, label=t('dry_multiplier'), info=t('Set to greater than 0 to enable DRY. Recommended value: 0.8.'))
                            shared.gradio['dry_allowed_length'] = gr.Slider(1, 20, value=shared.settings['dry_allowed_length'], step=1, label=t('dry_allowed_length'), info=t('Longest sequence that can be repeated without being penalized.'))
                            shared.gradio['dry_base'] = gr.Slider(1, 4, value=shared.settings['dry_base'], step=0.01, label=t('dry_base'), info=t('Controls how fast the penalty grows with increasing sequence length.'))
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=shared.settings['repetition_penalty'], step=0.01, label=t('repetition_penalty'))
                            shared.gradio['frequency_penalty'] = gr.Slider(0, 2, value=shared.settings['frequency_penalty'], step=0.05, label=t('frequency_penalty'))
                            shared.gradio['presence_penalty'] = gr.Slider(0, 2, value=shared.settings['presence_penalty'], step=0.05, label=t('presence_penalty'))
                            shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=shared.settings['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
                            shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=shared.settings['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=shared.settings['repetition_penalty_range'], label=t('repetition_penalty_range'))

                        with gr.Column():
                            gr.Markdown(t('## Alternative sampling methods'))
                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=shared.settings['penalty_alpha'], label=t('penalty_alpha'), info=t('For Contrastive Search. do_sample must be unchecked.'))
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=shared.settings['guidance_scale'], label=t('guidance_scale'), info=t('For CFG. 1.5 is a good value.'))
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=shared.settings['mirostat_mode'], label=t('mirostat_mode'), info=t('mode=1 is for llama.cpp only.'))
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=shared.settings['mirostat_tau'], label=t('mirostat_tau'))
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=shared.settings['mirostat_eta'], label=t('mirostat_eta'))
                            shared.gradio['adaptive_target'] = gr.Slider(0.0, 1.0, value=shared.settings['adaptive_target'], step=0.01, label=t('adaptive_target'), info=t('Target probability for adaptive-p sampling. Tokens near this probability are favored. 0 disables.'))
                            shared.gradio['adaptive_decay'] = gr.Slider(0.0, 0.99, value=shared.settings['adaptive_decay'], step=0.01, label=t('adaptive_decay'), info=t('EMA decay rate for adaptive-p. Controls history window (~1/(1-decay) tokens). Default: 0.9.'))

                            gr.Markdown(t('## Other options'))
                            shared.gradio['do_sample'] = gr.Checkbox(value=shared.settings['do_sample'], label='do_sample')
                            shared.gradio['temperature_last'] = gr.Checkbox(value=shared.settings['temperature_last'], label=t('temperature_last'), info=t('Moves temperature/dynamic temperature/quadratic sampling to the end of the sampler stack, ignoring their positions in "Sampler priority".'))
                            shared.gradio['sampler_priority'] = gr.DragDrop(value=shared.settings['sampler_priority'], label=t('Sampler priority'), info=t('Parameter names separated by new lines or commas.'), elem_classes=['add_scrollbar'])
                            shared.gradio['dry_sequence_breakers'] = gr.Textbox(value=shared.settings['dry_sequence_breakers'], label=t('dry_sequence_breakers'), info=t('Tokens across which sequence matching is not continued. Specified as a comma-separated list of quoted strings.'))

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            with gr.Blocks():
                                shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], value=shared.settings['max_new_tokens'], step=1, label=t('max_new_tokens'), info=t('⚠️ Setting this too high can cause prompt truncation.'))
                                shared.gradio['prompt_lookup_num_tokens'] = gr.Slider(value=shared.settings['prompt_lookup_num_tokens'], minimum=0, maximum=10, step=1, label='prompt_lookup_num_tokens', info=t('Activates Prompt Lookup Decoding.'))
                                shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label=t('Maximum tokens/second'), info=t('To make text readable in real time.'))

                            shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label=t('auto_max_new_tokens'), info=t('Expand max_new_tokens to the available context length.'))
                            shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label=t('Ban the eos_token'), info=t('Forces the model to never end the generation prematurely.'))
                            shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label=t('Add the bos_token to the beginning of prompts'), info=t('Only applies to text completion (notebook). In chat mode, templates control BOS tokens.'))
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info=t('Some specific models need this unset.'))
                            shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label=t('Activate text streaming'))
                            shared.gradio['static_cache'] = gr.Checkbox(value=shared.settings['static_cache'], label=t('Static KV cache'), info=t('Use a static cache for improved performance.'))

                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Number(precision=0, step=256, value=get_truncation_length(), label=t('Truncate the prompt up to this length'), info=t('The leftmost tokens are removed if the prompt exceeds this length.'))
                            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label=t('Seed (-1 for random)'))
                            shared.gradio['custom_system_message'] = gr.Textbox(value=shared.settings['custom_system_message'], lines=2, label=t('Custom system message'), info=t('If not empty, will be used instead of the default one.'), elem_classes=['add_scrollbar'])
                            shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=2, value=shared.settings["custom_stopping_strings"] or None, label=t('Custom stopping strings'), info=t('Written between \"\" and separated by commas.'), placeholder='"\\n", "\\nYou:"')
                            shared.gradio['custom_token_bans'] = gr.Textbox(value=shared.settings['custom_token_bans'] or None, label=t('Token bans'), info=t('Token IDs to ban, separated by commas. The IDs can be found in the Default or Notebook tab.'))
                            shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label=t('Negative prompt'), info=t('For CFG. Only used when guidance_scale is different than 1.'), lines=3, elem_classes=['add_scrollbar'])
                            with gr.Row() as shared.gradio['grammar_file_row']:
                                shared.gradio['grammar_file'] = gr.Dropdown(value='None', choices=utils.get_available_grammars(), label=t('Load grammar from file (.gbnf)'), elem_classes='slim-dropdown')
                                ui.create_refresh_button(shared.gradio['grammar_file'], lambda: None, lambda: {'choices': utils.get_available_grammars()}, 'refresh-button', interactive=not mu)
                                shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                                shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

                            shared.gradio['grammar_string'] = gr.Textbox(value=shared.settings['grammar_string'], label=t('Grammar'), lines=16, elem_classes=['add_scrollbar', 'monospace'])

        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(loaders.blacklist_samplers, gradio('filter_by_loader', 'dynamic_temperature'), gradio(loaders.list_all_samplers()), show_progress=False)
    shared.gradio['preset_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.load_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()), show_progress=False)

    shared.gradio['reset_preset'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.reset_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()), show_progress=False)

    shared.gradio['neutralize_samplers'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.neutralize_samplers_for_ui, gradio('interface_state'), gradio('interface_state') + gradio(presets.presets_params()), show_progress=False)

    shared.gradio['grammar_file'].change(load_grammar, gradio('grammar_file'), gradio('grammar_string'), show_progress=False)
    shared.gradio['dynamic_temperature'].change(lambda x: [gr.update(visible=x)] * 3, gradio('dynamic_temperature'), gradio('dynatemp_low', 'dynatemp_high', 'dynatemp_exponent'), show_progress=False)


def get_truncation_length():
    if shared.args.ctx_size > 0 and ('ctx_size' in shared.provided_arguments or shared.args.ctx_size != shared.args_defaults.ctx_size):
        return shared.args.ctx_size
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = shared.user_data_dir / 'grammars' / name
    if p.exists():
        return open(p, 'r', encoding='utf-8').read()
    else:
        return ''
