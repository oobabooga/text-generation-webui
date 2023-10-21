"""
This file is responsible for the UI and how the application interracts with the rest of the system.
"""
import os
from pathlib import Path

# Point to where nltk will find the required data.
os.environ['NLTK_DATA'] = str(Path("extensions/superboogav2/nltk_data").resolve())

import textwrap
import codecs
import gradio as gr

import extensions.superboogav2.parameters as parameters

from modules.logging_colors import logger
from modules import shared

from .utils import create_metadata_source
from .chromadb import make_collector
from .download_urls import feed_url_into_collector
from .data_processor import process_and_add_to_collector
from .benchmark import benchmark
from .optimize import optimize
from .notebook_handler import input_modifier_internal
from .chat_handler import custom_generate_chat_prompt_internal
from .api import APIManager

collector = None
api_manager = None

def setup():
    global collector
    global api_manager
    collector = make_collector()
    api_manager = APIManager(collector)

    if parameters.get_api_on():
        api_manager.start_server(parameters.get_api_port())

def _feed_data_into_collector(corpus):
    yield '### Processing data...'
    process_and_add_to_collector(corpus, collector, False, create_metadata_source('direct-text'))
    yield '### Done.'


def _feed_file_into_collector(file):
    yield '### Reading and processing the input dataset...'
    text = file.decode('utf-8')
    process_and_add_to_collector(text, collector, False, create_metadata_source('file'))
    yield '### Done.'


def _feed_url_into_collector(urls):
    for i in feed_url_into_collector(urls, collector):
        yield i
    yield '### Done.'


def _begin_benchmark():
    score, max_score = benchmark(Path("extensions/superboogav2/benchmark_texts/questions.json"), collector)
    return f'**Score**: {score}/{max_score}'


def _begin_optimization(progress=gr.Progress()):
    return optimize(collector, progress), *_get_optimizable_settings()


def _clear_data():
    collector.clear()
    return "### Data Cleared!"


def _get_optimizable_settings() -> list:
    preprocess_pipeline = []
    if parameters.should_to_lower():
        preprocess_pipeline.append('Lower Cases')
    if parameters.should_remove_punctuation():
        preprocess_pipeline.append('Remove Punctuation')
    if parameters.should_remove_specific_pos():
        preprocess_pipeline.append('Remove Adverbs')
    if parameters.should_remove_stopwords():
        preprocess_pipeline.append('Remove Stop Words')
    if parameters.should_lemmatize():
        preprocess_pipeline.append('Lemmatize')
    if parameters.should_merge_spaces():
        preprocess_pipeline.append('Merge Spaces')
    if parameters.should_strip():
        preprocess_pipeline.append('Strip Edges')
        
    return [
        parameters.get_time_power(),
        parameters.get_time_steepness(),
        parameters.get_significant_level(),
        parameters.get_min_num_sentences(),
        parameters.get_new_dist_strategy(),
        parameters.get_delta_start(),
        parameters.get_min_num_length(),
        parameters.get_num_conversion_strategy(),
        preprocess_pipeline,
        parameters.get_chunk_count(),
        parameters.get_context_len(),
        parameters.get_chunk_len()
    ]


def _apply_settings(optimization_steps, time_power, time_steepness, significant_level, min_sentences, new_dist_strat, delta_start, min_number_length, num_conversion, 
                    preprocess_pipeline, api_port, api_on, injection_strategy, add_chat_to_data, manual, postfix, data_separator, prefix, max_token_count, 
                    chunk_count, chunk_sep, context_len, chunk_regex, chunk_len, threads, strong_cleanup):
    logger.debug('Applying settings.')

    try:
        parameters.set_optimization_steps(optimization_steps)
        parameters.set_significant_level(significant_level)
        parameters.set_min_num_sentences(min_sentences)
        parameters.set_new_dist_strategy(new_dist_strat)
        parameters.set_delta_start(delta_start)
        parameters.set_min_num_length(min_number_length)
        parameters.set_num_conversion_strategy(num_conversion)
        parameters.set_api_port(api_port)
        parameters.set_api_on(api_on)
        parameters.set_injection_strategy(injection_strategy)
        parameters.set_add_chat_to_data(add_chat_to_data)
        parameters.set_manual(manual)
        parameters.set_postfix(codecs.decode(postfix, 'unicode_escape'))
        parameters.set_data_separator(codecs.decode(data_separator, 'unicode_escape'))
        parameters.set_prefix(codecs.decode(prefix, 'unicode_escape'))
        parameters.set_max_token_count(max_token_count)
        parameters.set_time_power(time_power)
        parameters.set_time_steepness(time_steepness)
        parameters.set_chunk_count(chunk_count)
        parameters.set_chunk_separator(codecs.decode(chunk_sep, 'unicode_escape'))
        parameters.set_context_len(context_len)
        parameters.set_chunk_regex(chunk_regex)
        parameters.set_chunk_len(chunk_len)
        parameters.set_num_threads(threads)
        parameters.set_strong_cleanup(strong_cleanup)

        preprocess_choices = ['Lower Cases', 'Remove Punctuation', 'Remove Adverbs', 'Remove Stop Words', 'Lemmatize', 'Merge Spaces', 'Strip Edges']
        for preprocess_method in preprocess_choices:
            if preprocess_method == 'Lower Cases':
                parameters.set_to_lower(preprocess_method in preprocess_pipeline)
            elif preprocess_method == 'Remove Punctuation':
                parameters.set_remove_punctuation(preprocess_method in preprocess_pipeline)
            elif preprocess_method == 'Remove Adverbs':
                parameters.set_remove_specific_pos(preprocess_method in preprocess_pipeline)
            elif preprocess_method == 'Remove Stop Words':
                parameters.set_remove_stopwords(preprocess_method in preprocess_pipeline)
            elif preprocess_method == 'Lemmatize':
                parameters.set_lemmatize(preprocess_method in preprocess_pipeline)
            elif preprocess_method == 'Merge Spaces':
                parameters.set_merge_spaces(preprocess_method in preprocess_pipeline)
            elif preprocess_method == 'Strip Edges':
                parameters.set_strip(preprocess_method in preprocess_pipeline)

        # Based on API on/off, start or stop the server
        if api_manager is not None:
            if parameters.get_api_on() and (not api_manager.is_server_running()):
                api_manager.start_server(parameters.get_api_port())
            elif (not parameters.get_api_on()) and api_manager.is_server_running():
                api_manager.stop_server()
    except Exception as e:
        logger.warn(f'Could not properly apply settings: {str(e)}')


def custom_generate_chat_prompt(user_input, state, **kwargs):
    return custom_generate_chat_prompt_internal(user_input, state, collector, **kwargs)


def input_modifier(string):
    return input_modifier_internal(string, collector)


def ui():
    with gr.Accordion("Click for more information...", open=False):
        gr.Markdown(textwrap.dedent("""

        ## About

        This extension takes a dataset as input, breaks it into chunks, and adds the result to a local/offline Chroma database.

        The database is then queried during inference time to get the excerpts that are closest to your input. The idea is to create an arbitrarily large pseudo context.

        The core methodology was developed and contributed by kaiokendev, who is working on improvements to the method in this repository: https://github.com/kaiokendev/superbig

        ## Data input

        Start by entering some data in the interface below and then clicking on "Load data".

        Each time you load some new data, the old chunks are discarded.

        ## Chat mode

        #### Instruct

        On each turn, the chunks will be compared to your current input and the most relevant matches will be appended to the input in the following format:

        ```
        Consider the excerpts below as additional context:
        ...
        ```

        The injection doesn't make it into the chat history. It is only used in the current generation.

        #### Regular chat

        The chunks from the external data sources are ignored, and the chroma database is built based on the chat history instead. The most relevant past exchanges relative to the present input are added to the context string. This way, the extension acts as a long term memory.

        ## Notebook/default modes

        Your question must be manually specified between `<|begin-user-input|>` and `<|end-user-input|>` tags, and the injection point must be specified with `<|injection-point|>`.

        The special tokens mentioned above (`<|begin-user-input|>`, `<|end-user-input|>`, and `<|injection-point|>`) are removed in the background before the text generation begins.

        Here is an example in Vicuna 1.1 format:

        ```
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

        USER:
        <|injection-point|>

        <|begin-user-input|>What datasets are mentioned in the text above?<|end-user-input|>
        ASSISTANT:
        ```
        """))

    with gr.Row():
        with gr.Column(min_width=600):
            with gr.Tab("Text input"):
                data_input = gr.Textbox(lines=20, label='Input data')
                update_data = gr.Button('Load data')

            with gr.Tab("URL input"):
                url_input = gr.Textbox(lines=10, label='Input URLs', info='Enter one or more URLs separated by newline characters.')
                strong_cleanup = gr.Checkbox(value=parameters.get_is_strong_cleanup(), label='Strong cleanup', info='Only keeps html elements that look like long-form text.')
                threads = gr.Number(value=parameters.get_num_threads(), label='Threads', info='The number of threads to use while downloading the URLs.', precision=0)
                update_url = gr.Button('Load data')

            with gr.Tab("File input"):
                file_input = gr.File(label='Input file', type='binary')
                update_file = gr.Button('Load data')
                
            with gr.Tab("Settings"):
                with gr.Accordion("Processing settings", open=True):
                    chunk_len = gr.Textbox(value=parameters.get_chunk_len(), label='Chunk length', info='In characters, not tokens. This value is used when you click on "Load data".')
                    chunk_regex = gr.Textbox(value=parameters.get_chunk_regex(), label='Chunk regex', info='Will specifically add the captured text to the embeddings.')
                    context_len = gr.Textbox(value=parameters.get_context_len(), label='Context length', info='In characters, not tokens. How much context to load around each chunk.')
                    chunk_sep = gr.Textbox(value=codecs.encode(parameters.get_chunk_separator(), 'unicode_escape').decode(), label='Chunk separator', info='Used to manually split chunks. Manually split chunks longer than chunk length are split again. This value is used when you click on "Load data".')

                with gr.Accordion("Generation settings", open=False):
                    chunk_count = gr.Number(value=parameters.get_chunk_count(), label='Chunk count', info='The number of closest-matching chunks to include in the prompt.')
                    max_token_count = gr.Number(value=parameters.get_max_token_count(), label='Max Context Tokens', info='The context length in tokens will not exceed this value.')
                    prefix = gr.Textbox(value=codecs.encode(parameters.get_prefix(), 'unicode_escape').decode(), label='Prefix', info='What to put before the injection point.')
                    data_separator = gr.Textbox(value=codecs.encode(parameters.get_data_separator(), 'unicode_escape').decode(), label='Data separator', info='When multiple pieces of distant data are added, they might be unrelated. It\'s important to separate them.')
                    postfix = gr.Textbox(value=codecs.encode(parameters.get_postfix(), 'unicode_escape').decode(), label='Postfix', info='What to put after the injection point.')
                    with gr.Row():
                        manual = gr.Checkbox(value=parameters.get_is_manual(), label="Is Manual", info="Manually specify when to use ChromaDB. Insert `!c` at the start or end of the message to trigger a query.", visible=shared.is_chat())
                        add_chat_to_data = gr.Checkbox(value=parameters.get_add_chat_to_data(), label="Add Chat to Data", info="Automatically feed the chat history as you chat.", visible=shared.is_chat())
                    injection_strategy = gr.Radio(choices=[parameters.PREPEND_TO_LAST, parameters.APPEND_TO_LAST, parameters.HIJACK_LAST_IN_CONTEXT], value=parameters.get_injection_strategy(), label='Injection Strategy', info='Where to inject the messages in chat or instruct mode.', visible=shared.is_chat())
                    with gr.Row():
                        api_on = gr.Checkbox(value=parameters.get_api_on(), label="Turn on API", info="Check this to turn on the API service.")
                        api_port = gr.Number(value=parameters.get_api_port(), label="API Port", info="The port on which the API service will run.")

                with gr.Accordion("Advanced settings", open=False):
                    preprocess_set_choices = []
                    if parameters.should_to_lower():
                        preprocess_set_choices.append('Lower Cases')
                    if parameters.should_remove_punctuation():
                        preprocess_set_choices.append('Remove Punctuation')
                    if parameters.should_remove_specific_pos():
                        preprocess_set_choices.append('Remove Adverbs')
                    if parameters.should_remove_stopwords():
                        preprocess_set_choices.append('Remove Stop Words')
                    if parameters.should_lemmatize():
                        preprocess_set_choices.append('Lemmatize')
                    if parameters.should_merge_spaces():
                        preprocess_set_choices.append('Merge Spaces')
                    if parameters.should_strip():
                        preprocess_set_choices.append('Strip Edges')

                    preprocess_pipeline = gr.CheckboxGroup(label='Preprocessing pipeline', choices=[
                        'Lower Cases',
                        'Remove Punctuation',
                        'Remove Adverbs',
                        'Remove Stop Words',
                        'Lemmatize',
                        'Merge Spaces',
                        'Strip Edges',
                    ], value=preprocess_set_choices, interactive=True, info='How to preprocess the text before it is turned into an embedding.')

                    with gr.Row():
                        num_conversion = gr.Dropdown(choices=[parameters.NUM_TO_WORD_METHOD, parameters.NUM_TO_CHAR_METHOD, parameters.NUM_TO_CHAR_LONG_METHOD, 'None'], value=parameters.get_num_conversion_strategy(), label="Number Conversion Method", info='How to preprocess numbers before creating the embeddings.', interactive=True)
                        min_number_length = gr.Number(value=parameters.get_min_num_length(), label='Number Length Threshold', info='In digits. Only numbers that have at least that many digits will be converted.', interactive=True)

                    delta_start = gr.Number(value=parameters.get_delta_start(), label='Delta Start Index', info='If the system encounters two identical embeddings, and they both start within the same delta, then only the first will be considered.', interactive=True)
                    new_dist_strat = gr.Dropdown(choices=[parameters.DIST_MIN_STRATEGY, parameters.DIST_HARMONIC_STRATEGY, parameters.DIST_GEOMETRIC_STRATEGY, parameters.DIST_ARITHMETIC_STRATEGY], value=parameters.get_new_dist_strategy(), label="Distance Strategy", info='When two embedding texts are merged, the distance of the new piece will be decided using one of these strategies.', interactive=True)
                    min_sentences = gr.Number(value=parameters.get_min_num_sentences(), label='Summary Threshold', info='In sentences. The minumum number of sentences to trigger text-rank summarization.', interactive=True)
                    significant_level = gr.Slider(0.8, 2, value=parameters.get_significant_level(), label='Significant Level', info='Defines the cut-off for what is considered a "significant" distance relative to the median distance among the returned samples.', interactive=True)
                    time_steepness = gr.Slider(0.01, 1.0, value=parameters.get_time_steepness(), label='Time Weighing Steepness', info='How differently two close excerpts are going to be weighed.')
                    time_power = gr.Slider(0.0, 1.0, value=parameters.get_time_power(), label='Time Weighing Power', info='How influencial is the weighing. At 1.0, old entries won\'t be considered')

            with gr.Tab("Benchmark"):
                benchmark_button = gr.Button('Benchmark')
                optimize_button = gr.Button('Optimize')
                optimization_steps = gr.Number(value=parameters.get_optimization_steps(), label='Optimization Steps', info='For how many steps to optimize.', interactive=True)


            clear_button = gr.Button('❌ Clear Data')

            
        with gr.Column():
            last_updated = gr.Markdown()

    all_params = [optimization_steps, time_power, time_steepness, significant_level, min_sentences, new_dist_strat, delta_start, min_number_length, num_conversion, 
                  preprocess_pipeline, api_port, api_on, injection_strategy, add_chat_to_data, manual, postfix, data_separator, prefix, max_token_count, 
                  chunk_count, chunk_sep, context_len, chunk_regex, chunk_len, threads, strong_cleanup]
    optimizable_params = [time_power, time_steepness, significant_level, min_sentences, new_dist_strat, delta_start, min_number_length, num_conversion, 
                  preprocess_pipeline, chunk_count, context_len, chunk_len]


    update_data.click(_feed_data_into_collector, [data_input], last_updated, show_progress=False)
    update_url.click(_feed_url_into_collector, [url_input], last_updated, show_progress=False)
    update_file.click(_feed_file_into_collector, [file_input], last_updated, show_progress=False)
    benchmark_button.click(_begin_benchmark, [], last_updated, show_progress=True)
    optimize_button.click(_begin_optimization, [], [last_updated] + optimizable_params, show_progress=True)
    clear_button.click(_clear_data, [], last_updated, show_progress=False)


    optimization_steps.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    time_power.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    time_steepness.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    significant_level.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    min_sentences.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    new_dist_strat.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    delta_start.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    min_number_length.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    num_conversion.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    preprocess_pipeline.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    api_port.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    api_on.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    injection_strategy.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    add_chat_to_data.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    manual.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    postfix.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    data_separator.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    prefix.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    max_token_count.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    chunk_count.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    chunk_sep.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    context_len.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    chunk_regex.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    chunk_len.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    threads.input(fn=_apply_settings, inputs=all_params, show_progress=False)
    strong_cleanup.input(fn=_apply_settings, inputs=all_params, show_progress=False)