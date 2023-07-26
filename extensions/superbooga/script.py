"""
This file is responsible for the UI and how the application interracts with the rest of the system.
"""

import re
import textwrap
import codecs
import gradio as gr

import extensions.superbooga.parameters as parameters

from bs4 import BeautifulSoup
from pathlib import Path

from modules import chat, shared
from modules.logging_colors import logger

from .chromadb import add_chunks_to_collector, make_collector
from .download_urls import download_urls
from .data_processor import process_and_add_to_collector, preprocess_text
from .benchmark import benchmark
from .optimize import optimize

collector = make_collector()
chat_collector = make_collector()

def feed_data_into_collector(corpus, chunk_len, context_len, chunk_regex, chunk_sep):
    for i in process_and_add_to_collector(corpus, chunk_len, context_len, chunk_regex, chunk_sep, collector):
        yield i


def feed_file_into_collector(file, chunk_len, context_len, chunk_regex, chunk_sep):
    yield 'Reading the input dataset...\n\n'
    text = file.decode('utf-8')
    for i in process_and_add_to_collector(text, chunk_len, context_len, chunk_regex, chunk_sep, collector):
        yield i


def feed_url_into_collector(urls, chunk_len, context_len, chunk_regex, chunk_sep, strong_cleanup, threads):
    all_text = ''
    cumulative = ''

    urls = urls.strip().split('\n')
    cumulative += f'Loading {len(urls)} URLs with {threads} threads...\n\n'
    yield cumulative
    for update, contents in download_urls(urls, threads=threads):
        yield cumulative + update

    cumulative += 'Processing the HTML sources...'
    yield cumulative
    for content in contents:
        soup = BeautifulSoup(content, features="lxml")
        for script in soup(["script", "style"]):
            script.extract()

        strings = soup.stripped_strings
        if strong_cleanup:
            strings = [s for s in strings if re.search("[A-Za-z] ", s)]

        text = '\n'.join([s.strip() for s in strings])
        all_text += text

    for i in process_and_add_to_collector(all_text, chunk_len, context_len, chunk_regex, chunk_sep, collector):
        yield i


def begin_benchmark(chunk_count, max_token_count, chunk_len, context_len, chunk_regex, chunk_sep):
    score = benchmark(Path("extensions/superbooga/benchmark_texts/questions.json"), chunk_count,
                      max_token_count, chunk_len, context_len, chunk_regex, chunk_sep, collector)
    return f'**Score**: {score}'


def begin_optimization(optimization_steps, progress=gr.Progress()):
    return optimize(collector, int(optimization_steps), progress), *get_optimizable_settings()


def get_optimizable_settings() -> list:
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
        parameters.get_confidence_interval(),
        parameters.get_min_num_sentences(),
        parameters.get_new_dist_strategy(),
        parameters.get_delta_start(),
        parameters.get_min_num_length(),
        parameters.get_num_conversion_strategy(),
        preprocess_pipeline,
        parameters.get_time_weight(),
        parameters.get_chunk_count(),
        parameters.get_context_len(),
        parameters.get_chunk_len()
    ]


def apply_settings(optimization_steps, confidence_interval, min_sentences, new_dist_strat, delta_start, min_number_length, num_conversion, 
                   preprocess_pipeline, data_separator, max_token_count, time_weight, chunk_count, chunk_sep, context_len, chunk_regex, chunk_len):
    logger.debug('Applying settings.')

    parameters.set_optimization_steps(optimization_steps)
    parameters.set_confidence_interval(confidence_interval)
    parameters.set_min_num_sentences(min_sentences)
    parameters.set_new_dist_strategy(new_dist_strat)
    parameters.set_delta_start(delta_start)
    parameters.set_min_num_length(min_number_length)
    parameters.set_num_conversion_strategy(num_conversion)
    parameters.set_data_separator(data_separator)
    parameters.set_max_token_count(max_token_count)
    parameters.set_time_weight(time_weight)
    parameters.set_chunk_count(chunk_count)
    parameters.set_chunk_separator(chunk_sep)
    parameters.set_context_len(context_len)
    parameters.set_chunk_regex(chunk_regex)
    parameters.set_chunk_len(chunk_len)

    for preprocess_method in preprocess_pipeline:
        if preprocess_method == 'Lower Cases':
            parameters.set_to_lower(True)
        elif preprocess_method == 'Remove Punctuation':
            parameters.set_remove_punctuation(True)
        elif preprocess_method == 'Remove Adverbs':
            parameters.set_remove_specific_pos(True)
        elif preprocess_method == 'Remove Stop Words':
            parameters.set_remove_stopwords(True)
        elif preprocess_method == 'Lemmatize':
            parameters.set_lemmatize(True)
        elif preprocess_method == 'Merge Spaces':
            parameters.set_merge_spaces(True)
        elif preprocess_method == 'Strip Edges':
            parameters.set_strip(True)


def custom_generate_chat_prompt(user_input, state, **kwargs):
    global chat_collector

    history = state['history']

    if state['mode'] == 'instruct':
        results = collector.get_sorted(user_input, n_results=parameters.get_chunk_count())
        additional_context = '\nYour reply should be based on the context below:\n\n' + '\n'.join(results)
        user_input += additional_context
    else:

        def make_single_exchange(id_):
            output = ''
            output += f"{state['name1']}: {history['internal'][id_][0]}\n"
            output += f"{state['name2']}: {history['internal'][id_][1]}\n"
            return output

        if len(history['internal']) > parameters.get_chunk_count() and user_input != '':
            chunks = []
            hist_size = len(history['internal'])
            for i in range(hist_size - 1):
                chunks.append(make_single_exchange(i))

            add_chunks_to_collector(chunks, chat_collector)
            query = '\n'.join(history['internal'][-1] + [user_input])
            try:
                best_ids = chat_collector.get_ids_sorted(query, n_results=parameters.get_chunk_count(), n_initial=parameters.get_chunk_count(), time_weight=parameters.get_time_weight())
                additional_context = '\n'
                for id_ in best_ids:
                    if history['internal'][id_][0] != '<|BEGIN-VISIBLE-CHAT|>':
                        additional_context += make_single_exchange(id_)

                logger.warning(f'Adding the following new context:\n{additional_context}')
                state['context'] = state['context'].strip() + '\n' + additional_context
                kwargs['history'] = {
                    'internal': [history['internal'][i] for i in range(hist_size) if i not in best_ids],
                    'visible': ''
                }
            except RuntimeError:
                logger.error("Couldn't query the database, moving on...")

    return chat.generate_chat_prompt(user_input, state, **kwargs)


def remove_special_tokens(string):
    pattern = r'(<\|begin-user-input\|>|<\|end-user-input\|>|<\|injection-point\|>)'
    return re.sub(pattern, '', string)


def input_modifier(string):
    if shared.is_chat():
        return string

    # Find the user input
    pattern = re.compile(r"<\|begin-user-input\|>(.*?)<\|end-user-input\|>", re.DOTALL)
    match = re.search(pattern, string)
    if match:
        user_input = match.group(1).strip()
        user_input = preprocess_text(user_input)

        logger.debug(f"Preprocessed User Input: {user_input}")

        # Get the most similar chunks
        results = collector.get_sorted_by_dist(user_input, n_results=parameters.get_chunk_count(), max_token_count=int(parameters.get_max_token_count()))

        # Make the injection
        string = string.replace('<|injection-point|>', parameters.get_data_separator().join(results))

    return remove_special_tokens(string)


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

        ⚠️  For best results, make sure to remove the spaces and new line characters after `ASSISTANT:`.

        *This extension is currently experimental and under development.*

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
                    chunk_sep = gr.Textbox(value=parameters.get_chunk_separator(), label='Chunk separator', info='Used to manually split chunks. Manually split chunks longer than chunk length are split again. This value is used when you click on "Load data".')

                with gr.Accordion("Generation settings", open=False):
                    chunk_count = gr.Number(value=parameters.get_chunk_count(), label='Chunk count', info='The number of closest-matching chunks to include in the prompt.')
                    time_weight = gr.Slider(0, 1, value=parameters.get_time_weight(), label='Time weight', info='Defines the strength of the time weighting. Zero means the feature is turned off.')
                    max_token_count = gr.Number(value=parameters.get_max_token_count(), label='Max Context Tokens', info='The context length in tokens will not exceed this value.')
                    data_separator = gr.Textbox(value=codecs.encode(parameters.get_data_separator(), 'unicode_escape').decode(), label='Data separator', info='When multiple pieces of distant data are added, they might be unrelated. It\'s important to separate them.')

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
                    confidence_interval = gr.Slider(0, 1, value=parameters.get_confidence_interval(), label='Confidence Interval', info='Consider only the smallest interval containing a certain percentage of entries, where entries are weighted according to their inverse distance (1/d).', interactive=True)

            with gr.Tab("Benchmark"):
                benchmark_button = gr.Button('Benchmark')
                optimize_button = gr.Button('Optimize')
                optimization_steps = gr.Number(value=parameters.get_optimization_steps(), label='Optimization Steps', info='For how many steps to optimize.', interactive=True)

            
        with gr.Column():
            last_updated = gr.Markdown()

    all_params = [optimization_steps, confidence_interval, min_sentences, new_dist_strat, delta_start, min_number_length, num_conversion, 
                  preprocess_pipeline, data_separator, max_token_count, time_weight, chunk_count, chunk_sep, context_len, chunk_regex, chunk_len]
    optimizable_params = [confidence_interval, min_sentences, new_dist_strat, delta_start, min_number_length, num_conversion, 
                  preprocess_pipeline, time_weight, chunk_count, context_len, chunk_len]


    update_data.click(feed_data_into_collector, [data_input, chunk_len, context_len, chunk_regex, chunk_sep], last_updated, show_progress=False)
    update_url.click(feed_url_into_collector, [url_input, chunk_len, context_len, chunk_regex, chunk_sep, strong_cleanup, threads], last_updated, show_progress=False)
    update_file.click(feed_file_into_collector, [file_input, chunk_len, context_len, chunk_regex, chunk_sep], last_updated, show_progress=False)
    benchmark_button.click(begin_benchmark, [chunk_count, max_token_count, chunk_len, context_len, chunk_regex, chunk_sep], last_updated, show_progress=True)
    optimize_button.click(begin_optimization, [optimization_steps], [last_updated] + optimizable_params, show_progress=True)


    optimization_steps.input(fn=apply_settings, inputs=all_params, show_progress=False)
    confidence_interval.input(fn=apply_settings, inputs=all_params, show_progress=False)
    min_sentences.input(fn=apply_settings, inputs=all_params, show_progress=False)
    new_dist_strat.input(fn=apply_settings, inputs=all_params, show_progress=False)
    delta_start.input(fn=apply_settings, inputs=all_params, show_progress=False)
    min_number_length.input(fn=apply_settings, inputs=all_params, show_progress=False)
    num_conversion.input(fn=apply_settings, inputs=all_params, show_progress=False)
    preprocess_pipeline.input(fn=apply_settings, inputs=all_params, show_progress=False)
    data_separator.input(fn=apply_settings, inputs=all_params, show_progress=False)
    max_token_count.input(fn=apply_settings, inputs=all_params, show_progress=False)
    time_weight.input(fn=apply_settings, inputs=all_params, show_progress=False)
    chunk_count.input(fn=apply_settings, inputs=all_params, show_progress=False)
    chunk_sep.input(fn=apply_settings, inputs=all_params, show_progress=False)
    context_len.input(fn=apply_settings, inputs=all_params, show_progress=False)
    chunk_regex.input(fn=apply_settings, inputs=all_params, show_progress=False)
    chunk_len.input(fn=apply_settings, inputs=all_params, show_progress=False)