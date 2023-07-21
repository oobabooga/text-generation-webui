import re
import textwrap
import bisect
import codecs

import gradio as gr
from bs4 import BeautifulSoup

from modules import chat, shared
from modules.logging_colors import logger

from .chromadb import add_chunks_to_collector, make_collector
from .download_urls import download_urls
from .text_preprocessor import TextPreprocessorBuilder, TextSummarizer

params = {
    'chunk_count': 200,
    'chunk_count_initial': 10,
    'time_weight': 0,
    'chunk_length': '40,50',
    'context_length': '400,800', # We provide less context before the embedding. This is because language typically flows forward, meaning the text that comes after an embedding often gives us more useful information.
    'chunk_separator': '',
    'data_separator': '\n\n<<document chunk>>\n\n',
    'strong_cleanup': False,
    'max_token_count': 3072,
    'chunk_regex': '(?<==== ).*?(?= ===)|User story: \d+',
    'threads': 4,
}

collector = make_collector()
chat_collector = make_collector()
summarizer = TextSummarizer()

def preprocess_text(text) -> list[str]:
    important_sentences = summarizer.process_long_text(text)
    return [TextPreprocessorBuilder(text).to_lower().remove_punctuation().merge_spaces().strip().num_to_word(1).build() for text in important_sentences]

def create_chunks_with_context(corpus, chunk_len, context_left, context_right):
    """
    This function takes a corpus of text and splits it into chunks of a specified length, 
    then adds a specified amount of context to each chunk. The context is added by first 
    going backwards from the start of the chunk and then going forwards from the end of the 
    chunk, ensuring that the context includes only whole words and that the total context length 
    does not exceed the specified limit. This function uses binary search for efficiency.

    Returns:
    chunks (list of str): The chunks of text.
    chunks_with_context (list of str): The chunks of text with added context.
    chunk_with_context_start_indices (list of int): The starting indices of each chunk with context in the corpus.
    """
    words = re.split('(\\s+)', corpus)
    word_start_indices = [0]
    current_index = 0

    for word in words:
        current_index += len(word)
        word_start_indices.append(current_index)

    chunks, chunk_lengths, chunk_start_indices, chunk_with_context_start_indices = [], [], [], []
    current_length = 0
    current_index = 0
    chunk = []

    for word in words:
        if current_length + len(word) > chunk_len:
            chunks.append(''.join(chunk))
            chunk_lengths.append(current_length)
            chunk_start_indices.append(current_index - current_length)
            chunk = [word]
            current_length = len(word)
        else:
            chunk.append(word)
            current_length += len(word)
        current_index += len(word)

    if chunk:
        chunks.append(''.join(chunk))
        chunk_lengths.append(current_length)
        chunk_start_indices.append(current_index - current_length)

    chunks_with_context = []
    for start_index, chunk_length in zip(chunk_start_indices, chunk_lengths):
        context_start_index = bisect.bisect_right(word_start_indices, start_index - context_left)
        context_end_index = bisect.bisect_left(word_start_indices, start_index + chunk_length + context_right)

        # Combine all the words in the context range (before, chunk, and after)
        chunk_with_context = ''.join(words[context_start_index:context_end_index])
        chunks_with_context.append(chunk_with_context)
        
        # Determine the start index of the chunk with context
        chunk_with_context_start_index = word_start_indices[context_start_index]
        chunk_with_context_start_indices.append(chunk_with_context_start_index)

    return chunks, chunks_with_context, chunk_with_context_start_indices


def feed_data_into_collector(corpus, chunk_len, context_len, chunk_regex, chunk_sep):
    global collector

    # Defining variables
    chunk_lens = [int(len.strip()) for len in chunk_len.split(',')]
    context_len = [int(len) for len in context_len.split(',')]
    if len(context_len) >= 3:
        raise f"Context len has too many values: {len(context_len)}"
    if len(context_len) == 2:
        context_left = context_len[0]
        context_right = context_len[1]
    else:
        context_left = context_right = context_len[0]
    chunk_sep = chunk_sep.replace(r'\n', '\n')
    cumulative = ''

    data_chunks = []
    data_chunks_with_context = []
    data_chunk_starting_indices = []

    # Handling chunk_regex
    if chunk_regex:
        if chunk_sep:
            cumulative_length = 0  # This variable will store the length of the processed corpus
            sections = corpus.split(chunk_sep)
            for section in sections:
                special_chunks = list(re.finditer(chunk_regex, section))
                for match in special_chunks:
                    chunk = match.group(0)
                    start_index = match.start()
                    end_index = start_index + len(chunk)
                    context = section[max(0, start_index - context_left):min(len(section), end_index + context_right)]
                    data_chunks.append(chunk)
                    data_chunks_with_context.append(context)
                    data_chunk_starting_indices.append(cumulative_length + max(0, start_index - context_left))
                cumulative_length += len(section) + len(chunk_sep)  # Update the length of the processed corpus
        else:
            special_chunks = list(re.finditer(chunk_regex, corpus))
            for match in special_chunks:
                chunk = match.group(0)
                start_index = match.start()
                end_index = start_index + len(chunk)
                context = corpus[max(0, start_index - context_left):min(len(corpus), end_index + context_right)]
                data_chunks.append(chunk)
                data_chunks_with_context.append(context)
                data_chunk_starting_indices.append(max(0, start_index - context_left))

        cumulative += f"{len(data_chunks)} special chunks have been found.\n\n"
        yield cumulative

    for chunk_len in chunk_lens:
        # Breaking the data into chunks and adding those to the db
        cumulative += "Breaking the input dataset...\n\n"
        yield cumulative
        if chunk_sep:
            cumulative_length = 0  # This variable will store the length of the processed corpus
            sections = corpus.split(chunk_sep)
            for section in sections:
                chunks, chunks_with_context, context_start_indices = create_chunks_with_context(section, chunk_len, context_left, context_right)
                context_start_indices = [cumulative_length + i for i in context_start_indices]  # Add the length of the processed corpus to each start index
                data_chunks.extend(chunks)
                data_chunks_with_context.extend(chunks_with_context)
                data_chunk_starting_indices.extend(context_start_indices)
                cumulative_length += len(section) + len(chunk_sep)  # Update the length of the processed corpus
        else:
            chunks, chunks_with_context, context_start_indices = create_chunks_with_context(corpus, chunk_len, context_left, context_right)
            data_chunks.extend(chunks)
            data_chunks_with_context.extend(chunks_with_context)
            data_chunk_starting_indices.extend(context_start_indices)

        cumulative += f"{len(data_chunks)} chunks have been found.\n\n"
        yield cumulative

    cumulative += f"Preprocessing chunks...\n\n"
    yield cumulative
    data_chunks = [preprocess_text(chunk) for chunk in data_chunks]

    cumulative += f"Adding all {len(data_chunks)} chunks to the database. This may take a while.\n\n"
    yield cumulative
    add_chunks_to_collector(data_chunks, data_chunks_with_context, data_chunk_starting_indices, collector)
    cumulative += "Done."
    yield cumulative


def feed_file_into_collector(file, chunk_len, context_len, chunk_regex, chunk_sep):
    yield 'Reading the input dataset...\n\n'
    text = file.decode('utf-8')
    for i in feed_data_into_collector(text, chunk_len, context_len, chunk_regex, chunk_sep):
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

    for i in feed_data_into_collector(all_text, chunk_len, context_len, chunk_regex, chunk_sep):
        yield i


def apply_settings(chunk_count, chunk_count_initial, time_weight, max_token_count, data_separator):
    global params
    params['chunk_count'] = int(chunk_count)
    params['chunk_count_initial'] = int(chunk_count_initial)
    params['time_weight'] = time_weight
    params['max_token_count'] = max_token_count
    params['data_separator'] = codecs.decode(data_separator, 'unicode_escape')
    settings_to_display = {k: params[k] for k in params if k in ['chunk_count', 'chunk_count_initial', 'time_weight', 'max_token_count']}
    yield f"The following settings are now active: {str(settings_to_display)}"


def custom_generate_chat_prompt(user_input, state, **kwargs):
    global chat_collector

    history = state['history']

    if state['mode'] == 'instruct':
        results = collector.get_sorted(user_input, n_results=params['chunk_count'])
        additional_context = '\nYour reply should be based on the context below:\n\n' + '\n'.join(results)
        user_input += additional_context
    else:

        def make_single_exchange(id_):
            output = ''
            output += f"{state['name1']}: {history['internal'][id_][0]}\n"
            output += f"{state['name2']}: {history['internal'][id_][1]}\n"
            return output

        if len(history['internal']) > params['chunk_count'] and user_input != '':
            chunks = []
            hist_size = len(history['internal'])
            for i in range(hist_size - 1):
                chunks.append(make_single_exchange(i))

            add_chunks_to_collector(chunks, chat_collector)
            query = '\n'.join(history['internal'][-1] + [user_input])
            try:
                best_ids = chat_collector.get_ids_sorted(query, n_results=params['chunk_count'], n_initial=params['chunk_count_initial'], time_weight=params['time_weight'])
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

        print(f"Preprocessed user input: {user_input}")

        # Get the most similar chunks
        results = collector.get_sorted_by_dist(user_input, n_results=params['chunk_count'], max_token_count=int(params['max_token_count']))

        # Make the injection
        string = string.replace('<|injection-point|>', params['data_separator'].join(results))

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
                strong_cleanup = gr.Checkbox(value=params['strong_cleanup'], label='Strong cleanup', info='Only keeps html elements that look like long-form text.')
                threads = gr.Number(value=params['threads'], label='Threads', info='The number of threads to use while downloading the URLs.', precision=0)
                update_url = gr.Button('Load data')

            with gr.Tab("File input"):
                file_input = gr.File(label='Input file', type='binary')
                update_file = gr.Button('Load data')

            with gr.Tab("Generation settings"):
                chunk_count = gr.Number(value=params['chunk_count'], label='Chunk count', info='The number of closest-matching chunks to include in the prompt.')
                gr.Markdown('Time weighting (optional, used in to make recently added chunks more likely to appear)')
                time_weight = gr.Slider(0, 1, value=params['time_weight'], label='Time weight', info='Defines the strength of the time weighting. 0 = no time weighting.')
                chunk_count_initial = gr.Number(value=params['chunk_count_initial'], label='Initial chunk count', info='The number of closest-matching chunks retrieved for time weight reordering in chat mode. This should be >= chunk count. -1 = All chunks are retrieved. Only used if time_weight > 0.')
                max_token_count = gr.Number(value=params['max_token_count'], label='Max Context Tokens', info='The context will not exceed this value unless the best match is longer. In this case, only it will be added.')
                data_separator = gr.Textbox(value=codecs.encode(params['data_separator'], 'unicode_escape').decode(), label='Data separator', info='When multiple pieces of distant data are added, they might be unrelated. It\'s important to separate them.')

                update_settings = gr.Button('Apply changes')

            chunk_len = gr.Textbox(value=params['chunk_length'], label='Chunk length', info='In characters, not tokens. This value is used when you click on "Load data".')
            chunk_regex = gr.Textbox(value=params['chunk_regex'], label='Chunk regex', info='Will specifically add the captured text to the embeddings.')
            context_len = gr.Textbox(value=params['context_length'], label='Context length', info='In characters, not tokens. How much context to load around each chunk.')
            chunk_sep = gr.Textbox(value=params['chunk_separator'], label='Chunk separator', info='Used to manually split chunks. Manually split chunks longer than chunk length are split again. This value is used when you click on "Load data".')
        with gr.Column():
            last_updated = gr.Markdown()

    update_data.click(feed_data_into_collector, [data_input, chunk_len, context_len, chunk_regex, chunk_sep], last_updated, show_progress=False)
    update_url.click(feed_url_into_collector, [url_input, chunk_len, context_len, chunk_regex, chunk_sep, strong_cleanup, threads], last_updated, show_progress=False)
    update_file.click(feed_file_into_collector, [file_input, chunk_len, context_len, chunk_regex, chunk_sep], last_updated, show_progress=False)
    update_settings.click(apply_settings, [chunk_count, chunk_count_initial, time_weight, max_token_count, data_separator], last_updated, show_progress=False)