import logging
import re
import textwrap

import chromadb
import gradio as gr
import posthog
import torch
from bs4 import BeautifulSoup
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from modules import chat, shared

from .download_urls import download_urls

logging.info('Intercepting all calls to posthog :)')
posthog.capture = lambda *args, **kwargs: None

# These parameters are customizable through settings.json
params = {
    'chunk_count': 5,
    'chunk_length': 700,
    'strong_cleanup': True,
    'threads': 4,
}


class Collecter():
    def __init__(self):
        pass

    def add(self, texts: list[str]):
        pass

    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        pass

    def clear(self):
        pass


class Embedder():
    def __init__(self):
        pass

    def embed(self, text: str) -> list[torch.Tensor]:
        pass


class ChromaCollector(Collecter):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.embedder = embedder
        self.collection = self.chroma_client.create_collection(name="context", embedding_function=embedder.embed)
        self.ids = []

    def add(self, texts: list[str]):
        self.ids = [f"id{i}" for i in range(len(texts))]
        self.collection.add(documents=texts, ids=self.ids)

    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        n_results = min(len(self.ids), n_results)
        result = self.collection.query(query_texts=search_strings, n_results=n_results, include=['documents'])['documents'][0]
        return result

    def get_ids(self, search_strings: list[str], n_results: int) -> list[str]:
        n_results = min(len(self.ids), n_results)
        result = self.collection.query(query_texts=search_strings, n_results=n_results, include=['documents'])['ids'][0]
        return list(map(lambda x: int(x[2:]), result))

    def clear(self):
        self.collection.delete(ids=self.ids)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.embed = self.model.encode


embedder = SentenceTransformerEmbedder()
collector = ChromaCollector(embedder)
chunk_count = 5


def add_chunks_to_collector(chunks):
    global collector
    collector.clear()
    collector.add(chunks)


def feed_data_into_collector(corpus, chunk_len):
    # Defining variables
    chunk_len = int(chunk_len)
    cumulative = ''

    # Breaking the data into chunks and adding those to the db
    cumulative += "Breaking the input dataset...\n\n"
    yield cumulative
    data_chunks = [corpus[i:i + chunk_len] for i in range(0, len(corpus), chunk_len)]
    cumulative += f"{len(data_chunks)} chunks have been found.\n\nAdding the chunks to the database...\n\n"
    yield cumulative
    add_chunks_to_collector(data_chunks)
    cumulative += "Done."
    yield cumulative


def feed_file_into_collector(file, chunk_len):
    yield 'Reading the input dataset...\n\n'
    text = file.decode('utf-8')
    for i in feed_data_into_collector(text, chunk_len):
        yield i


def feed_url_into_collector(urls, chunk_len, strong_cleanup, threads):
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
        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()

        strings = soup.stripped_strings
        if strong_cleanup:
            strings = [s for s in strings if re.search("[A-Za-z] ", s)]

        text = '\n'.join([s.strip() for s in strings])
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n\n'.join(chunk for chunk in chunks if chunk)
        all_text += text

    for i in feed_data_into_collector(all_text, chunk_len):
        yield i


def apply_settings(_chunk_count):
    global chunk_count
    chunk_count = int(_chunk_count)
    settings_to_display = {
        'chunk_count': chunk_count,
    }
    yield f"The following settings are now active: {str(settings_to_display)}"


def input_modifier(string):
    if shared.is_chat():
        return string

    # Find the user input
    pattern = re.compile(r"<\|begin-user-input\|>(.*?)<\|end-user-input\|>", re.DOTALL)
    match = re.search(pattern, string)
    if match:
        user_input = match.group(1).strip()
    else:
        user_input = ''

    # Get the most similar chunks
    results = collector.get(user_input, n_results=chunk_count)

    # Make the replacements
    string = string.replace('<|begin-user-input|>', '')
    string = string.replace('<|end-user-input|>', '')
    string = string.replace('<|injection-point|>', '\n'.join(results))

    return string


def custom_generate_chat_prompt(user_input, state, **kwargs):
    if len(shared.history['internal']) > 2 and user_input != '':
        chunks = []
        for i in range(len(shared.history['internal']) - 1):
            chunks.append('\n'.join(shared.history['internal'][i]))

        add_chunks_to_collector(chunks)
        query = '\n'.join(shared.history['internal'][-1] + [user_input])
        try:
            best_ids = collector.get_ids(query, n_results=len(shared.history['internal']) - 1)

            # Sort the history by relevance instead of by chronological order,
            # except for the latest message
            state['history'] = [shared.history['internal'][id_] for id_ in best_ids[::-1]] + [shared.history['internal'][-1]]
        except RuntimeError:
            logging.error("Couldn't query the database, moving on...")

    return chat.generate_chat_prompt(user_input, state, **kwargs)


def ui():
    with gr.Accordion("Click for more information...", open=False):
        gr.Markdown(textwrap.dedent("""

        ## About

        This extension takes a dataset as input, breaks it into chunks, and adds the result to a local/offline Chroma database.

        The database is then queried during inference time to get the excerpts that are closest to your input. The idea is to create
        an arbitrarily large pseudocontext.

        It is a modified version of the superbig extension by kaiokendev: https://github.com/kaiokendev/superbig

        ## Notebook/default modes

        ### How to use it

        1) Paste your input text (of whatever length) into the text box below.
        2) Click on "Load data" to feed this text into the Chroma database.
        3) In your prompt, enter your question between `<|begin-user-input|>` and `<|end-user-input|>`, and specify the injection point with `<|injection-point|>`.

        By default, the 5 closest chunks will be injected. You can customize this value in the "Generation settings" tab.

        The special tokens mentioned above (`<|begin-user-input|>`, `<|end-user-input|>`, and `<|injection-point|>`) are removed when the injection happens.

        ### Example

        For your convenience, you can use the following prompt as a starting point (for Vicuna 1.1 models):

        ```
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

        USER:

        <|begin-user-input|>
        What datasets are mentioned in the text below?
        <|end-user-input|>

        <|injection-point|>

        ASSISTANT:
        ```

        ⚠️  For best results, make sure to remove the spaces and new line characters after `ASSISTANT:`.

        ## Chat mode

        In chat mode, the extension automatically sorts the history by relevance instead of chronologically, except for the very latest input/reply pair.

        That is, the prompt will include (starting from the end):

        * Your input
        * The latest input/reply pair
        * The #1 most relevant input/reply pair prior to the latest
        * The #2 most relevant input/reply pair prior to the latest
        * Etc

        This way, the bot can have a long term history.

        *This extension is currently experimental and under development.*

        """))

    if not shared.is_chat():
        with gr.Row():
            with gr.Column():
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
                    update_settings = gr.Button('Apply changes')

                chunk_len = gr.Number(value=params['chunk_length'], label='Chunk length', info='In characters, not tokens. This value is used when you click on "Load data".')

            with gr.Column():
                last_updated = gr.Markdown()

        update_data.click(feed_data_into_collector, [data_input, chunk_len], last_updated, show_progress=False)
        update_url.click(feed_url_into_collector, [url_input, chunk_len, strong_cleanup, threads], last_updated, show_progress=False)
        update_file.click(feed_file_into_collector, [file_input, chunk_len], last_updated, show_progress=False)
        update_settings.click(apply_settings, [chunk_count], last_updated, show_progress=False)
