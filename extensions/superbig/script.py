import datetime
import re
import textwrap

import chromadb
import gradio as gr
import posthog
import torch
from chromadb.config import Settings
from modules import shared
from sentence_transformers import SentenceTransformer

print('Intercepting all calls to posthog :)')
posthog.capture = lambda *args, **kwargs: None


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
        result = self.collection.query(query_texts=search_strings, n_results=n_results, include=['documents'])['documents'][0]
        return result

    def clear(self):
        self.collection.delete(ids=self.ids)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.embed = self.model.encode


embedder = SentenceTransformerEmbedder()
collector = ChromaCollector(embedder)
global_data = ''


def feed_data_into_collector(corpus):
    global collector
    chunk_len = 700
    data_chunks = [corpus[i:i + chunk_len] for i in range(0, len(corpus), chunk_len)]
    collector.clear()
    collector.add(data_chunks)


def input_modifier(string):

    # Find the user input
    pattern = re.compile(r"<\|begin-user-input\|>(.*?)<\|end-user-input\|>")
    match = re.search(pattern, string)
    if match:
        user_input = match.group(1)
    else:
        user_input = ''

    # Get the 5 most similar chunks
    feed_data_into_collector(global_data)
    results = collector.get(user_input, n_results=5)

    # Make the replacements
    string = string.replace('<|begin-user-input|>', '')
    string = string.replace('<|end-user-input|>', '')
    string = string.replace('<|injection-point|>', '\n'.join(results))

    return string


def ui():
    gr.Markdown(textwrap.dedent("""

    *This extension is currently experimental and under development.*

    ## How to use it

    1) Paste your input text (of whatever length) into the text box below.
    2) Click on the "Apply" button located below the text box
    3) In your prompt, enter your question between <|begin-user-input|> and <|end-user-input|>, and specify the injection point with <|injection-point|>

    ## How it works

    In the background, the 5 chunks in the input text most similar to the user input will be placed at the injection point, and the special tokens above will be removed. Then the text generation will proceed as usual.

    ## Example

    For your convenience, you can use the following prompt as a starting point (for Alpaca models):

    ```
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    You are ArxivGPT, trained on millions of Arxiv papers. You always answer the question, even if full context isn't provided to you. The following are snippets from an Arxiv paper. Use the snippets to answer the question. Think about it step by step

    <|injection-point|>

    ### Input:
    <|begin-user-input|>What datasets are mentioned in the paper above?<|end-user-input|>

    ### Response:
    ```

    """))
    if shared.is_chat():
        # Chat mode has to be handled differently, probably using a custom_generate_chat_prompt
        pass
    else:
        data_input = gr.Textbox(lines=20, label='Input data', info='Paste your input data below and then click on Apply before generating.')
        with gr.Row():
            update = gr.Button('Apply')
            last_updated = gr.Markdown()

        update.click(
            lambda x: globals().update(global_data=x), data_input, None).then(
            lambda: "Last updated on " + str(datetime.datetime.now()), None, last_updated, show_progress=False)
