import re
import textwrap

import gradio as gr
from bs4 import BeautifulSoup

from modules import chat, shared
from modules.logging_colors import logger

from .chromadb import add_chunks_to_collector, make_collector
from .download_urls import download_urls

params = {
    "chunk_count": 5,
    "chunk_count_initial": 10,
    "time_weight": 0,
    "chunk_length": 700,
    "chunk_separator": "",
    "strong_cleanup": False,
    "threads": 4,
}

collector = make_collector()
chat_collector = make_collector()


def feed_data_into_collector(corpus, chunk_len, chunk_sep):
    global collector

    # Defining variables
    chunk_len = int(chunk_len)
    chunk_sep = chunk_sep.replace(r"\n", "\n")
    cumulative = ""

    # Breaking the data into chunks and adding those to the db
    cumulative += "Breaking the input dataset...\n\n"
    yield cumulative
    if chunk_sep:
        data_chunks = corpus.split(chunk_sep)
        data_chunks = [
            [
                data_chunk[i : i + chunk_len]
                for i in range(0, len(data_chunk), chunk_len)
            ]
            for data_chunk in data_chunks
        ]
        data_chunks = [x for y in data_chunks for x in y]
    else:
        data_chunks = [
            corpus[i : i + chunk_len] for i in range(0, len(corpus), chunk_len)
        ]

    cumulative += f"{len(data_chunks)} chunks have been found.\n\nAdding the chunks to the database...\n\n"
    yield cumulative
    add_chunks_to_collector(data_chunks, collector)
    cumulative += "Done."
    yield cumulative


def feed_file_into_collector(file, chunk_len, chunk_sep):
    yield "Reading the input dataset...\n\n"
    text = file.decode("utf-8")
    for i in feed_data_into_collector(text, chunk_len, chunk_sep):
        yield i


def feed_url_into_collector(urls, chunk_len, chunk_sep, strong_cleanup, threads):
    all_text = ""
    cumulative = ""

    urls = urls.strip().split("\n")
    cumulative += f"Loading {len(urls)} URLs with {threads} threads...\n\n"
    yield cumulative
    for update, contents in download_urls(urls, threads=threads):
        yield cumulative + update

    cumulative += "Processing the HTML sources..."
    yield cumulative
    for content in contents:
        soup = BeautifulSoup(content, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()

        strings = soup.stripped_strings
        if strong_cleanup:
            strings = [s for s in strings if re.search("[A-Za-z] ", s)]

        text = "\n".join([s.strip() for s in strings])
        all_text += text

    for i in feed_data_into_collector(all_text, chunk_len, chunk_sep):
        yield i


def apply_settings(chunk_count, chunk_count_initial, time_weight):
    global params
    params["chunk_count"] = int(chunk_count)
    params["chunk_count_initial"] = int(chunk_count_initial)
    params["time_weight"] = time_weight
    settings_to_display = {
        k: params[k]
        for k in params
        if k in ["chunk_count", "chunk_count_initial", "time_weight"]
    }
    yield f"The following settings are now active: {str(settings_to_display)}"


def custom_generate_chat_prompt(user_input, state, **kwargs):
    global chat_collector

    if state["mode"] == "instruct":
        results = collector.get_sorted(user_input, n_results=params["chunk_count"])
        additional_context = (
            "\nYour reply should be based on the context below:\n\n"
            + "\n".join(results)
        )
        user_input += additional_context
    else:

        def make_single_exchange(id_):
            output = ""
            output += f"{state['name1']}: {shared.history['internal'][id_][0]}\n"
            output += f"{state['name2']}: {shared.history['internal'][id_][1]}\n"
            return output

        if len(shared.history["internal"]) > params["chunk_count"] and user_input != "":
            chunks = []
            hist_size = len(shared.history["internal"])
            for i in range(hist_size - 1):
                chunks.append(make_single_exchange(i))

            add_chunks_to_collector(chunks, chat_collector)
            query = "\n".join(shared.history["internal"][-1] + [user_input])
            try:
                best_ids = chat_collector.get_ids_sorted(
                    query,
                    n_results=params["chunk_count"],
                    n_initial=params["chunk_count_initial"],
                    time_weight=params["time_weight"],
                )
                additional_context = "\n"
                for id_ in best_ids:
                    if shared.history["internal"][id_][0] != "<|BEGIN-VISIBLE-CHAT|>":
                        additional_context += make_single_exchange(id_)

                logger.warning(
                    f"Adding the following new context:\n{additional_context}"
                )
                state["context"] = state["context"].strip() + "\n" + additional_context
                kwargs["history"] = {
                    "internal": [
                        shared.history["internal"][i]
                        for i in range(hist_size)
                        if i not in best_ids
                    ],
                    "visible": "",
                }
            except RuntimeError:
                logger.error("Couldn't query the database, moving on...")

    return chat.generate_chat_prompt(user_input, state, **kwargs)


def remove_special_tokens(string):
    pattern = r"(<\|begin-user-input\|>|<\|end-user-input\|>|<\|injection-point\|>)"
    return re.sub(pattern, "", string)


def input_modifier(string):
    if shared.is_chat():
        return string

    # Find the user input
    pattern = re.compile(r"<\|begin-user-input\|>(.*?)<\|end-user-input\|>", re.DOTALL)
    match = re.search(pattern, string)
    if match:
        user_input = match.group(1).strip()

        # Get the most similar chunks
        results = collector.get_sorted(user_input, n_results=params["chunk_count"])

        # Make the injection
        string = string.replace("<|injection-point|>", "\n".join(results))

    return remove_special_tokens(string)


def ui():
    with gr.Accordion("Click for more information...", open=False):
        gr.Markdown(
            textwrap.dedent(
                """

        ## 关于

        该扩展将数据集作为输入，将其分割成块，并将结果添加到本地/离线 Chroma 数据库中。

        然后在推理过程中查询该数据库，以获得与输入内容最接近的摘录。我们的想法是创建一个任意大的伪上下文。

        该核心方法由kaiokendev开发和贡献，他正在该资源库中对该方法进行改进。: https://github.com/kaiokendev/superbig

        ## 数据输入
        
        首先在下面的界面中输入一些数据，然后点击 "加载数据"。

        每次加载新数据时，旧数据块将被丢弃。
        
        ## 聊天模式

        ### Instruct Mode

        在每一轮中，数据块将与您当前的输入进行比较，最匹配的数据块将以以下格式添加到输入中：

        ```
        请将以下摘录作为额外的上下文：
        ...
        ```

        注入不会进入聊天历史。它只用于当前生成。

        ### Chat Mode

        来自外部数据源的数据块将被忽略，chroma数据库将根据聊天历史建立。相对于当前输入，最相关的过去交流被添加到上下文字符串中。通过这种方式，扩展功能可作为长期记忆。

        ## 笔记本/默认模式

        您的问题必须在"<|begin-user-input|>"和"<|end-user-input|>"标签之间手动指定，注入点必须用"<|injection-point|>"指定。

        上面提到的特殊标记（`<|begin-user-input|>`, `<|end-user-input|>`和`<|injection-point|>`）会在文本生成开始前在后台被移除。

        下面是一个Vicuna 1.1格式的示例：

        ```
        一个好奇的用户和一个人工智能助手之间的聊天。人工智能助手对用户的问题给出了有用、详细和礼貌的回答。

        用户：

         <|begin-user-input|>
        What datasets are mentioned in the text below?
        <|end-user-input|>

        <|injection-point|>

        ASSISTANT:
        ```

        ⚠️ 为获得最佳效果，请确保删除`ASSISTANT:`后的空格和新行字符。

        *此扩展目前是试验性的，正在开发中。

        """
            )
        )

    with gr.Row():
        with gr.Column(min_width=600):
            with gr.Tab("Text input"):
                data_input = gr.Textbox(lines=20, label="Input data")
                update_data = gr.Button("Load data")

            with gr.Tab("URL input"):
                url_input = gr.Textbox(
                    lines=10,
                    label="Input URLs",
                    info="输入一个或多个以换行符分隔的 URL。",
                )
                strong_cleanup = gr.Checkbox(
                    value=params["strong_cleanup"],
                    label="Strong cleanup",
                    info="只保留看起来像长文本的html元素。",
                )
                threads = gr.Number(
                    value=params["threads"],
                    label="线程",
                    info="下载URL时使用的线程数。",
                    precision=0,
                )
                update_url = gr.Button("加载数据")

            with gr.Tab("文件上传"):
                file_input = gr.File(label="Input file", type="binary")
                update_file = gr.Button("加载数据")

            with gr.Tab("生成设置"):
                chunk_count = gr.Number(
                    value=params["chunk_count"],
                    label="Chunk count",
                    info="在提示中包含的最接近匹配块的数量。",
                )
                gr.Markdown("时间加权（可选，用于使最近添加的块更有可能出现）)")
                time_weight = gr.Slider(
                    0,
                    1,
                    value=params["time_weight"],
                    label="Time weight",
                    info="定义时间加权的强度。0 = 无时间加权。",
                )
                chunk_count_initial = gr.Number(
                    value=params["chunk_count_initial"],
                    label="Initial chunk count",
                    info="在聊天模式下为时间权重重排序而检索的最接近匹配的数据块数量。该值应>=数据块数量。-1 = 提取所有的数据块。仅当time_weight > 0时使用。",
                )

                update_settings = gr.Button("Apply changes")

            chunk_len = gr.Number(
                value=params["chunk_length"],
                label="Chunk length",
                info='以字符为单位。点击 "加载数据 "时使用该值。".',
            )
            chunk_sep = gr.Textbox(
                value=params["chunk_separator"],
                label="Chunk separator",
                info='用于手动分割数据块。手动分割的数据块长度大于数据块长度时会再次分割。点击 "加载数据 "时使用该值。',
            )
        with gr.Column():
            last_updated = gr.Markdown()

    update_data.click(
        feed_data_into_collector,
        [data_input, chunk_len, chunk_sep],
        last_updated,
        show_progress=False,
    )
    update_url.click(
        feed_url_into_collector,
        [url_input, chunk_len, chunk_sep, strong_cleanup, threads],
        last_updated,
        show_progress=False,
    )
    update_file.click(
        feed_file_into_collector,
        [file_input, chunk_len, chunk_sep],
        last_updated,
        show_progress=False,
    )
    update_settings.click(
        apply_settings,
        [chunk_count, chunk_count_initial, time_weight],
        last_updated,
        show_progress=False,
    )
