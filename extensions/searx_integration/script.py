import urllib.request
import html2text
import json
import requests
import io
import re

import gradio as gr
import modules.chat as chat
import modules.shared as shared
import requests
import torch

torch._C._jit_set_profiling_mode(False)

params = { # These can all be set in the settings.yml file
    'enable_search': False,
    'include_first_result': False,
    'max_bytes_first_result': 5000,
    'searx_instance': "",
    'max_bytes_total': 50000,
    'removal_list': ['\t', '\n', '\\n', '\\t'],
    'console_display': False
}

initial_context = "This is new information from after your knowledge cutoff date about"
do_search = False

def get_search_term(string): # This checks if you say "search ... about" something, and if the "Activate Searx integration" checkbox is ticked will search about that

    global do_search
    commands = ['search', 'tell me', 'give me a summary']
    marker = ['about']
    lowstr = string.lower()
    if any(command in lowstr for command in commands) and any(case in lowstr for case in marker):
        print("Found search term")
        do_search = True
        subject = string.split('about',1)[1]
        return subject

def search_string(search_term): # This is the main logic that sends the API request to Searx and returns the text to add to the context

    global do_search
    print("Searching about" + search_term + "...")
    do_search = False
    r = requests.get(params['searx_instance'], params={'q': search_term,'format': 'json','pageno': '1'})
    searchdata = r.json()
    searchdata = searchdata['results']
    new_context = initial_context + search_term + ":"
    if params['include_first_result']:
        weburl = urllib.request.urlopen(searchdata[0]['url'])
        page_content = html2text.html2text(str(weburl.read()))
        page_content = re.sub(r"[\n\t]*", "",page_content)
        for s in params['removal_list']:
            page_content = page_content.replace(s, '')
        page_content = io.StringIO(page_content)
        new_context = new_context + page_content.read(params['max_bytes_first_result'])
    for result in searchdata:
        if 'content' in result:
            summary = result['content']
            new_context = new_context + "\n" + summary
    new_context = io.StringIO(new_context)
    new_context = new_context.read(params['max_bytes_total'])
    new_context = new_context + "\n###"
    if params['console_display']:
        print(new_context)
    return new_context

def input_modifier(string):

    if params['enable_search'] and params['searx_instance']:
        search_term = get_search_term(string)
        if do_search:
            search_result = search_string(search_term)
            string = search_result + "\nNow here is my request: " + string
    return string


def ui():

    with gr.Accordion("Searx Integration", open=True):
        with gr.Row():
            with gr.Column():
                enable_search = gr.Checkbox(value=params['enable_search'], label='Activate Searx integration')
                console_display = gr.Checkbox(value=params['console_display'], label='Display search results on console')
#                include_first_result = gr.Checkbox(value=params['include_first_result'], label='Insert text of first search result')
#                max_bytes_first_result = gr.Slider(10000,100000,value=params['max_bytes_first_result'],step=100,label='Maximum bytes of first result added to context')
            with gr.Column():
                searx_instance = gr.Textbox(placeholder=params['searx_instance'], value=params['searx_instance'], label='Searx instance address')
                max_bytes_total = gr.Slider(10000,100000,value=params['max_bytes_total'],step=100,label='Total max bytes added to context')

    enable_search.change(lambda x: params.update({"enable_search": x}), enable_search, None)
    console_display.change(lambda x: params.update({"console_display": x}), console_display, None)
#    include_first_result.change(lambda x: params.update({"include_first_result": x}), include_first_result, None)
#    max_bytes_first_result.change(lambda x: params.update({"max_bytes_first_result": x}), max_bytes_first_result, None)
    searx_instance.change(lambda x: params.update({"searx_instance": x}), searx_instance, None)
    max_bytes_total.change(lambda x: params.update({"max_bytes_total": x}), max_bytes_total, None)
