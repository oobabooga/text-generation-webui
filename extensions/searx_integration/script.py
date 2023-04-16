import urllib.request
import trafilatura
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
    'include_first_results': True,
    'include_result_summary': False,
    'max_bytes_for_fetched_pages': 5000,
    'searx_instance': "",
    'max_bytes_total': 50000,
    'extra_query_information': "!ddg",
    'removal_list': ['\t', '\n', '\\n', '\\t'],
    'number_of_results': 1,
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
    query = search_term + " " + params['extra_query_information']
    r = requests.get(params['searx_instance'], params={'q': query,'format': 'json','pageno': '1'})
    searchdata = r.json()
    searchdata = searchdata['results']
    new_context = initial_context + search_term + ":\n"
    if params['include_first_results']:
        i = 0
        while i < params['number_of_results']:
            webpage = trafilatura.fetch_url(searchdata[i]['url'])
            page_content = trafilatura.extract(webpage, include_comments=False, include_tables=False, no_fallback=True)
    #        page_content = html2text.html2text(str(weburl.read()))
    #        page_content = re.sub(r"[\n\t]*", "",page_content)
    #        for s in params['removal_list']:
    #            page_content = page_content.replace(s, '')
            page_content = io.StringIO(page_content)
            new_context = new_context + page_content.read(params['max_bytes_for_fetched_pages'])
            i = i + 1
    if params['include_result_summary']:
        for result in searchdata:
            if 'content' in result:
                summary = result['content']
                new_context = new_context + "\n" + summary
    new_context = io.StringIO(new_context)
    new_context = new_context.read(params['max_bytes_total'])
    new_context = new_context + "\n"
    if params['console_display']:
        print(new_context)
    return new_context


def input_modifier(string):

    if params['enable_search'] and params['searx_instance']:
        search_term = get_search_term(string)
        if do_search:
            search_result = search_string(search_term)
            string = search_result + "\n\nUsing the information I just gave you, and not adding any thing new, respond to this request:" + string
    return string


def ui():

    with gr.Accordion("Searx Integration", open=True):
        with gr.Row():
            with gr.Column():
                enable_search = gr.Checkbox(value=params['enable_search'], label='Activate Searx integration')
                console_display = gr.Checkbox(value=params['console_display'], label='Display search results on console')
                include_first_results = gr.Checkbox(value=params['include_first_results'], label='Insert text of search result')
                number_of_results = gr.Slider(1,10,value=params['number_of_results'],step=1,label='Number of results to fetch')
            with gr.Column():
                searx_instance = gr.Textbox(placeholder=params['searx_instance'], value=params['searx_instance'], label='Searx instance address')
                extra_query_information = gr.Textbox(placeholder=params['extra_query_information'], value=params['extra_query_information'], label='Extra info to pass in Searx query')
                include_result_summary = gr.Checkbox(value=params['include_result_summary'], label='Insert summary of first page of search results')
                max_bytes_for_fetched_pages = gr.Slider(10000,100000,value=params['max_bytes_for_fetched_pages'],step=100,label='Maximum bytes of fetched pages added to context')
                max_bytes_total = gr.Slider(10000,100000,value=params['max_bytes_total'],step=100,label='Total max bytes added to context')

    enable_search.change(lambda x: params.update({"enable_search": x}), enable_search, None)
    console_display.change(lambda x: params.update({"console_display": x}), console_display, None)
    include_first_results.change(lambda x: params.update({"include_first_results": x}), include_first_results, None)
    include_result_summary.change(lambda x: params.update({"include_result_summary": x}), include_result_summary, None)
    number_of_results.change(lambda x: params.update({"number_of_results": x}), number_of_results, None)
    max_bytes_for_fetched_pages.change(lambda x: params.update({"max_bytes_for_fetched_pages": x}), max_bytes_for_fetched_pages, None)
    searx_instance.change(lambda x: params.update({"searx_instance": x}), searx_instance, None)
    extra_query_information.change(lambda x: params.update({"extra_query_information": x}), extra_query_information, None)
    max_bytes_total.change(lambda x: params.update({"max_bytes_total": x}), max_bytes_total, None)
