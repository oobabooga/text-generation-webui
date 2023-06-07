import json
import requests
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4.element import Comment

import gradio as gr
import modules.chat as chat
import modules.shared as shared
import requests
import torch

torch._C._jit_set_profiling_mode(False)

params = { # These can all be set in the settings.yml file
    'enable_search': True,
    'include_first_result_content': True,
    'include_result_summary': False,
    'max_characters_per_page': 4096,
    'searx_instance': "",
    'max_total_characters': 8192,
    'extra_query_information': "",
    'removal_list': ['\t', '\n', '\\n', '\\t'],
    'number_of_results': 1,
    'console_log': True
}

html_element_blacklist = [
    '[document]',
    'noscript',
    'header',
    'meta',
    'head', 
    'input',
    'script',
    'style'
]


def url_to_text(url):
    
    html = urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(html_element_blacklist):
        tag.decompose()
    text = soup.get_text(strip=True)
    for string in params['removal_list']:
        text.replace(string, " ")
    return text[0:params['max_characters_per_page']]

def get_search_term(string): # This checks if you say "search ... about" something, and if the "Activate Searx integration" checkbox is ticked will search about that

    commands = ['search', 'tell me', 'give me a summary']
    marker = ['about']
    lowstr = string.lower()
    for s in ['\"', '\'']:
        lowstr = lowstr.replace(s, '')
    if any(command in lowstr for command in commands) and any(case in lowstr for case in marker):
        print("Found search term")
        subject = string.split('about',1)[1]
        return subject

def search_string(search_term): # This is the main logic that sends the API request to Searx and returns the text to add to the context

    print("Searching about" + search_term + "...")
    query = f"{search_term} {params['extra_query_information']}"
    r = requests.get(params['searx_instance'], params={'q': query,'format': 'json','pageno': '1'})
    try:
        searchdata = r.json()
        searchdata = searchdata['results']
    except:
        new_context = "Tell me that you could not find the results I asked for."
    else:
        new_context = f"This is new information from after your knowledge cutoff date about {search_term} :\n"
        if params['include_first_result_content']:
            for i in range(params['number_of_results']):
                new_context += url_to_text(searchdata[i]['url']) + "\n"
        if params['include_result_summary']:
            for result in searchdata:
                if 'content' in result:
                    new_context += result['content'] + "\n"
        new_context = new_context[0:params['max_total_characters']]
    finally:
        if params['console_log']:
            print(new_context)
        return new_context

def input_modifier(string):

    if params['enable_search'] and params['searx_instance']:
        if get_search_term(string):
            search_result = search_string(string)
            if search_result == "Tell me that you could not find the results I asked for.": # If it failed to get a result, ask the LLM to tell user it did
                return search_result
            else:
                return f"{search_result} Using the information I just gave you, without adding any thing new, respond to this request: {string}"
    return string

def ui():
    
    with gr.Accordion("Searx Integration", open=True):
        with gr.Row():
            with gr.Column():
                enable_search = gr.Checkbox(value=params['enable_search'], label='Activate Searx integration')
                console_log = gr.Checkbox(value=params['console_log'], label='Display search results on console')
                include_first_result_content = gr.Checkbox(value=params['include_first_result_content'], label='Include content from the first result')
                number_of_results = gr.Slider(1,10,value=params['number_of_results'],step=1,label='Number of results to fetch')
            with gr.Column():
                searx_instance = gr.Textbox(placeholder=params['searx_instance'], value=params['searx_instance'], label='Searx instance address')
                extra_query_information = gr.Textbox(placeholder=params['extra_query_information'], value=params['extra_query_information'], label='Extra info to pass in Searx query')
                include_result_summary = gr.Checkbox(value=params['include_result_summary'], label='Include summary from each search result')
                max_characters_per_page = gr.Slider(256,16384,value=params['max_characters_per_page'],step=64,label='Maximum characters per fetched pages')
                max_total_characters = gr.Slider(256,16384,value=params['max_total_characters'],step=64,label='Total max characters')

    enable_search.change(lambda x: params.update({"enable_search": x}), enable_search, None)
    console_log.change(lambda x: params.update({"console_display": x}), console_log, None)
    include_first_result_content.change(lambda x: params.update({"include_first_result_content": x}), include_first_result_content, None)
    include_result_summary.change(lambda x: params.update({"include_result_summary": x}), include_result_summary, None)
    number_of_results.change(lambda x: params.update({"number_of_results": x}), number_of_results, None)
    max_characters_per_page.change(lambda x: params.update({"max_characters_per_page": x}), max_characters_per_page, None)
    searx_instance.change(lambda x: params.update({"searx_instance": x}), searx_instance, None)
    extra_query_information.change(lambda x: params.update({"extra_query_information": x}), extra_query_information, None)
    max_total_characters.change(lambda x: params.update({"max_total_characters": x}), max_total_characters, None)