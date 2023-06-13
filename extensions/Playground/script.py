import gradio as gr
import pickle
import modules.shared as shared
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from modules.text_generation import generate_reply
from modules.text_generation import generate_reply_wrapper
from modules.text_generation import stop_everything_event
from modules.ui import list_interface_input_elements
from modules.ui import gather_interface_values
from modules.html_generator import generate_basic_html

right_symbol = '\U000027A1'
left_symbol = '\U00002B05'


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

try:
    with open('notebook.sav', 'rb') as f:
        params = pickle.load(f)
except FileNotFoundError:
    params = {
        "display_name": "Playground",
        "is_tab": True,
        "usePR": False,
        "pUSER": 'USER:',
        "pBOT": 'ASSISTANT:',
        "selectA": [0,0],
        "selectB": [0,0]
    }


def get_last_line(string):
    lines = string.splitlines()
    if lines:
        last_line = lines[-1]
        return last_line
    else:
        return ""


def input_modifier(string):

    global params
    modified_string = string
    addLineReply = ""

    if params['usePR']:
        if "---" in string:
            lines = string.splitlines()  # Split the text into lines
            modified_string = ""
            for i, line in enumerate(lines):
                if addLineReply:
                    line.lstrip()
                    line = addLineReply + line
                    addLineReply = ""
                elif line.startswith("---"):
                    line = line.replace("---", params['pUSER'])  
                    addLineReply = params['pBOT'] 
                    
                modified_string = modified_string+ line +"\n"

            if addLineReply:
                modified_string = modified_string + addLineReply

    return modified_string


    
def output_modifier(string):
    #print(f"output_modifier: {string}") 
    return string

def copynote(string):
    return string

def formatted_outputs(reply):
 
    return reply, generate_basic_html(reply)



def generate_reply_wrapperMYSEL(question, state,selectState):

    global params
    selF = params[selectState][0]
    selT = params[selectState][1]
    if not selF==selT:
        print(f"\033[1;32;1m\nGenerarting from selected text in {selectState} and inserting after {params[selectState]}\033[0;37;0m")
        params[selectState] = [0,0]
        before = question[:selF]
        current = question[selF:selT]
        after = question[selT:]
    else:
        current = question
        before = ""
        after = ""
        print(f"\033[1;31;1m\nNo selection in {selectState}, reverting to full text Generate\033[0;37;0m") 
        

    # if use quick prompt, add \n if none
    if params['usePR']:
        if not current.endswith("\n"):
            lastline = get_last_line(current)
            if lastline.startswith("---"):
                current+="\n"

    for reply in generate_reply(current, state, eos_token = None, stopping_strings=None, is_chat=False):
        if shared.model_type not in ['HF_seq2seq']:
            reply = current + reply
        reply = before+reply+after
        yield formatted_outputs(reply)

def generate_reply_wrapperMY(question, state, selectState):

    global params
    params[selectState] = [0,0]
    # if use quick prompt, add \n if none
    if params['usePR']:
        if not question.endswith("\n"):
            lastline = get_last_line(question)
            if lastline.startswith("---"):
                question+="\n"

    for reply in generate_reply(question, state, eos_token = None, stopping_strings=None, is_chat=False):
        if shared.model_type not in ['HF_seq2seq']:
            reply = question + reply

        yield formatted_outputs(reply)

def ui():
    #input_elements = list_interface_input_elements(chat=False)
    #interface_state = gr.State({k: None for k in input_elements})
    global params
    global switchAB
    switchAB = int(0)

    params['selectA'] = [0,0]
    params['selectB'] = [0,0]

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Tab('Text'):
                    with gr.Row():
                        text_boxA = gr.Textbox(value='', elem_classes="textbox", lines=20, label = 'Notebook A')
                with gr.Tab('HTML'):
                    with gr.Row():
                        htmlA = gr.HTML()
            with gr.Row():
                with gr.Column(scale=10):
                     with gr.Row():
                        generate_btnA = gr.Button('Generate', variant='primary', elem_classes="small-button")
                        generate_SelA = gr.Button('Generate [SEL]', variant='primary', elem_classes="small-button")
                        stop_btnA = gr.Button('Stop', elem_classes="small-button")
                with gr.Column(scale=1, min_width=50):    
                    toNoteB = ToolButton(value=right_symbol)
       
            with gr.Row():
                with gr.Box():
                    with gr.Column():    
                        usePR = gr.Checkbox(value=params['usePR'], label='Enable Quick Instruct (line starts with three dashes --- )')
                        with gr.Row():
                            preset_type = gr.Dropdown(label="Preset", choices=["Custom", "Vicuna", "Alpaca", "Guanaco", "OpenAssistant"], value="Custom")
                            text_USR = gr.Textbox(value=params['pUSER'], lines=1, label='User string')
                            text_BOT = gr.Textbox(value=params['pBOT'], lines=1, label='Bot string')
                        gr.Markdown('Example: --- What are synonyms for happy?')    

        with gr.Column():
            with gr.Row():
                with gr.Tab('Text'):
                    with gr.Row():
                        text_boxB = gr.Textbox(value='', elem_classes="textbox", lines=20, label = 'Notebook B')
                with gr.Tab('HTML'):
                    with gr.Row():
                        htmlB = gr.HTML()
            with gr.Row():
                with gr.Column(scale=10):
                    with gr.Row():    
                        generate_btnB = gr.Button('Generate', variant='primary', elem_classes="small-button")
                        generate_SelB = gr.Button('Generate [SEL]',variant='primary', elem_classes="small-button")
                        stop_btnB = gr.Button('Stop', elem_classes="small-button")
                with gr.Column(scale=1, min_width=50):       
                    toNoteA = ToolButton(value=left_symbol)
            with gr.Row():
                with gr.Box():
                    with gr.Column():    
                        
                        with gr.Row():
                            gr.Markdown('v 6/12/2023 FPHam')    


    selectStateA = gr.State('selectA')
    selectStateB = gr.State('selectB')
        #shared.gradio['Undo'] = gr.Button('Undo', elem_classes="small-button")
        #shared.gradio['Regenerate'] = gr.Button('Regenerate', elem_classes="small-button")
    # Todo:
    # add silider for independend temperature, top_p and top_k
    # shared.input_elements, shared.gradio['top_p'] 
    input_paramsA = [text_boxA,shared.gradio['interface_state'],selectStateA]
    output_paramsA =[text_boxA, htmlA]
    input_paramsB = [text_boxB,shared.gradio['interface_state'],selectStateB]
    output_paramsB =[text_boxB, htmlB]
  
    generate_btnA.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMY, inputs=input_paramsA, outputs= output_paramsA, show_progress=False)
    
    generate_SelA.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMYSEL, inputs=input_paramsA, outputs=output_paramsA, show_progress=False)

    stop_btnA.click(stop_everything_event, None, None, queue=False)

    toNoteA.click(copynote, text_boxB, text_boxA)

    generate_btnB.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMY, inputs=input_paramsB, outputs=output_paramsB, show_progress=False)
    
    generate_SelB.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        generate_reply_wrapperMYSEL, inputs=input_paramsB, outputs=output_paramsB, show_progress=False)

    stop_btnB.click(stop_everything_event, None, None, queue=False)

    toNoteB.click(copynote, text_boxA, text_boxB)    
   

    def on_selectA(evt: gr.SelectData):  # SelectData is a subclass of EventData
        #print (f"You selected {evt.value} at {evt.index} from {evt.target}")
        global params
        params['selectA'] = evt.index
        return ""
    
    def on_selectB(evt: gr.SelectData):  # SelectData is a subclass of EventData
        #print (f"You selected {evt.value} at {evt.index} from {evt.target}")
        global params
        params['selectB'] = evt.index
        return ""


    text_boxA.select(on_selectA, None, None)
    text_boxB.select(on_selectB, None, None)

    def save_pickle():
        global params
        with open('notebook.sav', 'wb') as f:
            pickle.dump(params, f)
 
    
    def update_activate(x):
        global params
        params.update({"usePR": x})
        save_pickle()
 
    def update_activate(x):
        global params
        params.update({"usePR": x})
        save_pickle()
     
    def update_stringU(x):
        global params
        params.update({"pUSER": x})
        save_pickle()

    def update_stringB(x):
        global params
        params.update({"pBOT": x})
        save_pickle()

    def update_preset(x):
        if x == "Vicuna":
            return 'USER:','ASSISTANT:'
        elif x == "Alpaca":
            return '### Instruction:','### Response:'
        elif x == "Guanaco":
            return '### Human:','### Assistant:'
        elif x == "OpenAssistant":
            return '<|prompter|>','<|endoftext|><|assistant|>'
        
        return 'USER:','ASSISTANT:'           


    usePR.change(update_activate, usePR, None)   
    text_USR.change(update_stringU, text_USR, None) 
    text_BOT.change(update_stringB, text_BOT, None) 
    preset_type.change(update_preset,preset_type,[text_USR,text_BOT])

