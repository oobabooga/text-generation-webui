import time

import html
import functools
import re

import gradio
import numpy as np
import torch
from transformers import LogitsProcessor
import colorsys

from modules import html_generator, shared

params = {
    'active': True,
    'color_by_perplexity': False,
    'color_by_probability': False,
    'ppl_scale': 15.0,  # No slider for this right now, because I don't think it really needs to be changed. Very large perplexity scores don't show up often.
    'probability_dropdown': False,
    'verbose': False  # For debugging mostly
}


class PerplexityLogits(LogitsProcessor):
    def __init__(self, verbose=False):
        self.generated_token_ids = []
        self.selected_probs = []
        self.top_token_ids_list = []
        self.top_probs_list = []
        self.perplexities_list = []
        self.last_probs = None
        self.verbose = verbose

    def __call__(self, input_ids, scores):
        #t0 = time.time()
        probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        log_probs = torch.nan_to_num(torch.log(probs))  # Note: This is to convert log(0) nan to 0, but probs*log_probs makes this 0 not affect the perplexity.
        entropy = -torch.sum(probs * log_probs)
        entropy = entropy.cpu().numpy()
        perplexity = round(float(np.exp(entropy)), 4)
        self.perplexities_list.append(perplexity)
        last_token_id = int(input_ids[0][-1].cpu().numpy().item())
        # Store the generated tokens (not sure why this isn't accessible in the output endpoint!)
        self.generated_token_ids.append(last_token_id)
        # Get last probability, and add to the list if it wasn't there
        if len(self.selected_probs) > 0:
            # Is the selected token in the top tokens?
            if self.verbose:
                print(shared.tokenizer.decode(last_token_id), [shared.tokenizer.decode(token_id) for token_id in self.top_token_ids_list[-1][0]],
                    [round(float(prob), 4) for prob in self.top_probs_list[-1][0]])
            if last_token_id in self.top_token_ids_list[-1][0]:
                idx = self.top_token_ids_list[-1][0].index(last_token_id)
                self.selected_probs.append(self.top_probs_list[-1][0][idx])
            else:
                self.top_token_ids_list[-1][0].append(last_token_id)
                last_prob = round(float(self.last_probs[last_token_id]), 4)
                self.top_probs_list[-1][0].append(last_prob)
                self.selected_probs.append(last_prob)
        else:
            self.selected_probs.append(1.0)  # Placeholder for the last token of the prompt

        if self.verbose:
            pplbar = "-"
            if not np.isnan(perplexity):
                pplbar = "*" * round(perplexity)
            print(f"PPL for token after {shared.tokenizer.decode(last_token_id)}: {perplexity:.2f} {pplbar}")

        # Get top 5 probabilities
        top_tokens_and_probs = torch.topk(probs, 5)
        top_probs = top_tokens_and_probs.values.cpu().numpy().astype(float).tolist()
        top_token_ids = top_tokens_and_probs.indices.cpu().numpy().astype(int).tolist()

        self.top_token_ids_list.append(top_token_ids)
        self.top_probs_list.append(top_probs)

        probs = probs.cpu().numpy().flatten()
        self.last_probs = probs  # Need to keep this as a reference for top probs

        #t1 = time.time()
        #print(f"PPL Processor: {(t1-t0):.3f} s")
        # About 1 ms, though occasionally up to around 100 ms, not sure why...
        # Doesn't actually modify the logits!
        return scores


# Stores the perplexity and top probabilities
# global ppl_logits_processor
ppl_logits_processor = None


def logits_processor_modifier(logits_processor_list, input_ids):
    global ppl_logits_processor
    if params['active']:
        ppl_logits_processor = PerplexityLogits(verbose=params['verbose'])
        logits_processor_list.append(ppl_logits_processor)


def get_last_token(text, tokens_list, token_ids_list, token_probs_list):
    for token, token_id, prob in zip(tokens_list, token_ids_list, token_probs_list):
        if text.strip().endswith(token.strip()): # Whitespace could be a problem
            return token, token_id, prob
    # Unknown?
    print("Last token not found in list:", tokens_list)
    return '', -1, 0.0


def output_modifier(text):
    global ppl_logits_processor
    #t0 = time.time()
    original_text = text

    if not params['active'] or ppl_logits_processor is None:
        return text

    # Space at the beginning to account for tokenization spaces...
    text = ' ' + html.unescape(text)

    # TODO: It's probably more efficient to do this above rather than modifying all these lists
    # Remove last element of perplexities_list, top_token_ids_list, top_tokens_list, top_probs_list since everything is off by one because this extension runs before generation
    perplexities = ppl_logits_processor.perplexities_list
    top_token_ids_list = ppl_logits_processor.top_token_ids_list
    top_tokens_list = [[shared.tokenizer.decode(token_id) for token_id in top_token_ids[0]] for top_token_ids in top_token_ids_list]
    top_probs_list = ppl_logits_processor.top_probs_list
    # Remove first element of generated_token_ids, generated_tokens, selected_probs because they are for the last token of the prompt
    gen_token_ids = ppl_logits_processor.generated_token_ids[1:]
    # Add last sampled token, if possible (it could be past the end of the top 5 list)
    last_token, last_token_id, last_prob = get_last_token(text, top_tokens_list[-1], top_token_ids_list[-1][0], top_probs_list[-1][0])
    if last_token_id != -1:
        gen_token_ids.append(last_token_id)
    gen_tokens = [shared.tokenizer.decode(token_id) for token_id in gen_token_ids]
    sel_probs = ppl_logits_processor.selected_probs[1:]
    if last_token_id != -1:
        sel_probs.append(last_prob)

    end_part = '</div></div>' if params['probability_dropdown'] else '</span>'  # Helps with finding the index after replacing part of the text.

    # Initial space added to deal with some tokenizers...
    # Used to find where the message started generating, for working with "continue" generations
    # Doesn't work for longer messages... Not sure how I should handle this
    full_msg = shared.tokenizer.decode([token_id for token_id in gen_token_ids[:-1]]).strip()
    
    # There was an issue with tab lengths being off by one...
    # Seems like it might be model-dependent...
    #text = re.sub(r'( {3,})', r'\1 ', text)
    # Subtracting 2 to hopefully help with the tokenization spaces and continue issues,
    # Though it's possible it could overwrite the previous token if it's the same in the last 2 chars
    i = text.find(full_msg) - 2
    if i < 0:
        # Backup, try removing the extra whitespace (needed for continue)
        i = text.find(full_msg.strip()) - 2
        if i < 0:
            i = 0

    #i = 0
    # Add token index for ability to regenerate from there
    nonwhitespace_token_found = False
    missing_token_count = 0
    for index, token, prob, ppl, top_tokens, top_probs in zip(range(len(gen_tokens)), gen_tokens, sel_probs, perplexities, top_tokens_list, top_probs_list):
        # Somehow this works without issues, but not sure how...
        if not nonwhitespace_token_found and token.strip() == '':
            #print('Ignoring initial whitespace token...')
            continue
        nonwhitespace_token_found = True
        max_prob = top_probs[0][0]
        color = 'ffffff'
        if params['color_by_probability'] and params['color_by_perplexity']:
            color = probability_perplexity_color_scale(prob, max_prob, ppl)
        elif params['color_by_perplexity']:
            color = perplexity_color_scale(ppl)
        elif params['color_by_probability']:
            color = probability_color_scale(prob)
        if token.strip() in text[i:]:
            if params['probability_dropdown']:
                text = text[:i] + text[i:].replace(token.replace('\n', ''), add_dropdown_html(token, index, i, color, top_tokens, top_probs[0], ppl), 1)
            else:
                text = text[:i] + text[i:].replace(token.replace('\n', ''), add_color_html(token, color), 1)
            
            # This might be slightly inefficient
            i += text[i:].find(end_part) + len(end_part)
        else:
            missing_token_count += 1
            print('Missing token:', token, '...', text[i:i+20])
            # If there are any missing tokens, then either the tokenization was off, or this is the start of a conversation, or something else went wrong
        if missing_token_count > 5:
            print("Canceling token coloring...")
            return original_text


    # Use full perplexity list for calculating the average here.
    # Fix issue with mean of empty slice
    if len(ppl_logits_processor.perplexities_list) > 1:
        print('Average perplexity:', round(np.mean(ppl_logits_processor.perplexities_list[:-1]), 4))
    #t1 = time.time()
    #print(f"Output modifier: {(t1-t0):.3f} s")
    # About 50 ms
    return text.strip() # Remove extra beginning whitespace that some tokenizers add


def probability_color_scale(prob):
    '''
    Green-yellow-red color scale
    '''
    # hue (0.0 = red, 0.33 = green)
    # saturation (0.0 = gray / white, 1.0 = normal, just leave at 1.0)
    # brightness (0.0 = black, 1.0 = brightest, use something in between for better readability if you want...)
    hue = prob * 0.33
    rv, gv, bv = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    # to hex
    hex_col = f"{int(rv*255):02x}{int(gv*255):02x}{int(bv*255):02x}"

    return hex_col


def perplexity_color_scale(ppl):
    '''
    Red component only, white for 0 perplexity (sorry if you're not in dark mode)
    '''
    # hue (0.0 = red)
    # saturation (1.0 = red)
    # brightness (0.0 = black, 1.0 = red)
    # scale saturation from white to red the higher the perplexity

    ppl = min(ppl, params['ppl_scale'])  # clip ppl to 0-params['ppl_scale'] for color scaling. 15 should be fine for clipping and scaling
    sat = ppl / params['ppl_scale']
    rv, gv, bv = colorsys.hsv_to_rgb(0.0, sat, 1.0)

    # to hex
    hex_col = f"{int(rv*255):02x}{int(gv*255):02x}{int(bv*255):02x}"
    
    return hex_col


def probability_perplexity_color_scale(prob, max_prob, ppl):
    '''
    Green-yellow-red for relative probability compared to maximum for the current token, and blue component for perplexity
    '''
    hue = prob/max_prob * 0.33
    rv, gv, _ = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    
    ppl = min(ppl, params['ppl_scale'])  # clip ppl to 0-params['ppl_scale'] for color scaling. 15 should be fine for clipping and scaling
    bv = ppl / params['ppl_scale']
    
    # to hex
    hex_col = f"{int(rv*255):02x}{int(gv*255):02x}{int(bv*255):02x}"
    
    return hex_col


def add_color_html(token, color):
    output = ''
    output += f'<span style="color: #{color}">{html.escape(repr(token)[1:-1])}</span>'
    #if '\n' in token or '\r' in token: #token.isspace():
    #    output += '<br>'
    return output


# TODO: Might also need message index for the click-to-regenerate feature to work... For now it only works in the last message, which I think is fine.

# TODO: Major issue: Applying this to too many tokens will cause a permanent slowdown in generation speed until the messages are removed from the history. The slowdown seems to be mostly resolved in the current version though
# I think the issue is from HTML elements taking up space in the visible history, and things like history deepcopy add latency proportional to the size of the history.
# Potential solution is maybe to modify the main generation code to send just the internal text and not the visible history, to avoid moving too much around.
# I wonder if we can also avoid using deepcopy here.
def add_dropdown_html(token, index, msg_position, color, top_tokens, top_probs, perplexity=0):
    #print("Token:", token, token.isspace(), '\n' in token or '\r' in token)
    output = ''
    # Use the repr to get characters like \n visible. Exclude the quotes around it
    output += f'<div class="hoverable" name="tok_{index}_{msg_position}"><span style="color: #{color}">{html.escape(repr(token)[1:-1])}</span><div class="dropdown"><table class="dropdown-content"><tbody>'
    for i, token_option, prob in zip(range(len(top_tokens)), top_tokens, top_probs):
        # TODO: Bold for selected token?
        # Using divs prevented the problem of divs inside spans causing issues.
        # Now the problem is that divs show the same whitespace of one space between every token.
        # There is probably some way to fix this in CSS that I don't know about.
        row_color = probability_color_scale(prob)
        row_class = ' class="selected"' if token_option == token else ''
        # This time we want to include the quotes around it so that we can see where the spaces are.
        output += f'<tr{row_class}><td name="opt_{index}_{i}_{msg_position}" style="color: #{row_color}">{html.escape(repr(token_option))}</td><td style="color: #{row_color}">{prob:.4f}</td></tr>'
    if perplexity != 0:
        ppl_color = perplexity_color_scale(perplexity)
        output += f'<tr><td>Perplexity:</td><td style="color: #{ppl_color}">{perplexity:.4f}</td></tr>'
    output += '</tbody></table></div></div>'
    #if '\n' in token or '\r' in token: #token.isspace():
    #    output += '<br>' # I imagine this will cause problems sometimes
    return output  # About 750 characters per token...


def custom_css():
    return """
        .dropdown {
            display: none;
            position: absolute;
            z-index: 50;
            background-color: var(--background-fill-secondary);
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,1.0);
            width: max-content;
            overflow: visible;
            padding: 5px;
            border-radius: 10px;
            border: 1px solid var(--border-color-primary);
        }

        .dropdown-content {
            border: none;
            z-index: 50;
        }

        .dropdown-content tr.selected {
            background-color: var(--background-fill-primary);
        }

        .dropdown-content td {
            color: var(--body-text-color);
        }

        .hoverable {
            color: var(--body-text-color);
            position: relative;
            display: inline-block;
            overflow: visible;
            font-size: 15px;
            line-height: 1.75;
            margin: 0;
            padding: 0;
        }

        .hoverable:hover .dropdown {
            display: block;
        }

        pre {
            white-space: pre-wrap;
        }

        # TODO: This makes the hover menus extend outside the bounds of the chat area, which is good.
        # However, it also makes the scrollbar disappear, which is bad.
        # The scroll bar needs to still be present. So for now, we can't see dropdowns that extend past the edge of the chat area.
        .chat {
            overflow-y: auto;
        }
    """

def custom_js():
    return """

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}    

// Note that this will only work as intended on the last agent message
document.addEventListener("click", async function(event) {
    //console.log(event.target);
    const name = event.target.getAttribute("name");
    if (name != null && name.includes("opt_")) {
        const name_parts = name.split("_");
        const token_index = name_parts[1];
        const option_index = name_parts[2];
        const msg_pos = name_parts[3];
        // Exclude the quotes and convert newlines... Not sure about the newlines though
        // TODO: Seems like continuing generation from a newline causes problems whether you add it or not!
        const token_string = event.target.innerHTML.substring(1, event.target.innerHTML.length-1).replace(new RegExp(String.fromCharCode(92)+String.fromCharCode(92)+"r", "g"), '').replace(new RegExp(String.fromCharCode(92)+String.fromCharCode(92)+"n", "g"), '');
        //console.log(token_index + ", " + option_index + ", " + token_string);
        // Get all the previous text (I'm sure there is a more efficient way to do this)
        var msg_text = ""
        const msg_html = event.target.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement;
        var msg_parts = msg_html.childNodes;
        for (var i = 0; i < msg_parts.length; i++) {
            var msg_part = msg_parts[i];
            if (msg_part.nodeType === Node.ELEMENT_NODE) {
                if (msg_part.nodeName == "DIV") {
                    msg_part_name = msg_part.getAttribute("name")
                    if (msg_part_name != null) {
                        var current_token_index = msg_part_name.split("_")[1];
                        var current_message_pos = msg_part_name.split("_")[2];
                        if (current_token_index == token_index && current_message_pos == msg_pos) {
                            // Use the replacement token
                            // TODO: Don't have access to the tokenizer here, and sometimes there needs to be a space added before this token
                            msg_text += token_string //.replace(new RegExp(String.fromCharCode(92)+String.fromCharCode(92)+"r", "g"), '').replace(new RegExp(String.fromCharCode(92)+String.fromCharCode(92)+"n", "g"), '');
                            break;
                        }
                        else {
                            // Replace here or at the end?
                            var text = msg_part.firstChild.innerHTML.replace(new RegExp(String.fromCharCode(92)+String.fromCharCode(92)+"r", "g"), '').replace(new RegExp(String.fromCharCode(92)+String.fromCharCode(92)+"n", "g"), '')
                            msg_text += text;
                        }
                    }
                }
                else {
                    // Break tag (hacky workaround because the newline literal can't be parsed here)
                    //msg_text += String.fromCharCode(10);
                    // Do nothing???
                }
            }
            else if (msg_part.nodeType === Node.TEXT_NODE) {
                msg_text +=  msg_part.textContent;
            }
        }
        var textbox = document.querySelector("#chat-input textarea");
        textbox.focus();
        textbox.value = msg_text.trimStart() // Fix initial tokenization spaces
        //console.log(textbox.value);
        
        // Add some delays to make sure it's processed correctly. Without these, there's a chance the events don't go through correctly and it doesn't work
        // It's unknown how long this will take, and probably depends on the size of the message...
        // It would be better to somehow wait for gradio to update instead of waiting a fixed amount of time.
        // Hopefully 1 second of delay before starting generation isn't unacceptable.
        var inputEvent = new Event('input', {
            bubbles: true,
            cancelable: true,
        });
        textbox.dispatchEvent(inputEvent);
        var changeEvent = new Event('change', {
            bubbles: true,
            cancelable: true,
        });
        textbox.dispatchEvent(changeEvent);
        await sleep(250);
        document.getElementById("Replace-last").click();
        // This can take a while to execute
        await sleep(750);
        document.getElementById("Continue").click();
    }
});

console.log("Custom JS for perplexity_colors loaded");
"""

# Monkeypatch applied to html_generator.py
# We simply don't render markdown into HTML. We wrap everything in <pre> tags to preserve whitespace
# formatting. If you're coloring tokens by perplexity or probability, or especially if you're using
# the probability dropdown, you probably care more about seeing the tokens the model actually outputted
# rather than rendering ```code blocks``` or *italics*.
@functools.lru_cache(maxsize=4096)
def convert_to_markdown(string):
    return '<pre>' + string + '</pre>'

def convert_to_markdown_wrapped(string, use_cache=True):
    if use_cache:
        return convert_to_markdown(string)
    return convert_to_markdown.__wrapped__(string)

# This is still necessary for formatting to work correctly
html_generator.convert_to_markdown = convert_to_markdown


def ui():
    def update_active_check(x):
        params.update({'active': x})

    def update_color_by_ppl_check(x):
        params.update({'color_by_perplexity': x})

    def update_color_by_prob_check(x):
        params.update({'color_by_probability': x})

    def update_prob_dropdown_check(x):
        params.update({'probability_dropdown': x})

    active_check = gradio.Checkbox(value=True, label="Compute probabilities and perplexity scores", info="Activate this extension. Note that this extension currently does not work with llama.cpp, but it does work with ExLlamav2_HF and llamacpp_HF when set up correctly")
    color_by_ppl_check = gradio.Checkbox(value=False, label="Color by perplexity", info="Higher perplexity is more red. If also showing probability, higher perplexity has more blue component.")
    color_by_prob_check = gradio.Checkbox(value=False, label="Color by probability", info="Green-yellow-red linear scale, with 100% green, 50% yellow, 0% red.")
    prob_dropdown_check = gradio.Checkbox(value=False, label="Probability dropdown", info="Hover over a token to show a dropdown of top token probabilities. Currently slightly buggy with whitespace between tokens.")

    active_check.change(update_active_check, active_check, None)
    color_by_ppl_check.change(update_color_by_ppl_check, color_by_ppl_check, None)
    color_by_prob_check.change(update_color_by_prob_check, color_by_prob_check, None)
    prob_dropdown_check.change(update_prob_dropdown_check, prob_dropdown_check, None)
