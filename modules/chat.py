import base64
import copy
import functools
import html
import json
import pprint
import re
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image

import modules.shared as shared
from modules import utils
from modules.extensions import apply_extensions
from modules.html_generator import (
    chat_html_wrapper,
    convert_to_markdown,
    make_thumbnail
)
from modules.logging_colors import logger
from modules.text_generation import (
    generate_reply,
    get_encoded_length,
    get_max_prompt_length
)
from modules.tool_calling import (
    split_message_by_tool_calls,
    get_visible_tools,
    define_tool_action,
    process_tool_calls,
    extract_tool_calls_to_be_executed,
    execute_tool_call
)
from modules.utils import delete_file, get_available_characters, save_file


def strftime_now(format):
    return datetime.now().strftime(format)


jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
jinja_env.globals["strftime_now"] = strftime_now


def str_presenter(dumper, data):
    """
    Copied from https://github.com/yaml/pyyaml/issues/240
    Makes pyyaml output prettier multiline strings.
    """

    if data.count('\n') > 0:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)


def get_generation_prompt(renderer, impersonate=False, strip_trailing_spaces=True):
    '''
    Given a Jinja template, reverse-engineers the prefix and the suffix for
    an assistant message (if impersonate=False) or an user message
    (if impersonate=True)
    '''

    if impersonate:
        messages = [
            {"role": "user", "content": "<<|user-message-1|>>"},
            {"role": "user", "content": "<<|user-message-2|>>"},
        ]
    else:
        messages = [
            {"role": "assistant", "content": "<<|user-message-1|>>"},
            {"role": "assistant", "content": "<<|user-message-2|>>"},
        ]

    prompt = renderer(messages=messages)

    suffix_plus_prefix = prompt.split("<<|user-message-1|>>")[1].split("<<|user-message-2|>>")[0]
    suffix = prompt.split("<<|user-message-2|>>")[1]
    prefix = suffix_plus_prefix[len(suffix):]

    if strip_trailing_spaces:
        prefix = prefix.rstrip(' ')

    return prefix, suffix


def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    also_return_rows = kwargs.get('also_return_rows', False)
    history = kwargs.get('history', state['history'])['internal']

    # Templates
    chat_template_str = state['chat_template_str']
    if state['mode'] != 'instruct':
        chat_template_str = replace_character_names(chat_template_str, state['name1'], state['name2'])

    instruction_template = jinja_env.from_string(state['instruction_template_str'])
    chat_template = jinja_env.from_string(chat_template_str)

    instruct_renderer = partial(
        instruction_template.render,
        #builtin_tools=None, # Somehow this is still being used? I'm getting an error of NoneType is not iterable on Llama 3 8B (but not Llama 3 1B)
        tools=get_visible_tools(state), # Visible tools don't include the source code for the tool function
        tools_in_user_message=state['tools_in_user_message'],
        add_generation_prompt=False
    )

    chat_renderer = partial(
        chat_template.render,
        add_generation_prompt=False,
        name1=state['name1'],
        name2=state['name2'],
        user_bio=replace_character_names(state['user_bio'], state['name1'], state['name2']),
        tools=get_visible_tools(state),
        tools_in_user_message=state['tools_in_user_message'],
    )
    # Adding tools to the chat renderer did not work. Currently tools only work in instruct mode.
    # Might need the system prompt, but adding it to the chat renderer would cause compatibility issues with existing chats...

    messages = []

    if state['mode'] == 'instruct':
        renderer = instruct_renderer
        if state['custom_system_message'].strip() != '':
            messages.append({"role": "system", "content": state['custom_system_message']})
    else:
        renderer = chat_renderer
        if state['context'].strip() != '' or state['user_bio'].strip() != '':
            context = replace_character_names(state['context'], state['name1'], state['name2'])
            messages.append({"role": "system", "content": context})

    insert_pos = len(messages)
    for user_msg, assistant_msg in reversed(history):
        user_msg = user_msg.strip()
        assistant_msg = assistant_msg.strip()

        if assistant_msg:
            # The assistant message may contain tool calls and responses, so split it into multiple messages
            assistant_msgs = reversed(split_message_by_tool_calls(assistant_msg))
            for msg in assistant_msgs:
                messages.insert(insert_pos, msg)
            # messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg})

        if user_msg not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            messages.insert(insert_pos, {"role": "user", "content": user_msg})

    user_input = user_input.strip()
    if user_input and not impersonate and not _continue:
        messages.append({"role": "user", "content": user_input})

    # Add tools from the last response. This 'last_assistant_response' param is used to store the part of the message before the tool call, if any
    # It could probably use a better name though
    last_assistant_response = kwargs.get('last_assistant_response', None)
    print("Last assistant response:", last_assistant_response)
    if last_assistant_response is not None:
        if last_assistant_response != "":
            messages.append({"role": "assistant", "content": last_assistant_response})
        # Add the messages for tool calls and tool responses
        # The content field is ignored (at least in Llama 3, which is why it needs to be added to a previous message)
        if 'tool_calls' in kwargs and len(kwargs.get('tool_calls', [])) > 0:
            messages.append({"role": "assistant", "content": "", "tool_calls": kwargs.get('tool_calls', [])})
        for tool_result in kwargs.get('tool_results', []):
            messages.append(tool_result) # {"role": "tool", "tool_call_id": ..., "content": ...}

    # Debug: Display messages before rendering the template
    print("Messages:")
    for msg in messages:
        print(msg)

    def remove_extra_bos(prompt):
        for bos_token in ['<s>', '<|startoftext|>', '<BOS_TOKEN>', '<|endoftext|>']:
            while prompt.startswith(bos_token):
                prompt = prompt[len(bos_token):]

        return prompt

    def make_prompt(messages):
        if state['mode'] == 'chat-instruct' and _continue:
            prompt = renderer(messages=messages[:-1])
        else:
            prompt = renderer(messages=messages)

        if state['mode'] == 'chat-instruct':
            outer_messages = []
            if state['custom_system_message'].strip() != '':
                outer_messages.append({"role": "system", "content": state['custom_system_message']})

            prompt = remove_extra_bos(prompt)
            command = state['chat-instruct_command']
            command = command.replace('<|character|>', state['name2'] if not impersonate else state['name1'])
            command = command.replace('<|prompt|>', prompt)
            command = replace_character_names(command, state['name1'], state['name2'])

            if _continue:
                prefix = get_generation_prompt(renderer, impersonate=impersonate, strip_trailing_spaces=False)[0]
                prefix += messages[-1]["content"]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

            outer_messages.append({"role": "user", "content": command})
            outer_messages.append({"role": "assistant", "content": prefix})

            prompt = instruction_template.render(messages=outer_messages)
            suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
            if len(suffix) > 0:
                prompt = prompt[:-len(suffix)]

        else:
            if _continue:
                suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
                if len(suffix) > 0:
                    prompt = prompt[:-len(suffix)]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if state['mode'] == 'chat' and not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

                prompt += prefix

        prompt = remove_extra_bos(prompt)
        return prompt

    prompt = make_prompt(messages)

    # Handle truncation
    if shared.tokenizer is not None:
        max_length = get_max_prompt_length(state)
        encoded_length = get_encoded_length(prompt)
        while len(messages) > 0 and encoded_length > max_length:

            # Remove old message, save system message
            if len(messages) > 2 and messages[0]['role'] == 'system':
                messages.pop(1)

            # Remove old message when no system message is present
            elif len(messages) > 1 and messages[0]['role'] != 'system':
                messages.pop(0)

            # Resort to truncating the user input
            else:

                user_message = messages[-1]['content']

                # Bisect the truncation point
                left, right = 0, len(user_message) - 1

                while right - left > 1:
                    mid = (left + right) // 2

                    messages[-1]['content'] = user_message[:mid]
                    prompt = make_prompt(messages)
                    encoded_length = get_encoded_length(prompt)

                    if encoded_length <= max_length:
                        left = mid
                    else:
                        right = mid

                messages[-1]['content'] = user_message[:left]
                prompt = make_prompt(messages)
                encoded_length = get_encoded_length(prompt)
                if encoded_length > max_length:
                    logger.error(f"Failed to build the chat prompt. The input is too long for the available context length.\n\nTruncation length: {state['truncation_length']}\nmax_new_tokens: {state['max_new_tokens']} (is it too high?)\nAvailable context length: {max_length}\n")
                    raise ValueError
                else:
                    logger.warning(f"The input has been truncated. Context length: {state['truncation_length']}, max_new_tokens: {state['max_new_tokens']}, available context length: {max_length}.")
                    break

            prompt = make_prompt(messages)
            encoded_length = get_encoded_length(prompt)

    if also_return_rows:
        return prompt, [message['content'] for message in messages]
    else:
        return prompt


def get_stopping_strings(state):
    stopping_strings = []
    renderers = []

    if state['mode'] in ['instruct', 'chat-instruct']:
        template = jinja_env.from_string(state['instruction_template_str'])
        renderer = partial(template.render, add_generation_prompt=False)
        renderers.append(renderer)

    if state['mode'] in ['chat', 'chat-instruct']:
        template = jinja_env.from_string(state['chat_template_str'])
        renderer = partial(template.render, add_generation_prompt=False, name1=state['name1'], name2=state['name2'])
        renderers.append(renderer)

    for renderer in renderers:
        prefix_bot, suffix_bot = get_generation_prompt(renderer, impersonate=False)
        prefix_user, suffix_user = get_generation_prompt(renderer, impersonate=True)

        stopping_strings += [
            suffix_user + prefix_bot,
            suffix_user + prefix_user,
            suffix_bot + prefix_bot,
            suffix_bot + prefix_user,
        ]

    # Try to find the EOT token
    for item in stopping_strings.copy():
        item = item.strip()
        if item.startswith("<") and ">" in item:
            stopping_strings.append(item.split(">")[0] + ">")
        elif item.startswith("[") and "]" in item:
            stopping_strings.append(item.split("]")[0] + "]")

    if 'stopping_strings' in state and isinstance(state['stopping_strings'], list):
        stopping_strings += state.pop('stopping_strings')

    # Remove redundant items that start with another item
    result = [item for item in stopping_strings if not any(item.startswith(other) and item != other for other in stopping_strings)]
    result = list(set(result))

    if shared.args.verbose:
        logger.info("STOPPING_STRINGS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(result)
        print()

    return result


def chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    output = copy.deepcopy(history)
    output = apply_extensions('history', output)
    state = apply_extensions('state', state)

    visible_text = None
    stopping_strings = get_stopping_strings(state)
    is_stream = state['stream']

    # Allow for continuing after tool use
    last_reply = ["", ""]
    # Prepare the input
    if not (regenerate or _continue):
        visible_text = html.escape(text)

        # Apply extensions
        text, visible_text = apply_extensions('chat_input', text, visible_text, state)
        text = apply_extensions('input', text, state, is_chat=True)

        output['internal'].append([text, ''])
        output['visible'].append([visible_text, ''])

        # *Is typing...*
        if loading_message:
            yield {
                'visible': output['visible'][:-1] + [[output['visible'][-1][0], shared.processing_message]],
                'internal': output['internal']
            }
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        if regenerate:
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, shared.processing_message]],
                    'internal': output['internal'][:-1] + [[text, '']]
                }
        elif _continue:
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
                    'internal': output['internal']
                }

    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': output if _continue else {k: v[:-1] for k, v in output.items()}
    }
    prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # This loop is added to allow the model to continue after a tool call, so that it can interpret the result.
    tool_used = False
    continue_generation = True
    consecutive_tool_uses = 0
    while continue_generation:
        continue_generation = False

        # When in the mode which requires clicking "continue", check for tool calls in previous message which haven't been executed
        if _continue and state['confirm_tool_use']:
            print("Executing tool call (pre-generate)")
            tool_calls = extract_tool_calls_to_be_executed(last_reply[0], state['tools'])
            #kwargs['last_assistant_response'] = '' # The message prior to the tool call?
            #kwargs['tool_calls'] = tool_calls
            #kwargs['tool_results'] = []
            for tool_call in tool_calls:
                tool_result = execute_tool_call(tool_call)
                #kwargs['tool_results'].append(tool_result)
                output['internal'][-1][1] += json.dumps(tool_result)
                output['visible'][-1][1] += f"\n\nTOOL RESULT: ```\n{tool_result['content']}\n```\n\n"
            yield output

            # Without this, the message keeps getting deleted over and over
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]

            # We've already added the tool calls and results to the output, so remove them from the list
            # Hence removing the kwargs settings above
            
            # After all tool calls (or max tool uses)
            if consecutive_tool_uses < state['max_consecutive_tool_uses']:
                print("Previous prompt:")
                print(prompt)
                new_prompt = generate_chat_prompt(text, state, **kwargs)
                print("New prompt:")
                print(new_prompt) # Testing
                #print("Continuing generation after tool use...")
                #continue_generation = True
                #tool_used = True
                prompt = new_prompt
            else:
                print("Max consecutive tool uses reached. Stopping generation.")
                break

        # Generate
        reply = None
        for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)):

            # Extract the reply
            if state['mode'] in ['chat', 'chat-instruct']:
                visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply + '▍')
            else:
                visible_reply = reply + '▍'

            visible_reply = html.escape(visible_reply)

            if shared.stop_everything:
                if output['visible'][-1][1].endswith('▍'):
                    output['visible'][-1][1] = output['visible'][-1][1][:-1]

                output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
                yield output
                return

            if _continue or tool_used:
                output['internal'][-1] = [text, last_reply[0] + reply]
                output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
                if is_stream:
                    yield output
            elif not (j == 0 and visible_reply.strip() == ''):
                output['internal'][-1] = [text, reply.lstrip(' ')]
                output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
                if is_stream:
                    yield output

        if output['visible'][-1][1].endswith('▍'):
            output['visible'][-1][1] = output['visible'][-1][1][:-1]

        # Displaying the reply for debugging
        print("Reply:")
        print(reply)
        # Detect whether any tool calls were made by the model. Only use the last part of the reponse
        modified_message, tool_calls = process_tool_calls(reply.lstrip(' '), state['tools'])

        if len(tool_calls) > 0:
            print("Tool calls detected:")
            print(tool_calls)
            # Use chat/instruct template to modify the message to include the tool call(s) and response(s)
            # Delete the raw tool call
            output['internal'][-1][1] = last_reply[0] + modified_message
            output['visible'][-1][1] = last_reply[1] + modified_message
            kwargs['last_assistant_response'] = last_reply[0] + modified_message
            kwargs['tool_calls'] = tool_calls
            kwargs['tool_results'] = []
            # Add tool call to visible results but not internal results
            for tool_call in tool_calls:
                output['internal'][-1][1] += json.dumps(tool_call)
                output['visible'][-1][1] += f"\n\nTOOL CALL: ```\n{json.dumps(tool_call, indent=4)}\n```"
                tool_call_type = tool_call['type']
                if tool_call_type in tool_call.keys():
                    tool_call_details = tool_call[tool_call_type]
                    tool_call_params = None
                    if 'parameters' in tool_call_details:
                        tool_call_params = tool_call_details['parameters']
                    elif 'arguments' in tool_call_details:
                        tool_call_params = tool_call_details['arguments']
                    if tool_call_params is not None and 'code' in tool_call_params.keys():
                        output['visible'][-1][1] += f"\n\nTOOL CODE:\n<pre><code>{tool_call_params['code']}</code></pre>"
            yield output
            
            # If this setting is enabled, wait for confirmation (by pressing "continue") before executing the tool call
            if state['confirm_tool_use']:
                output['visible'][-1][1] += "\n(Waiting for confirmation, press continue to run...)"
                yield output
                return # Stop generation until clicking "continue"

            else:
                print("Executing tool call (post-generate)")
                # Get tool results
                for tool_call in tool_calls:
                    tool_result = execute_tool_call(tool_call)
                    kwargs['tool_results'].append(tool_result)
                    output['internal'][-1][1] += json.dumps(tool_result)
                    output['visible'][-1][1] += f"\n\nTOOL RESULT: ```\n{tool_result['content']}\n```\n\n"
                    consecutive_tool_uses += 1
                    if consecutive_tool_uses >= state['max_consecutive_tool_uses']:
                        print("Max consecutive tool uses reached.")
                        break
                yield output

                # Without this, the message keeps getting deleted over and over
                last_reply = [output['internal'][-1][1], output['visible'][-1][1]]

                # After all tool calls (or max tool uses)
                if consecutive_tool_uses < state['max_consecutive_tool_uses']:
                    # Displaying the prompt for debugging, as it can be different depending on how the model uses special tokens to format tool calls
                    print("Previous prompt:")
                    print(prompt)
                    new_prompt = generate_chat_prompt(text, state, **kwargs)
                    print("New prompt:")
                    print(new_prompt) # Testing
                    print("Continuing generation after tool use...")
                    continue_generation = True
                    tool_used = True
                    prompt = new_prompt
                else:
                    break

    output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
    yield output


def impersonate_wrapper(text, state):
    static_output = chat_html_wrapper(state['history'], state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    prompt = generate_chat_prompt('', state, impersonate=True)
    stopping_strings = get_stopping_strings(state)

    yield text + '...', static_output
    reply = None
    for reply in generate_reply(prompt + text, state, stopping_strings=stopping_strings, is_chat=True):
        yield (text + reply).lstrip(' '), static_output
        if shared.stop_everything:
            return


def generate_chat_reply(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    if regenerate or _continue:
        text = ''
        if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
            yield history
            return

    show_after = html.escape(state["show_after"]) if state["show_after"] else None
    for history in chatbot_wrapper(text, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message, for_ui=for_ui):
        if show_after:
            after = history["visible"][-1][1].partition(show_after)[2] or "*Is thinking...*"
            yield {
                'internal': history['internal'],
                'visible': history['visible'][:-1] + [[history['visible'][-1][0], after]]
            }
        else:
            yield history


def character_is_loaded(state, raise_exception=False):
    if state['mode'] in ['chat', 'chat-instruct'] and state['name2'] == '':
        logger.error('It looks like no character is loaded. Please load one under Parameters > Character.')
        if raise_exception:
            raise ValueError

        return False
    else:
        return True


def generate_chat_reply_wrapper(text, state, regenerate=False, _continue=False):
    '''
    Same as above but returns HTML for the UI
    '''

    if not character_is_loaded(state):
        return

    if state['start_with'] != '' and not _continue:
        if regenerate:
            text, state['history'] = remove_last_message(state['history'])
            regenerate = False

        _continue = True
        send_dummy_message(text, state)
        send_dummy_reply(state['start_with'], state)

    history = state['history']
    for i, history in enumerate(generate_chat_reply(text, state, regenerate, _continue, loading_message=True, for_ui=True)):
        yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu']), history

    save_history(history, state['unique_id'], state['character_menu'], state['mode'])


def remove_last_message(history):
    if len(history['visible']) > 0 and history['internal'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
        last = history['visible'].pop()
        history['internal'].pop()
    else:
        last = ['', '']

    return html.unescape(last[0]), history


def send_last_reply_to_input(history):
    if len(history['visible']) > 0:
        return html.unescape(history['visible'][-1][1])
    else:
        return ''


def replace_last_reply(text, state):
    history = state['history']

    if len(text.strip()) == 0:
        return history
    elif len(history['visible']) > 0:
        history['visible'][-1][1] = html.escape(text)
        history['internal'][-1][1] = apply_extensions('input', text, state, is_chat=True)

    return history


def send_dummy_message(text, state):
    history = state['history']
    history['visible'].append([html.escape(text), ''])
    history['internal'].append([apply_extensions('input', text, state, is_chat=True), ''])
    return history


def send_dummy_reply(text, state):
    history = state['history']
    if len(history['visible']) > 0 and not history['visible'][-1][1] == '':
        history['visible'].append(['', ''])
        history['internal'].append(['', ''])

    history['visible'][-1][1] = html.escape(text)
    history['internal'][-1][1] = apply_extensions('input', text, state, is_chat=True)
    return history


def redraw_html(history, name1, name2, mode, style, character, reset_cache=False):
    return chat_html_wrapper(history, name1, name2, mode, style, character, reset_cache=reset_cache)


def start_new_chat(state):
    mode = state['mode']
    history = {'internal': [], 'visible': []}

    if mode != 'instruct':
        greeting = replace_character_names(state['greeting'], state['name1'], state['name2'])
        if greeting != '':
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
            history['visible'] += [['', apply_extensions('output', html.escape(greeting), state, is_chat=True)]]

    unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    save_history(history, unique_id, state['character_menu'], state['mode'])

    return history


def get_history_file_path(unique_id, character, mode):
    if mode == 'instruct':
        p = Path(f'logs/instruct/{unique_id}.json')
    else:
        p = Path(f'logs/chat/{character}/{unique_id}.json')

    return p


def save_history(history, unique_id, character, mode):
    if shared.args.multi_user:
        return

    p = get_history_file_path(unique_id, character, mode)
    if not p.parent.is_dir():
        p.parent.mkdir(parents=True)

    with open(p, 'w', encoding='utf-8') as f:
        f.write(json.dumps(history, indent=4, ensure_ascii=False))


def rename_history(old_id, new_id, character, mode):
    if shared.args.multi_user:
        return

    old_p = get_history_file_path(old_id, character, mode)
    new_p = get_history_file_path(new_id, character, mode)
    if new_p.parent != old_p.parent:
        logger.error(f"The following path is not allowed: \"{new_p}\".")
    elif new_p == old_p:
        logger.info("The provided path is identical to the old one.")
    elif new_p.exists():
        logger.error(f"The new path already exists and will not be overwritten: \"{new_p}\".")
    else:
        logger.info(f"Renaming \"{old_p}\" to \"{new_p}\"")
        old_p.rename(new_p)


def get_paths(state):
    if state['mode'] == 'instruct':
        return Path('logs/instruct').glob('*.json')
    else:
        character = state['character_menu']

        # Handle obsolete filenames and paths
        old_p = Path(f'logs/{character}_persistent.json')
        new_p = Path(f'logs/persistent_{character}.json')
        if old_p.exists():
            logger.warning(f"Renaming \"{old_p}\" to \"{new_p}\"")
            old_p.rename(new_p)

        if new_p.exists():
            unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            p = get_history_file_path(unique_id, character, state['mode'])
            logger.warning(f"Moving \"{new_p}\" to \"{p}\"")
            p.parent.mkdir(exist_ok=True)
            new_p.rename(p)

        return Path(f'logs/chat/{character}').glob('*.json')


def find_all_histories(state):
    if shared.args.multi_user:
        return ['']

    paths = get_paths(state)
    histories = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
    return [path.stem for path in histories]


def find_all_histories_with_first_prompts(state):
    if shared.args.multi_user:
        return []

    paths = get_paths(state)
    histories = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)

    result = []
    for i, path in enumerate(histories):
        filename = path.stem
        file_content = ""
        with open(path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        if state['search_chat'] and state['search_chat'] not in file_content:
            continue

        data = json.loads(file_content)
        if re.match(r'^[0-9]{8}-[0-9]{2}-[0-9]{2}-[0-9]{2}$', filename):
            first_prompt = ""
            if data and 'visible' in data and len(data['visible']) > 0:
                if data['internal'][0][0] == '<|BEGIN-VISIBLE-CHAT|>':
                    if len(data['visible']) > 1:
                        first_prompt = html.unescape(data['visible'][1][0])
                    elif i == 0:
                        first_prompt = "New chat"
                else:
                    first_prompt = html.unescape(data['visible'][0][0])
            elif i == 0:
                first_prompt = "New chat"
        else:
            first_prompt = filename

        first_prompt = first_prompt.strip()

        # Truncate the first prompt if it's longer than 30 characters
        if len(first_prompt) > 30:
            first_prompt = first_prompt[:30 - 3] + '...'

        result.append((first_prompt, filename))

    return result


def load_latest_history(state):
    '''
    Loads the latest history for the given character in chat or chat-instruct
    mode, or the latest instruct history for instruct mode.
    '''

    if shared.args.multi_user:
        return start_new_chat(state)

    histories = find_all_histories(state)

    if len(histories) > 0:
        history = load_history(histories[0], state['character_menu'], state['mode'])
    else:
        history = start_new_chat(state)

    return history


def load_history_after_deletion(state, idx):
    '''
    Loads the latest history for the given character in chat or chat-instruct
    mode, or the latest instruct history for instruct mode.
    '''

    if shared.args.multi_user:
        return start_new_chat(state)

    histories = find_all_histories_with_first_prompts(state)
    idx = min(int(idx), len(histories) - 1)
    idx = max(0, idx)

    if len(histories) > 0:
        history = load_history(histories[idx][1], state['character_menu'], state['mode'])
    else:
        history = start_new_chat(state)
        histories = find_all_histories_with_first_prompts(state)

    return history, gr.update(choices=histories, value=histories[idx][1])


def update_character_menu_after_deletion(idx):
    characters = utils.get_available_characters()
    idx = min(int(idx), len(characters) - 1)
    idx = max(0, idx)
    return gr.update(choices=characters, value=characters[idx])


def update_tool_preset_menu_after_deletion(idx):
    tool_presets = utils.get_available_tool_presets()
    idx = min(int(idx), len(tool_presets) - 1)
    idx = max(0, idx)
    return gr.update(choices=tool_presets, value=tool_presets[idx])


def update_tools_after_deletion(tool_name, state):
    # Remove tool from state, if present
    if 'tools' in state:
        for i, tool in enumerate(state['tools']):
            if tool['name'] == tool_name:
                del state['tools'][i]
                return


def load_history(unique_id, character, mode):
    p = get_history_file_path(unique_id, character, mode)

    f = json.loads(open(p, 'rb').read())
    if 'internal' in f and 'visible' in f:
        history = f
    else:
        history = {
            'internal': f['data'],
            'visible': f['data_visible']
        }

    return history


def load_history_json(file, history):
    try:
        file = file.decode('utf-8')
        f = json.loads(file)
        if 'internal' in f and 'visible' in f:
            history = f
        else:
            history = {
                'internal': f['data'],
                'visible': f['data_visible']
            }

        return history
    except:
        return history


def delete_history(unique_id, character, mode):
    p = get_history_file_path(unique_id, character, mode)
    delete_file(p)


def replace_character_names(text, name1, name2):
    text = text.replace('{{user}}', name1).replace('{{char}}', name2)
    return text.replace('<USER>', name1).replace('<BOT>', name2)


def generate_pfp_cache(character):
    cache_folder = Path(shared.args.disk_cache_dir)
    if not cache_folder.exists():
        cache_folder.mkdir()

    for path in [Path(f"characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
        if path.exists():
            original_img = Image.open(path)
            original_img.save(Path(f'{cache_folder}/pfp_character.png'), format='PNG')

            thumb = make_thumbnail(original_img)
            thumb.save(Path(f'{cache_folder}/pfp_character_thumb.png'), format='PNG')

            return thumb

    return None


def load_character(character, name1, name2):
    context = greeting = ""
    greeting_field = 'greeting'
    picture = None

    filepath = None
    for extension in ["yml", "yaml", "json"]:
        filepath = Path(f'characters/{character}.{extension}')
        if filepath.exists():
            break

    if filepath is None or not filepath.exists():
        logger.error(f"Could not find the character \"{character}\" inside characters/. No character has been loaded.")
        raise ValueError

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)
    cache_folder = Path(shared.args.disk_cache_dir)

    for path in [Path(f"{cache_folder}/pfp_character.png"), Path(f"{cache_folder}/pfp_character_thumb.png")]:
        if path.exists():
            path.unlink()

    picture = generate_pfp_cache(character)

    # Finding the bot's name
    for k in ['name', 'bot', '<|bot|>', 'char_name']:
        if k in data and data[k] != '':
            name2 = data[k]
            break

    # Find the user name (if any)
    for k in ['your_name', 'user', '<|user|>']:
        if k in data and data[k] != '':
            name1 = data[k]
            break

    if 'context' in data:
        context = data['context'].strip()
    elif "char_persona" in data:
        context = build_pygmalion_style_context(data)
        greeting_field = 'char_greeting'

    greeting = data.get(greeting_field, greeting)
    return name1, name2, picture, greeting, context


def load_instruction_template(template):
    if template == 'None':
        return ''

    for filepath in [Path(f'instruction-templates/{template}.yaml'), Path('instruction-templates/Alpaca.yaml')]:
        if filepath.exists():
            break
    else:
        return ''

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    data = yaml.safe_load(file_contents)
    if 'instruction_template' in data:
        return data['instruction_template']
    else:
        return jinja_template_from_old_format(data)


# Assume JSON but idk
def load_tool(tool_name):
    filepath = None
    for extension in ["yml", "yaml", "json"]:
        filepath = Path(f'tools/{tool_name}.{extension}')
        if filepath.exists():
            break

    if filepath is None or not filepath.exists():
        logger.error(f"Could not find the tool \"{tool_name}\" inside tools/. No tool has been loaded.")
        raise ValueError

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)
    
    tool_type = 'function'
    tool_description = ""
    tool_parameters = {}
    tool_action = ""
    if 'name' in data:
        tool_name = data['name'] # If this is different from the filename, it could cause issues...
    if 'type' in data:
        tool_type = data['type']
    if 'description' in data:
        tool_description = data['description']
    if 'parameters' in data:
        tool_parameters = data['parameters']
    if 'action' in data:
        tool_action = data['action']
    
    return tool_name, tool_type, tool_description, tool_parameters, tool_action


def load_tool_preset(preset):
    filepath = Path(f'tools/presets/{preset}.txt')
    if not filepath.exists():
        return
    file_contents = open(filepath, 'r', encoding='utf-8').read()
    return file_contents


@functools.cache
def load_character_memoized(character, name1, name2):
    return load_character(character, name1, name2)


@functools.cache
def load_instruction_template_memoized(template):
    return load_instruction_template(template)


def upload_character(file, img, tavern=False):
    decoded_file = file if isinstance(file, str) else file.decode('utf-8')
    try:
        data = json.loads(decoded_file)
    except:
        data = yaml.safe_load(decoded_file)

    if 'char_name' in data:
        name = data['char_name']
        greeting = data['char_greeting']
        context = build_pygmalion_style_context(data)
        yaml_data = generate_character_yaml(name, greeting, context)
    else:
        name = data['name']
        yaml_data = generate_character_yaml(data['name'], data['greeting'], data['context'])

    outfile_name = name
    i = 1
    while Path(f'characters/{outfile_name}.yaml').exists():
        outfile_name = f'{name}_{i:03d}'
        i += 1

    with open(Path(f'characters/{outfile_name}.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_data)

    if img is not None:
        img.save(Path(f'characters/{outfile_name}.png'))

    logger.info(f'New character saved to "characters/{outfile_name}.yaml".')
    return gr.update(value=outfile_name, choices=get_available_characters())


def build_pygmalion_style_context(data):
    context = ""
    if 'char_persona' in data and data['char_persona'] != '':
        context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"

    if 'world_scenario' in data and data['world_scenario'] != '':
        context += f"Scenario: {data['world_scenario']}\n"

    if 'example_dialogue' in data and data['example_dialogue'] != '':
        context += f"{data['example_dialogue'].strip()}\n"

    context = f"{context.strip()}\n"
    return context


def upload_tavern_character(img, _json):
    _json = {'char_name': _json['name'], 'char_persona': _json['description'], 'char_greeting': _json['first_mes'], 'example_dialogue': _json['mes_example'], 'world_scenario': _json['scenario']}
    return upload_character(json.dumps(_json), img, tavern=True)


def check_tavern_character(img):
    if "chara" not in img.info:
        return "Not a TavernAI card", None, None, gr.update(interactive=False)

    decoded_string = base64.b64decode(img.info['chara']).replace(b'\\r\\n', b'\\n')
    _json = json.loads(decoded_string)
    if "data" in _json:
        _json = _json["data"]

    return _json['name'], _json['description'], _json, gr.update(interactive=True)


def upload_your_profile_picture(img):
    cache_folder = Path(shared.args.disk_cache_dir)
    if not cache_folder.exists():
        cache_folder.mkdir()

    if img is None:
        if Path(f"{cache_folder}/pfp_me.png").exists():
            Path(f"{cache_folder}/pfp_me.png").unlink()
    else:
        img = make_thumbnail(img)
        img.save(Path(f'{cache_folder}/pfp_me.png'))
        logger.info(f'Profile picture saved to "{cache_folder}/pfp_me.png"')


def generate_character_yaml(name, greeting, context):
    data = {
        'name': name,
        'greeting': greeting,
        'context': context,
    }

    data = {k: v for k, v in data.items() if v}  # Strip falsy
    return yaml.dump(data, sort_keys=False, width=float("inf"))


def generate_instruction_template_yaml(instruction_template):
    data = {
        'instruction_template': instruction_template
    }

    return my_yaml_output(data)


def save_character(name, greeting, context, picture, filename):
    if filename == "":
        logger.error("The filename is empty, so the character will not be saved.")
        return

    data = generate_character_yaml(name, greeting, context)
    filepath = Path(f'characters/{filename}.yaml')
    save_file(filepath, data)
    path_to_img = Path(f'characters/{filename}.png')
    if picture is not None:
        picture.save(path_to_img)
        logger.info(f'Saved {path_to_img}.')


def delete_character(name, instruct=False):
    for extension in ["yml", "yaml", "json"]:
        delete_file(Path(f'characters/{name}.{extension}'))

    delete_file(Path(f'characters/{name}.png'))


# Assuming name == filename for now..
def generate_tool_json(name, type, description, parameters, action):
    try:
        data = {
            'name': name,
            'type': type,
            'description': description,
            'parameters': json.loads(parameters),
            'action': action
        }
    except json.decoder.JSONDecodeError:
        logger.error("Failed to parse tool parameters JSON")
        gr.Warning("Failed to parse tool parameters JSON")
        return
    
    return json.dumps(data, indent=4, sort_keys=True)


# This isn't a good format tbh
def save_tool_preset(filename, selected_tools):
    if filename == "":
        logger.error("The filename is empty, so the tool preset will not be saved.")
        return
    data = ",".join(selected_tools)
    # Check whether the tools/presets directory exists, and if not, create it
    if not Path('tools/presets').exists():
        Path('tools/presets').mkdir(parents=True)
    filepath = Path(f'tools/presets/{filename}.txt')
    save_file(filepath, data)


def delete_tool_preset(name):
    delete_file(Path(f'tools/presets/{name}.txt'))


def save_tool(filename, type, description, parameters, action):
    if filename == "":
        logger.error("The filename is empty, so the tool will not be saved.")
        return

    data = generate_tool_json(filename, type, description, parameters, action)
    if data is not None:
        filepath = Path(f'tools/{filename}.json')
        save_file(filepath, data)
        return True
    return False


def delete_tool(name):
    for extension in ["yml", "yaml", "json"]:
        delete_file(Path(f'tools/{name}.{extension}'))


def jinja_template_from_old_format(params, verbose=False):
    MASTER_TEMPLATE = """
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
{%- if not ns.found -%}
    {{- '<|PRE-SYSTEM|>' + '<|SYSTEM-MESSAGE|>' + '<|POST-SYSTEM|>' -}}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        {{- '<|PRE-SYSTEM|>' + message['content'] + '<|POST-SYSTEM|>' -}}
    {%- else -%}
        {%- if message['role'] == 'user' -%}
            {{-'<|PRE-USER|>' + message['content'] + '<|POST-USER|>'-}}
        {%- else -%}
            {{-'<|PRE-ASSISTANT|>' + message['content'] + '<|POST-ASSISTANT|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{-'<|PRE-ASSISTANT-GENERATE|>'-}}
{%- endif -%}
"""

    if 'context' in params and '<|system-message|>' in params['context']:
        pre_system = params['context'].split('<|system-message|>')[0]
        post_system = params['context'].split('<|system-message|>')[1]
    else:
        pre_system = ''
        post_system = ''

    pre_user = params['turn_template'].split('<|user-message|>')[0].replace('<|user|>', params['user'])
    post_user = params['turn_template'].split('<|user-message|>')[1].split('<|bot|>')[0]

    pre_assistant = '<|bot|>' + params['turn_template'].split('<|bot-message|>')[0].split('<|bot|>')[1]
    pre_assistant = pre_assistant.replace('<|bot|>', params['bot'])
    post_assistant = params['turn_template'].split('<|bot-message|>')[1]

    def preprocess(string):
        return string.replace('\n', '\\n').replace('\'', '\\\'')

    pre_system = preprocess(pre_system)
    post_system = preprocess(post_system)
    pre_user = preprocess(pre_user)
    post_user = preprocess(post_user)
    pre_assistant = preprocess(pre_assistant)
    post_assistant = preprocess(post_assistant)

    if verbose:
        print(
            '\n',
            repr(pre_system) + '\n',
            repr(post_system) + '\n',
            repr(pre_user) + '\n',
            repr(post_user) + '\n',
            repr(pre_assistant) + '\n',
            repr(post_assistant) + '\n',
        )

    result = MASTER_TEMPLATE
    if 'system_message' in params:
        result = result.replace('<|SYSTEM-MESSAGE|>', preprocess(params['system_message']))
    else:
        result = result.replace('<|SYSTEM-MESSAGE|>', '')

    result = result.replace('<|PRE-SYSTEM|>', pre_system)
    result = result.replace('<|POST-SYSTEM|>', post_system)
    result = result.replace('<|PRE-USER|>', pre_user)
    result = result.replace('<|POST-USER|>', post_user)
    result = result.replace('<|PRE-ASSISTANT|>', pre_assistant)
    result = result.replace('<|PRE-ASSISTANT-GENERATE|>', pre_assistant.rstrip(' '))
    result = result.replace('<|POST-ASSISTANT|>', post_assistant)

    result = result.strip()

    return result


def my_yaml_output(data):
    '''
    pyyaml is very inconsistent with multiline strings.
    for simple instruction template outputs, this is enough.
    '''
    result = ""
    for k in data:
        result += k + ": |-\n"
        for line in data[k].splitlines():
            result += "  " + line.rstrip(' ') + "\n"

    return result


def handle_replace_last_reply_click(text, state):
    history = replace_last_reply(text, state)
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html, ""]


def handle_send_dummy_message_click(text, state):
    history = send_dummy_message(text, state)
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html, ""]


def handle_send_dummy_reply_click(text, state):
    history = send_dummy_reply(text, state)
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html, ""]


def handle_remove_last_click(state):
    last_input, history = remove_last_message(state['history'])
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html, last_input]


def handle_unique_id_select(state):
    history = load_history(state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    return [history, html]


def handle_start_new_chat_click(state):
    history = start_new_chat(state)
    histories = find_all_histories_with_first_prompts(state)
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [history, html, past_chats_update]


def handle_delete_chat_confirm_click(state):
    index = str(find_all_histories(state).index(state['unique_id']))
    delete_history(state['unique_id'], state['character_menu'], state['mode'])
    history, unique_id = load_history_after_deletion(state, index)
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    return [
        history,
        html,
        unique_id,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False)
    ]


def handle_branch_chat_click(state):
    history = state['history']
    new_unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    save_history(history, new_unique_id, state['character_menu'], state['mode'])

    histories = find_all_histories_with_first_prompts(state)
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    past_chats_update = gr.update(choices=histories, value=new_unique_id)

    return [history, html, past_chats_update]


def handle_rename_chat_click():
    return [
        gr.update(value="My New Chat"),
        gr.update(visible=True),
    ]


def handle_rename_chat_confirm(rename_to, state):
    rename_history(state['unique_id'], rename_to, state['character_menu'], state['mode'])
    histories = find_all_histories_with_first_prompts(state)

    return [
        gr.update(choices=histories, value=rename_to),
        gr.update(visible=False),
    ]


def handle_search_chat_change(state):
    histories = find_all_histories_with_first_prompts(state)
    return gr.update(choices=histories)


def handle_upload_chat_history(load_chat_history, state):
    history = start_new_chat(state)
    history = load_history_json(load_chat_history, history)
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    histories = find_all_histories_with_first_prompts(state)

    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [
        history,
        html,
        past_chats_update
    ]


def handle_character_menu_change(state):
    name1, name2, picture, greeting, context = load_character(state['character_menu'], state['name1'], state['name2'])

    state['name1'] = name1
    state['name2'] = name2
    state['character_picture'] = picture
    state['greeting'] = greeting
    state['context'] = context

    history = load_latest_history(state)
    histories = find_all_histories_with_first_prompts(state)
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [
        history,
        html,
        name1,
        name2,
        picture,
        greeting,
        context,
        past_chats_update,
    ]


def handle_mode_change(state):
    history = load_latest_history(state)
    histories = find_all_histories_with_first_prompts(state)
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=histories[0][1])
    else:
        past_chats_update = gr.update(choices=histories)

    return [
        history,
        html,
        gr.update(visible=state['mode'] != 'instruct'),
        gr.update(visible=state['mode'] == 'chat-instruct'),
        past_chats_update
    ]


def handle_save_character_click(name2):
    return [
        name2,
        gr.update(visible=True)
    ]


def handle_load_template_click(instruction_template):
    output = load_instruction_template(instruction_template)
    return [
        output,
        "Select template to load..."
    ]


def handle_save_template_click(instruction_template_str):
    contents = generate_instruction_template_yaml(instruction_template_str)
    return [
        "My Template.yaml",
        "instruction-templates/",
        contents,
        gr.update(visible=True)
    ]


def handle_delete_template_click(template):
    return [
        f"{template}.yaml",
        "instruction-templates/",
        gr.update(visible=False)
    ]


def handle_your_picture_change(picture, state):
    upload_your_profile_picture(picture)
    html = redraw_html(state['history'], state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'], reset_cache=True)

    return html


def handle_save_tool_preset_click(tool_preset_menu):
    return [
        f"{tool_preset_menu}", # Filename
        gr.update(visible=True) # Saver
    ]


def handle_tool_preset_change(tool_preset_menu):
    tool_preset = load_tool_preset(tool_preset_menu)
    selected_tools = []
    if tool_preset is not None:
        selected_tools = tool_preset.split(',')
    return gr.update(value=selected_tools)


def handle_tool_change(tool_checkbox_group, tools):
    # TODO: Define this when state is initialized to avoid needing to do this
    if tools is None:
        tools = []
        
    # Determine which element was toggled...
    tools_in_state = set([tool['name'] for tool in tools])
    tools_selected = set(tool_checkbox_group)
    #print("Tools in list:", tools_in_state)
    #print("Tools selected:", tools_selected)

    selected_tool = None
    tools_to_add = []
    tools_to_remove = []

    if len(tools_selected - tools_in_state) > 0:
        # Tool(s) selected
        tools_to_add = sorted(list(tools_selected - tools_in_state))
    if len(tools_in_state - tools_selected) > 0:
        # Tool(s) deselected
        tools_to_remove = sorted(list(tools_in_state - tools_selected))
    if tools_in_state == tools_selected:
        # No tool selected (e.g. the selected tool was deleted or the "None" preset was chosen)
        # Try to load the first tool
        available_tools = utils.get_available_tools()
        if len(available_tools) > 0:
            selected_tool = available_tools[0]
        else:
            return "", "function", "", "{}", "", tools

    # Remove deselected tools
    for tool_name in tools_to_remove:
        index = None
        for idx, tool in enumerate(tools):
            if tool['name'] == tool_name:
                index = idx
                break
        if index is not None:
            del tools[index]
            print("Removed tool:", tool_name)
        else:
            print("Tool not found in list?", tool_name)

    # Then add selected tools
    for selected_tool in tools_to_add:
        # Load tool file based on name, then add/update in the list of tools
        tool_name, tool_type, tool_description, tool_parameters, tool_action = load_tool(selected_tool)

        tool = {
            'name': tool_name,
            'type': tool_type,
            'description': tool_description,
            'parameters': tool_parameters,
            'action': tool_action
        }
        tools.append(tool)
        # Evaluate tool action
        valid = define_tool_action(tool)
        if not valid:
            gr.Warning("Tool action definition failed! Check your code.")
        print("Added tool:", tool_name)
    
    # Account for situation where tool(s) were removed but not added, but still need to display the selection
    if selected_tool is None:
        selected_tool = tool_name # Use the last removed tool
        tool_name, tool_type, tool_description, tool_parameters, tool_action = load_tool(selected_tool)

    print("Current tools:", tools)
    
    return tool_name, tool_type, tool_description, json.dumps(tool_parameters, indent=4, sort_keys=True), tool_action, tools


def handle_save_tool_click(tool_name):
    return [
        f"{tool_name}", # Filename (.json)
        gr.update(visible=True) # Saver
    ]


def handle_save_tool(tool_name, tool_type, tool_description, tool_parameters, tool_action, tools):
    # TODO: Define this when state is initialized to avoid needing to do this
    if tools is None:
        tools = []
    
    tool_exists = False
    tools_in_state = set([tool['name'] for tool in tools])
    tools_accounted_for = set()
    tool_options = utils.get_available_tools()
    for tool in tool_options:
        if tool in tools_in_state:
            tools_accounted_for.add(tool)
        if tool == tool_name:
            # Existing Tool
            tool_exists = True

    if tools_accounted_for != tools_in_state:
        # Tool was renamed!
        # should be a set with one element
        possible_previous_tool_names = tools_in_state - tools_accounted_for
        if len(possible_previous_tool_names) > 1:
            print("More than one renamed tool? How is that possible?")
        # Just assume it's the first one...
        previous_tool_name = list(possible_previous_tool_names)[0]
        tool_exists = True
        # Rename in list
        for i, tool_option in enumerate(tool_options):
            if tool_option == previous_tool_name:
                tool_options[i] = tool_name
    else:
        previous_tool_name = tool_name

    # Change the tool content in the state
    for tool in tools:
        if tool['name'] == previous_tool_name:
            tool['name'] = tool_name
            tool['type'] = tool_type
            tool['description'] = tool_description
            tool['parameters'] = json.dumps(tool_parameters, indent=4, sort_keys=True)
            tool['action'] = tool_action
            # Re-evaluate tool action
            valid = define_tool_action(tool)
            if not valid:
                gr.Warning("Tool action definition failed! Check your code.")

    if not tool_exists:
        # New Tool (not initially added to state)
        tool_options.append(tool_name)

    return [
        gr.update(choices=sorted(tool_options)),
        gr.update(visible=False),
        tools
    ]


def handle_send_instruction_click(state):
    state['mode'] = 'instruct'
    state['history'] = {'internal': [], 'visible': []}

    output = generate_chat_prompt("Input", state)

    return output


def handle_send_chat_click(state):
    output = generate_chat_prompt("", state, _continue=True)

    return output
