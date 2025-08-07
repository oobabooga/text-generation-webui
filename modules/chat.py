import base64
import copy
import functools
import html
import json
import pprint
import re
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import yaml
from jinja2.ext import loopcontrols
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
from modules.utils import delete_file, get_available_characters, save_file
from modules.web_search import add_web_search_attachments


def strftime_now(format):
    return datetime.now().strftime(format)


def get_current_timestamp():
    """Returns the current time in 24-hour format"""
    return datetime.now().strftime('%b %d, %Y %H:%M')


def update_message_metadata(metadata_dict, role, index, **fields):
    """
    Updates or adds metadata fields for a specific message.

    Args:
        metadata_dict: The metadata dictionary
        role: The role (user, assistant, etc)
        index: The message index
        **fields: Arbitrary metadata fields to update/add
    """
    key = f"{role}_{index}"
    if key not in metadata_dict:
        metadata_dict[key] = {}

    # Update with provided fields
    for field_name, field_value in fields.items():
        metadata_dict[key][field_name] = field_value


jinja_env = ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[loopcontrols]
)
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


def get_thinking_suppression_string(template):
    """
    Determines what string needs to be added to suppress thinking mode
    by comparing template renderings with thinking enabled vs disabled.
    """

    # Render with thinking enabled
    with_thinking = template.render(
        messages=[{'role': 'user', 'content': ''}],
        builtin_tools=None,
        tools=None,
        tools_in_user_message=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    # Render with thinking disabled
    without_thinking = template.render(
        messages=[{'role': 'user', 'content': ''}],
        builtin_tools=None,
        tools=None,
        tools_in_user_message=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Find the difference (what gets added to suppress thinking)
    i = 0
    while i < min(len(with_thinking), len(without_thinking)) and with_thinking[i] == without_thinking[i]:
        i += 1

    j = 0
    while j < min(len(with_thinking), len(without_thinking)) - i and with_thinking[-1 - j] == without_thinking[-1 - j]:
        j += 1

    return without_thinking[i:len(without_thinking) - j if j else None]


def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    also_return_rows = kwargs.get('also_return_rows', False)
    history_data = kwargs.get('history', state['history'])
    history = history_data['internal']
    metadata = history_data.get('metadata', {})

    # Templates
    chat_template_str = state['chat_template_str']
    if state['mode'] != 'instruct':
        chat_template_str = replace_character_names(chat_template_str, state['name1'], state['name2'])

    instruction_template = jinja_env.from_string(state['instruction_template_str'])
    chat_template = jinja_env.from_string(chat_template_str)

    instruct_renderer = partial(
        instruction_template.render,
        builtin_tools=None,
        tools=state['tools'] if 'tools' in state else None,
        tools_in_user_message=False,
        add_generation_prompt=False,
        reasoning_effort=state['reasoning_effort']
    )

    chat_renderer = partial(
        chat_template.render,
        add_generation_prompt=False,
        name1=state['name1'],
        name2=state['name2'],
        user_bio=replace_character_names(state['user_bio'], state['name1'], state['name2']),
    )

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
    for i, entry in enumerate(reversed(history)):
        user_msg = entry[0].strip()
        assistant_msg = entry[1].strip()
        tool_msg = entry[2].strip() if len(entry) > 2 else ''

        row_idx = len(history) - i - 1

        if tool_msg:
            messages.insert(insert_pos, {"role": "tool", "content": tool_msg})

        if assistant_msg:
            # Handle GPT-OSS as a special case
            if '<|channel|>analysis<|message|>' in assistant_msg or '<|channel|>final<|message|>' in assistant_msg:

                thinking_content = ""
                final_content = ""

                # Extract analysis content if present
                if '<|channel|>analysis<|message|>' in assistant_msg:
                    # Split the message by the analysis tag to isolate the content that follows
                    parts = assistant_msg.split('<|channel|>analysis<|message|>', 1)
                    if len(parts) > 1:
                        # The content is everything after the tag
                        potential_content = parts[1]

                        # Now, find the end of this content block
                        analysis_end_tag = '<|end|>'
                        if analysis_end_tag in potential_content:
                            thinking_content = potential_content.split(analysis_end_tag, 1)[0].strip()
                        else:
                            # Fallback: if no <|end|> tag, stop at the start of the final channel if it exists
                            final_channel_tag = '<|channel|>final<|message|>'
                            if final_channel_tag in potential_content:
                                thinking_content = potential_content.split(final_channel_tag, 1)[0].strip()
                            else:
                                thinking_content = potential_content.strip()

                # Extract final content if present
                final_tag_to_find = '<|channel|>final<|message|>'
                if final_tag_to_find in assistant_msg:
                    # Split the message by the final tag to isolate the content that follows
                    parts = assistant_msg.split(final_tag_to_find, 1)
                    if len(parts) > 1:
                        # The content is everything after the tag
                        potential_content = parts[1]

                        # Now, find the end of this content block
                        final_end_tag = '<|end|>'
                        if final_end_tag in potential_content:
                            final_content = potential_content.split(final_end_tag, 1)[0].strip()
                        else:
                            final_content = potential_content.strip()

                # Insert as structured message
                msg_dict = {"role": "assistant", "content": final_content}
                if '<|channel|>analysis<|message|>' in assistant_msg:
                    msg_dict["thinking"] = thinking_content

                messages.insert(insert_pos, msg_dict)

            else:
                messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg})

        if user_msg not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            # Check for user message attachments in metadata
            user_key = f"user_{row_idx}"
            enhanced_user_msg = user_msg

            # Add attachment content if present AND if past attachments are enabled
            if (state.get('include_past_attachments', True) and user_key in metadata and "attachments" in metadata[user_key]):
                attachments_text = ""
                for attachment in metadata[user_key]["attachments"]:
                    filename = attachment.get("name", "file")
                    content = attachment.get("content", "")
                    if attachment.get("type") == "text/html" and attachment.get("url"):
                        attachments_text += f"\nName: {filename}\nURL: {attachment['url']}\nContents:\n\n=====\n{content}\n=====\n\n"
                    else:
                        attachments_text += f"\nName: {filename}\nContents:\n\n=====\n{content}\n=====\n\n"

                if attachments_text:
                    enhanced_user_msg = f"{user_msg}\n\nATTACHMENTS:\n{attachments_text}"

            messages.insert(insert_pos, {"role": "user", "content": enhanced_user_msg})

    user_input = user_input.strip()

    # Check if we have attachments even with empty input
    has_attachments = False
    if not impersonate and not _continue and len(history_data.get('metadata', {})) > 0:
        current_row_idx = len(history)
        user_key = f"user_{current_row_idx}"
        has_attachments = user_key in metadata and "attachments" in metadata[user_key]

    if (user_input or has_attachments) and not impersonate and not _continue:
        # For the current user input being processed, check if we need to add attachments
        if not impersonate and not _continue and len(history_data.get('metadata', {})) > 0:
            current_row_idx = len(history)
            user_key = f"user_{current_row_idx}"

            if user_key in metadata and "attachments" in metadata[user_key]:
                attachments_text = ""
                for attachment in metadata[user_key]["attachments"]:
                    filename = attachment.get("name", "file")
                    content = attachment.get("content", "")
                    if attachment.get("type") == "text/html" and attachment.get("url"):
                        attachments_text += f"\nName: {filename}\nURL: {attachment['url']}\nContents:\n\n=====\n{content}\n=====\n\n"
                    else:
                        attachments_text += f"\nName: {filename}\nContents:\n\n=====\n{content}\n=====\n\n"

                if attachments_text:
                    user_input = f"{user_input}\n\nATTACHMENTS:\n{attachments_text}"

        messages.append({"role": "user", "content": user_input})

    def make_prompt(messages):
        if state['mode'] == 'chat-instruct' and _continue:
            prompt = renderer(messages=messages[:-1])
        else:
            prompt = renderer(messages=messages)

        if state['mode'] == 'chat-instruct':
            outer_messages = []
            if state['custom_system_message'].strip() != '':
                outer_messages.append({"role": "system", "content": state['custom_system_message']})

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

            prompt = instruct_renderer(messages=outer_messages)
            suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
            if len(suffix) > 0:
                prompt = prompt[:-len(suffix)]
        else:
            # Handle GPT-OSS as a special case when continuing
            if _continue and '<|channel|>final<|message|>' in state['instruction_template_str']:
                last_message_to_continue = messages[-1]
                prompt = renderer(messages=messages[:-1])

                # Start the assistant turn wrapper
                assistant_reply_so_far = "<|start|>assistant"

                if 'thinking' in last_message_to_continue:
                    assistant_reply_so_far += f"<|channel|>analysis<|message|>{last_message_to_continue['thinking']}<|end|>"

                assistant_reply_so_far += f"<|channel|>final<|message|>{last_message_to_continue.get('content', '')}"

                prompt += assistant_reply_so_far

            else:
                prompt = renderer(messages=messages)
                if _continue:
                    suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
                    if len(suffix) > 0:
                        prompt = prompt[:-len(suffix)]
                else:
                    prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]

                    # Handle GPT-OSS as a special case when not continuing
                    if '<|channel|>final<|message|>' in state['instruction_template_str']:
                        if prefix.endswith("<|channel|>final<|message|>"):
                            prefix = prefix[:-len("<|channel|>final<|message|>")]

                        if impersonate:
                            prefix += "<|message|>"

                    if state['mode'] == 'chat' and not impersonate:
                        prefix = apply_extensions('bot_prefix', prefix, state)

                    prompt += prefix

        if state['mode'] == 'instruct' and 'enable_thinking' in state['instruction_template_str'] and not any((_continue, impersonate, state['enable_thinking'])):
            prompt += get_thinking_suppression_string(instruction_template)

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
                left, right = 0, len(user_message)

                while left < right:
                    mid = (left + right + 1) // 2

                    messages[-1]['content'] = user_message[:mid]
                    prompt = make_prompt(messages)
                    encoded_length = get_encoded_length(prompt)

                    if encoded_length <= max_length:
                        left = mid
                    else:
                        right = mid - 1

                messages[-1]['content'] = user_message[:left]
                prompt = make_prompt(messages)
                encoded_length = get_encoded_length(prompt)
                if encoded_length > max_length:
                    logger.error(f"Failed to build the chat prompt. The input is too long for the available context length.\n\nTruncation length: {state['truncation_length']}\nmax_new_tokens: {state['max_new_tokens']} (is it too high?)\nAvailable context length: {max_length}\n")
                    raise ValueError
                else:
                    # Calculate token counts for the log message
                    original_user_tokens = get_encoded_length(user_message)
                    truncated_user_tokens = get_encoded_length(user_message[:left])
                    total_context = max_length + state['max_new_tokens']

                    logger.warning(
                        f"User message truncated from {original_user_tokens} to {truncated_user_tokens} tokens. "
                        f"Context full: {max_length} input tokens ({total_context} total, {state['max_new_tokens']} for output). "
                        f"Increase ctx-size while loading the model to avoid truncation."
                    )

                    break

            prompt = make_prompt(messages)
            encoded_length = get_encoded_length(prompt)

    if also_return_rows:
        return prompt, [message['content'] for message in messages]
    else:
        return prompt


def count_prompt_tokens(text_input, state):
    """Count tokens for current history + input including attachments"""
    if shared.tokenizer is None:
        return "Tokenizer not available"

    try:
        # Handle dict format with text and files
        files = []
        if isinstance(text_input, dict):
            files = text_input.get('files', [])
            text = text_input.get('text', '')
        else:
            text = text_input
            files = []

        # Create temporary history copy to add attachments
        temp_history = copy.deepcopy(state['history'])
        if 'metadata' not in temp_history:
            temp_history['metadata'] = {}

        # Process attachments if any
        if files:
            row_idx = len(temp_history['internal'])
            for file_path in files:
                add_message_attachment(temp_history, row_idx, file_path, is_user=True)

        # Create temp state with modified history
        temp_state = copy.deepcopy(state)
        temp_state['history'] = temp_history

        # Build prompt using existing logic
        prompt = generate_chat_prompt(text, temp_state)
        current_tokens = get_encoded_length(prompt)
        max_tokens = temp_state['truncation_length']

        percentage = (current_tokens / max_tokens) * 100 if max_tokens > 0 else 0

        return f"History + Input:<br/>{current_tokens:,} / {max_tokens:,} tokens ({percentage:.1f}%)"

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return f"Error: {str(e)}"


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

    # Handle GPT-OSS as a special case
    if '<|channel|>final<|message|>' in state['instruction_template_str'] and "<|end|>" in result:
        result.remove("<|end|>")
        result.append("<|result|>")
        result = list(set(result))

    if shared.args.verbose:
        logger.info("STOPPING_STRINGS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(result)
        print()

    return result


def add_message_version(history, role, row_idx, is_current=True):
    key = f"{role}_{row_idx}"
    if 'metadata' not in history:
        history['metadata'] = {}
    if key not in history['metadata']:
        history['metadata'][key] = {}

    if "versions" not in history['metadata'][key]:
        history['metadata'][key]["versions"] = []

    # Determine which index to use for content based on role
    content_idx = 0 if role == 'user' else 1
    current_content = history['internal'][row_idx][content_idx]
    current_visible = history['visible'][row_idx][content_idx]

    history['metadata'][key]["versions"].append({
        "content": current_content,
        "visible_content": current_visible,
        "timestamp": get_current_timestamp()
    })

    if is_current:
        # Set the current_version_index to the newly added version (which is now the last one).
        history['metadata'][key]["current_version_index"] = len(history['metadata'][key]["versions"]) - 1


def add_message_attachment(history, row_idx, file_path, is_user=True):
    """Add a file attachment to a message in history metadata"""
    if 'metadata' not in history:
        history['metadata'] = {}

    key = f"{'user' if is_user else 'assistant'}_{row_idx}"

    if key not in history['metadata']:
        history['metadata'][key] = {"timestamp": get_current_timestamp()}
    if "attachments" not in history['metadata'][key]:
        history['metadata'][key]["attachments"] = []

    # Get file info using pathlib
    path = Path(file_path)
    filename = path.name
    file_extension = path.suffix.lower()

    try:
        # Handle different file types
        if file_extension == '.pdf':
            # Process PDF file
            content = extract_pdf_text(path)
            file_type = "application/pdf"
        elif file_extension == '.docx':
            content = extract_docx_text(path)
            file_type = "application/docx"
        else:
            # Default handling for text files
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            file_type = "text/plain"

        # Add attachment
        attachment = {
            "name": filename,
            "type": file_type,
            "content": content,
        }

        history['metadata'][key]["attachments"].append(attachment)
        return content  # Return the content for reuse
    except Exception as e:
        logger.error(f"Error processing attachment {filename}: {e}")
        return None


def extract_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    import PyPDF2

    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"

        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return f"[Error extracting PDF text: {str(e)}]"


def extract_docx_text(docx_path):
    """
    Extract text from a .docx file, including headers,
    body (paragraphs and tables), and footers.
    """
    try:
        import docx

        doc = docx.Document(docx_path)
        parts = []

        # 1) Extract non-empty header paragraphs from each section
        for section in doc.sections:
            for para in section.header.paragraphs:
                text = para.text.strip()
                if text:
                    parts.append(text)

        # 2) Extract body blocks (paragraphs and tables) in document order
        parent_elm = doc.element.body
        for child in parent_elm.iterchildren():
            if isinstance(child, docx.oxml.text.paragraph.CT_P):
                para = docx.text.paragraph.Paragraph(child, doc)
                text = para.text.strip()
                if text:
                    parts.append(text)

            elif isinstance(child, docx.oxml.table.CT_Tbl):
                table = docx.table.Table(child, doc)
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    parts.append("\t".join(cells))

        # 3) Extract non-empty footer paragraphs from each section
        for section in doc.sections:
            for para in section.footer.paragraphs:
                text = para.text.strip()
                if text:
                    parts.append(text)

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return f"[Error extracting DOCX text: {str(e)}]"


def generate_search_query(user_message, state):
    """Generate a search query from user message using the LLM"""
    # Augment the user message with search instruction
    augmented_message = f"{user_message}\n\n=====\n\nPlease turn the message above into a short web search query in the same language as the message. Respond with only the search query, nothing else."

    # Use a minimal state for search query generation but keep the full history
    search_state = state.copy()
    search_state['auto_max_new_tokens'] = True
    search_state['enable_thinking'] = False
    search_state['reasoning_effort'] = 'low'
    search_state['start_with'] = ""

    # Generate the full prompt using existing history + augmented message
    formatted_prompt = generate_chat_prompt(augmented_message, search_state)

    query = ""
    for reply in generate_reply(formatted_prompt, search_state, stopping_strings=[], is_chat=True):
        query = reply

    # Check for thinking block delimiters and extract content after them
    if "</think>" in query:
        query = query.rsplit("</think>", 1)[1]
    elif "<|start|>assistant<|channel|>final<|message|>" in query:
        query = query.rsplit("<|start|>assistant<|channel|>final<|message|>", 1)[1]

    # Strip and remove surrounding quotes if present
    query = query.strip()
    if len(query) >= 2 and query.startswith('"') and query.endswith('"'):
        query = query[1:-1]

    return query


def chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    # Handle dict format with text and files
    files = []
    if isinstance(text, dict):
        files = text.get('files', [])
        text = text.get('text', '')

    history = state['history']
    output = copy.deepcopy(history)
    output = apply_extensions('history', output)
    state = apply_extensions('state', state)

    # Handle GPT-OSS as a special case
    if '<|channel|>final<|message|>' in state['instruction_template_str']:
        state['skip_special_tokens'] = False

    # Let the jinja2 template handle the BOS token
    if state['mode'] in ['instruct', 'chat-instruct']:
        state['add_bos_token'] = False

    # Initialize metadata if not present
    if 'metadata' not in output:
        output['metadata'] = {}

    visible_text = None
    stopping_strings = get_stopping_strings(state)
    is_stream = state['stream']

    # Prepare the input
    if not (regenerate or _continue):
        visible_text = html.escape(text)

        # Process file attachments and store in metadata
        row_idx = len(output['internal'])

        # Add attachments to metadata only, not modifying the message text
        for file_path in files:
            add_message_attachment(output, row_idx, file_path, is_user=True)

        # Add web search results as attachments if enabled
        if state.get('enable_web_search', False):
            search_query = generate_search_query(text, state)
            add_web_search_attachments(output, row_idx, text, search_query, state)

        # Apply extensions
        text, visible_text = apply_extensions('chat_input', text, visible_text, state)
        text = apply_extensions('input', text, state, is_chat=True)

        # Current row index
        output['internal'].append([text, ''])
        output['visible'].append([visible_text, ''])
        # Add metadata with timestamp
        update_message_metadata(output['metadata'], "user", row_idx, timestamp=get_current_timestamp())

        # *Is typing...*
        if loading_message:
            yield {
                'visible': output['visible'][:-1] + [[output['visible'][-1][0], shared.processing_message]],
                'internal': output['internal'],
                'metadata': output['metadata']
            }
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        if regenerate:
            row_idx = len(output['internal']) - 1

            # Store the old response as a version before regenerating
            if not output['metadata'].get(f"assistant_{row_idx}", {}).get('versions'):
                add_message_version(output, "assistant", row_idx, is_current=False)

            # Add new empty version (will be filled during streaming)
            key = f"assistant_{row_idx}"
            output['metadata'][key]["versions"].append({
                "content": "",
                "visible_content": "",
                "timestamp": get_current_timestamp()
            })
            output['metadata'][key]["current_version_index"] = len(output['metadata'][key]["versions"]) - 1

            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, shared.processing_message]],
                    'internal': output['internal'][:-1] + [[text, '']],
                    'metadata': output['metadata']
                }
        elif _continue:
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
            if loading_message:
                yield {
                    'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']],
                    'internal': output['internal'],
                    'metadata': output['metadata']
                }

    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': output if _continue else {
            k: (v[:-1] if k in ['internal', 'visible'] else v)
            for k, v in output.items()
        }
    }

    prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Add timestamp for assistant's response at the start of generation
    row_idx = len(output['internal']) - 1
    update_message_metadata(output['metadata'], "assistant", row_idx, timestamp=get_current_timestamp(), model_name=shared.model_name)

    # Generate
    reply = None
    for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)):

        # Extract the reply
        if state['mode'] in ['chat', 'chat-instruct']:
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
        else:
            visible_reply = reply

        visible_reply = html.escape(visible_reply)

        if shared.stop_everything:
            output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
            yield output
            return

        if _continue:
            output['internal'][-1] = [text, last_reply[0] + reply]
            output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
        elif not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply.lstrip(' ')]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]

        # Keep version metadata in sync during streaming (for regeneration)
        if regenerate:
            row_idx = len(output['internal']) - 1
            key = f"assistant_{row_idx}"
            current_idx = output['metadata'][key]['current_version_index']
            output['metadata'][key]['versions'][current_idx].update({
                'content': output['internal'][row_idx][1],
                'visible_content': output['visible'][row_idx][1]
            })

        if is_stream:
            yield output

    if _continue:
        # Reprocess the entire internal text for extensions (like translation)
        full_internal = output['internal'][-1][1]
        if state['mode'] in ['chat', 'chat-instruct']:
            full_visible = re.sub("(<USER>|<user>|{{user}})", state['name1'], full_internal)
        else:
            full_visible = full_internal

        full_visible = html.escape(full_visible)
        output['visible'][-1][1] = apply_extensions('output', full_visible, state, is_chat=True)
    else:
        output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)

    # Final sync for version metadata (in case streaming was disabled)
    if regenerate:
        row_idx = len(output['internal']) - 1
        key = f"assistant_{row_idx}"
        current_idx = output['metadata'][key]['current_version_index']
        output['metadata'][key]['versions'][current_idx].update({
            'content': output['internal'][row_idx][1],
            'visible_content': output['visible'][row_idx][1]
        })

    yield output


def impersonate_wrapper(textbox, state):
    text = textbox['text']
    static_output = chat_html_wrapper(state['history'], state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    prompt = generate_chat_prompt('', state, impersonate=True)
    stopping_strings = get_stopping_strings(state)

    textbox['text'] = text + '...'
    yield textbox, static_output
    reply = None
    for reply in generate_reply(prompt + text, state, stopping_strings=stopping_strings, is_chat=True):
        textbox['text'] = (text + reply).lstrip(' ')
        yield textbox, static_output
        if shared.stop_everything:
            return


def generate_chat_reply(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    if regenerate or _continue:
        text = ''
        if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
            yield history
            return

    for history in chatbot_wrapper(text, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message, for_ui=for_ui):
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
    last_save_time = time.monotonic()
    save_interval = 8
    for i, history in enumerate(generate_chat_reply(text, state, regenerate, _continue, loading_message=True, for_ui=True)):
        yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'], last_message_only=(i > 0)), history
        if i == 0:
            time.sleep(0.125)  # We need this to make sure the first update goes through

        current_time = time.monotonic()
        # Save on first iteration or if save_interval seconds have passed
        if i == 0 or (current_time - last_save_time) >= save_interval:
            save_history(history, state['unique_id'], state['character_menu'], state['mode'])
            last_save_time = current_time

    save_history(history, state['unique_id'], state['character_menu'], state['mode'])


def remove_last_message(history):
    if 'metadata' not in history:
        history['metadata'] = {}

    if len(history['visible']) > 0 and history['internal'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
        row_idx = len(history['internal']) - 1
        last = history['visible'].pop()
        history['internal'].pop()

        # Remove metadata directly by known keys
        if f"user_{row_idx}" in history['metadata']:
            del history['metadata'][f"user_{row_idx}"]
        if f"assistant_{row_idx}" in history['metadata']:
            del history['metadata'][f"assistant_{row_idx}"]
    else:
        last = ['', '']

    return html.unescape(last[0]), history


def send_dummy_message(text, state):
    history = state['history']

    # Handle both dict and string inputs
    if isinstance(text, dict):
        text = text['text']

    # Initialize metadata if not present
    if 'metadata' not in history:
        history['metadata'] = {}

    row_idx = len(history['internal'])
    history['visible'].append([html.escape(text), ''])
    history['internal'].append([apply_extensions('input', text, state, is_chat=True), ''])
    update_message_metadata(history['metadata'], "user", row_idx, timestamp=get_current_timestamp())

    return history


def send_dummy_reply(text, state):
    history = state['history']

    # Handle both dict and string inputs
    if isinstance(text, dict):
        text = text['text']

    # Initialize metadata if not present
    if 'metadata' not in history:
        history['metadata'] = {}

    if len(history['visible']) > 0 and not history['visible'][-1][1] == '':
        row_idx = len(history['internal'])
        history['visible'].append(['', ''])
        history['internal'].append(['', ''])
        # We don't need to add system metadata

    row_idx = len(history['internal']) - 1
    history['visible'][-1][1] = html.escape(text)
    history['internal'][-1][1] = apply_extensions('input', text, state, is_chat=True)
    update_message_metadata(history['metadata'], "assistant", row_idx, timestamp=get_current_timestamp())

    return history


def redraw_html(history, name1, name2, mode, style, character, reset_cache=False):
    return chat_html_wrapper(history, name1, name2, mode, style, character, reset_cache=reset_cache)


def start_new_chat(state):
    mode = state['mode']
    # Initialize with empty metadata dictionary
    history = {'internal': [], 'visible': [], 'metadata': {}}

    if mode != 'instruct':
        greeting = replace_character_names(state['greeting'], state['name1'], state['name2'])
        if greeting != '':
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
            history['visible'] += [['', apply_extensions('output', html.escape(greeting), state, is_chat=True)]]

            # Add timestamp for assistant's greeting
            update_message_metadata(history['metadata'], "assistant", 0, timestamp=get_current_timestamp())

    unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    save_history(history, unique_id, state['character_menu'], state['mode'])

    return history


def get_history_file_path(unique_id, character, mode):
    if mode == 'instruct':
        p = Path(f'user_data/logs/instruct/{unique_id}.json')
    else:
        p = Path(f'user_data/logs/chat/{character}/{unique_id}.json')

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
        return Path('user_data/logs/instruct').glob('*.json')
    else:
        character = state['character_menu']

        # Handle obsolete filenames and paths
        old_p = Path(f'user_data/logs/{character}_persistent.json')
        new_p = Path(f'user_data/logs/persistent_{character}.json')
        if old_p.exists():
            logger.warning(f"Renaming \"{old_p}\" to \"{new_p}\"")
            old_p.rename(new_p)

        if new_p.exists():
            unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            p = get_history_file_path(unique_id, character, state['mode'])
            logger.warning(f"Moving \"{new_p}\" to \"{p}\"")
            p.parent.mkdir(exist_ok=True)
            new_p.rename(p)

        return Path(f'user_data/logs/chat/{character}').glob('*.json')


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
        return start_new_chat(state), None

    histories = find_all_histories(state)

    if len(histories) > 0:
        # Try to load the last visited chat for this character/mode
        chat_state = load_last_chat_state()
        key = get_chat_state_key(state['character_menu'], state['mode'])
        last_chat_id = chat_state.get("last_chats", {}).get(key)

        # If we have a stored last chat and it still exists, use it
        if last_chat_id and last_chat_id in histories:
            unique_id = last_chat_id
        else:
            # Fall back to most recent (current behavior)
            unique_id = histories[0]

        history = load_history(unique_id, state['character_menu'], state['mode'])
        return history, unique_id
    else:
        return start_new_chat(state), None


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


def get_chat_state_key(character, mode):
    """Generate a key for storing last chat state"""
    if mode == 'instruct':
        return 'instruct'
    else:
        return f"chat_{character}"


def load_last_chat_state():
    """Load the last chat state from file"""
    state_file = Path('user_data/logs/chat_state.json')
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.loads(f.read())
        except:
            pass

    return {"last_chats": {}}


def save_last_chat_state(character, mode, unique_id):
    """Save the last visited chat for a character/mode"""
    if shared.args.multi_user:
        return

    state = load_last_chat_state()
    key = get_chat_state_key(character, mode)
    state["last_chats"][key] = unique_id

    state_file = Path('user_data/logs/chat_state.json')
    state_file.parent.mkdir(exist_ok=True)
    with open(state_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(state, indent=2))


def load_history(unique_id, character, mode):
    p = get_history_file_path(unique_id, character, mode)

    if not p.exists():
        return {'internal': [], 'visible': [], 'metadata': {}}

    f = json.loads(open(p, 'rb').read())
    if 'internal' in f and 'visible' in f:
        history = f
    else:
        history = {
            'internal': f['data'],
            'visible': f['data_visible']
        }

    # Add metadata if it doesn't exist
    if 'metadata' not in history:
        history['metadata'] = {}
        # Add placeholder timestamps for existing messages
        for i, (user_msg, asst_msg) in enumerate(history['internal']):
            if user_msg and user_msg != '<|BEGIN-VISIBLE-CHAT|>':
                update_message_metadata(history['metadata'], "user", i, timestamp="")
            if asst_msg:
                update_message_metadata(history['metadata'], "assistant", i, timestamp="")

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

        # Add metadata if it doesn't exist
        if 'metadata' not in history:
            history['metadata'] = {}
            # Add placeholder timestamps
            for i, (user_msg, asst_msg) in enumerate(history['internal']):
                if user_msg and user_msg != '<|BEGIN-VISIBLE-CHAT|>':
                    update_message_metadata(history['metadata'], "user", i, timestamp="")
                if asst_msg:
                    update_message_metadata(history['metadata'], "assistant", i, timestamp="")

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

    for path in [Path(f"user_data/characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
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
        filepath = Path(f'user_data/characters/{character}.{extension}')
        if filepath.exists():
            break

    if filepath is None or not filepath.exists():
        logger.error(f"Could not find the character \"{character}\" inside user_data/characters. No character has been loaded.")
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


def restore_character_for_ui(state):
    """Reset character fields to the currently loaded character's saved values"""
    if state['character_menu'] and state['character_menu'] != 'None':
        try:
            name1, name2, picture, greeting, context = load_character(state['character_menu'], state['name1'], state['name2'])

            state['name2'] = name2
            state['greeting'] = greeting
            state['context'] = context
            state['character_picture'] = picture  # This triggers cache update via generate_pfp_cache

            return state, name2, context, greeting, picture

        except Exception as e:
            logger.error(f"Failed to reset character '{state['character_menu']}': {e}")
            return clear_character_for_ui(state)
    else:
        return clear_character_for_ui(state)


def clear_character_for_ui(state):
    """Clear all character fields and picture cache"""
    state['name2'] = shared.settings['name2']
    state['context'] = shared.settings['context']
    state['greeting'] = shared.settings['greeting']
    state['character_picture'] = None

    # Clear the cache files
    cache_folder = Path(shared.args.disk_cache_dir)
    for cache_file in ['pfp_character.png', 'pfp_character_thumb.png']:
        cache_path = Path(f'{cache_folder}/{cache_file}')
        if cache_path.exists():
            cache_path.unlink()

    return state, state['name2'], state['context'], state['greeting'], None


def load_instruction_template(template):
    if template == 'None':
        return ''

    for filepath in [Path(f'user_data/instruction-templates/{template}.yaml'), Path('user_data/instruction-templates/Alpaca.yaml')]:
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
    while Path(f'user_data/characters/{outfile_name}.yaml').exists():
        outfile_name = f'{name}_{i:03d}'
        i += 1

    with open(Path(f'user_data/characters/{outfile_name}.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_data)

    if img is not None:
        img.save(Path(f'user_data/characters/{outfile_name}.png'))

    logger.info(f'New character saved to "user_data/characters/{outfile_name}.yaml".')
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
    filepath = Path(f'user_data/characters/{filename}.yaml')
    save_file(filepath, data)
    path_to_img = Path(f'user_data/characters/{filename}.png')
    if picture is not None:
        picture.save(path_to_img)
        logger.info(f'Saved {path_to_img}.')


def delete_character(name, instruct=False):
    for extension in ["yml", "yaml", "json"]:
        delete_file(Path(f'user_data/characters/{name}.{extension}'))

    delete_file(Path(f'user_data/characters/{name}.png'))


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


def handle_send_dummy_message_click(text, state):
    history = send_dummy_message(text, state)
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html, {"text": "", "files": []}]


def handle_send_dummy_reply_click(text, state):
    history = send_dummy_reply(text, state)
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html, {"text": "", "files": []}]


def handle_remove_last_click(state):
    last_input, history = remove_last_message(state['history'])
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html, {"text": last_input, "files": []}]


def handle_unique_id_select(state):
    history = load_history(state['unique_id'], state['character_menu'], state['mode'])
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    # Save this as the last visited chat
    save_last_chat_state(state['character_menu'], state['mode'], state['unique_id'])

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
    filtered_histories = find_all_histories_with_first_prompts(state)
    filtered_ids = [h[1] for h in filtered_histories]
    index = str(filtered_ids.index(state['unique_id']))

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
    ]


def handle_branch_chat_click(state):
    branch_from_index = state['branch_index']
    if branch_from_index == -1:
        history = state['history']
    else:
        history = state['history']
        history['visible'] = history['visible'][:branch_from_index + 1]
        history['internal'] = history['internal'][:branch_from_index + 1]
    new_unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    save_history(history, new_unique_id, state['character_menu'], state['mode'])

    histories = find_all_histories_with_first_prompts(state)
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    past_chats_update = gr.update(choices=histories, value=new_unique_id)

    return [history, html, past_chats_update, -1]


def handle_edit_message_click(state):
    history = state['history']
    message_index = int(state['edit_message_index'])
    new_text = state['edit_message_text']
    role = state['edit_message_role']  # "user" or "assistant"

    if message_index >= len(history['internal']):
        html_output = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])
        return [history, html_output]

    role_idx = 0 if role == "user" else 1

    if 'metadata' not in history:
        history['metadata'] = {}

    key = f"{role}_{message_index}"
    if key not in history['metadata']:
        history['metadata'][key] = {}

    # If no versions exist yet for this message, store the current (pre-edit) content as the first version.
    if "versions" not in history['metadata'][key] or not history['metadata'][key]["versions"]:
        original_content = history['internal'][message_index][role_idx]
        original_visible = history['visible'][message_index][role_idx]
        original_timestamp = history['metadata'][key].get('timestamp', get_current_timestamp())

        history['metadata'][key]["versions"] = [{
            "content": original_content,
            "visible_content": original_visible,
            "timestamp": original_timestamp
        }]

    history['internal'][message_index][role_idx] = apply_extensions('input', new_text, state, is_chat=True)
    history['visible'][message_index][role_idx] = html.escape(new_text)

    add_message_version(history, role, message_index, is_current=True)

    save_history(history, state['unique_id'], state['character_menu'], state['mode'])
    html_output = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    return [history, html_output]


def handle_navigate_version_click(state):
    history = state['history']
    message_index = int(state['navigate_message_index'])
    direction = state['navigate_direction']
    role = state['navigate_message_role']

    if not role:
        logger.error("Role not provided for version navigation.")
        html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])
        return [history, html]

    key = f"{role}_{message_index}"
    if 'metadata' not in history or key not in history['metadata'] or 'versions' not in history['metadata'][key]:
        html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])
        return [history, html]

    metadata = history['metadata'][key]
    versions = metadata['versions']
    # Default to the last version if current_version_index is not set
    current_idx = metadata.get('current_version_index', len(versions) - 1 if versions else 0)

    if direction == 'left':
        new_idx = max(0, current_idx - 1)
    else:  # right
        new_idx = min(len(versions) - 1, current_idx + 1)

    if new_idx == current_idx:
        html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])
        return [history, html]

    msg_content_idx = 0 if role == 'user' else 1  # 0 for user content, 1 for assistant content in the pair
    version_to_load = versions[new_idx]
    history['internal'][message_index][msg_content_idx] = version_to_load['content']
    history['visible'][message_index][msg_content_idx] = version_to_load['visible_content']
    metadata['current_version_index'] = new_idx
    update_message_metadata(history['metadata'], role, message_index, timestamp=version_to_load['timestamp'])

    # Redraw and save
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])
    save_history(history, state['unique_id'], state['character_menu'], state['mode'])

    return [history, html]


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

    history, loaded_unique_id = load_latest_history(state)
    histories = find_all_histories_with_first_prompts(state)
    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=loaded_unique_id or histories[0][1])
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
        past_chats_update
    ]


def handle_character_picture_change(picture):
    """Update or clear cache when character picture changes"""
    cache_folder = Path(shared.args.disk_cache_dir)
    if not cache_folder.exists():
        cache_folder.mkdir()

    if picture is not None:
        # Save to cache
        picture.save(Path(f'{cache_folder}/pfp_character.png'), format='PNG')
        thumb = make_thumbnail(picture)
        thumb.save(Path(f'{cache_folder}/pfp_character_thumb.png'), format='PNG')
    else:
        # Remove cache files when picture is cleared
        for cache_file in ['pfp_character.png', 'pfp_character_thumb.png']:
            cache_path = Path(f'{cache_folder}/{cache_file}')
            if cache_path.exists():
                cache_path.unlink()


def handle_mode_change(state):
    history, loaded_unique_id = load_latest_history(state)
    histories = find_all_histories_with_first_prompts(state)

    # Ensure character picture cache exists
    if state['mode'] in ['chat', 'chat-instruct'] and state['character_menu'] and state['character_menu'] != 'None':
        generate_pfp_cache(state['character_menu'])

    html = redraw_html(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    convert_to_markdown.cache_clear()

    if len(histories) > 0:
        past_chats_update = gr.update(choices=histories, value=loaded_unique_id or histories[0][1])
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
        "user_data/instruction-templates/",
        contents,
        gr.update(visible=True)
    ]


def handle_delete_template_click(template):
    return [
        f"{template}.yaml",
        "user_data/instruction-templates/",
        gr.update(visible=False)
    ]


def handle_your_picture_change(picture, state):
    upload_your_profile_picture(picture)
    html = redraw_html(state['history'], state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'], reset_cache=True)

    return html


def handle_send_instruction_click(state):
    state['mode'] = 'instruct'
    state['history'] = {'internal': [], 'visible': [], 'metadata': {}}

    output = generate_chat_prompt("Input", state)

    if state["show_two_notebook_columns"]:
        return gr.update(), output, ""
    else:
        return output, gr.update(), gr.update()


def handle_send_chat_click(state):
    output = generate_chat_prompt("", state, _continue=True)

    if state["show_two_notebook_columns"]:
        return gr.update(), output, ""
    else:
        return output, gr.update(), gr.update()
