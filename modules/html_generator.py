import datetime
import functools
import html
import os
import re
import time
from pathlib import Path

import markdown
from PIL import Image, ImageOps

from modules import shared
from modules.sane_markdown_lists import SaneListExtension
from modules.utils import get_available_chat_styles

# This is to store the paths to the thumbnails of the profile pictures
image_cache = {}


def minify_css(css: str) -> str:
    # Step 1: Remove comments
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.DOTALL)

    # Step 2: Remove leading and trailing whitespace
    css = re.sub(r'^[ \t]*|[ \t]*$', '', css, flags=re.MULTILINE)

    # Step 3: Remove spaces after specific characters ({ : ; ,})
    css = re.sub(r'([:{;,])\s+', r'\1', css)

    # Step 4: Remove spaces before `{`
    css = re.sub(r'\s+{', '{', css)

    # Step 5: Remove empty lines
    css = re.sub(r'^\s*$', '', css, flags=re.MULTILINE)

    # Step 6: Collapse all lines into one
    css = re.sub(r'\n', '', css)

    return css


with open(Path(__file__).resolve().parent / '../css/html_readable_style.css', 'r', encoding='utf-8') as f:
    readable_css = f.read()
with open(Path(__file__).resolve().parent / '../css/html_instruct_style.css', 'r', encoding='utf-8') as f:
    instruct_css = f.read()

# Custom chat styles
chat_styles = {}
for k in get_available_chat_styles():
    with open(Path(f'css/chat_style-{k}.css'), 'r', encoding='utf-8') as f:
        chat_styles[k] = f.read()

# Handle styles that derive from other styles
for k in chat_styles:
    lines = chat_styles[k].split('\n')
    input_string = lines[0]
    match = re.search(r'chat_style-([a-z\-]*)\.css', input_string)

    if match:
        style = match.group(1)
        chat_styles[k] = chat_styles.get(style, '') + '\n\n' + '\n'.join(lines[1:])

# Reduce the size of the CSS sources above
readable_css = minify_css(readable_css)
instruct_css = minify_css(instruct_css)
for k in chat_styles:
    chat_styles[k] = minify_css(chat_styles[k])


def fix_newlines(string):
    string = string.replace('\n', '\n\n')
    string = re.sub(r"\n{3,}", "\n\n", string)
    string = string.strip()
    return string


def replace_quotes(text):
    # Define a list of quote pairs (opening and closing), using HTML entities
    quote_pairs = [
        ('&quot;', '&quot;'),  # Double quotes
        ('&ldquo;', '&rdquo;'),  # Unicode left and right double quotation marks
        ('&lsquo;', '&rsquo;'),  # Unicode left and right single quotation marks
        ('&laquo;', '&raquo;'),  # French quotes
        ('&bdquo;', '&ldquo;'),  # German quotes
        ('&lsquo;', '&rsquo;'),  # Alternative single quotes
        ('&#8220;', '&#8221;'),  # Unicode quotes (numeric entities)
        ('&#x201C;', '&#x201D;'),  # Unicode quotes (hex entities)
        ('\u201C', '\u201D'),  # Unicode quotes (literal chars)
    ]

    # Create a regex pattern that matches any of the quote pairs, including newlines
    pattern = '|'.join(f'({re.escape(open_q)})(.*?)({re.escape(close_q)})' for open_q, close_q in quote_pairs)

    # Replace matched patterns with <q> tags, keeping original quotes
    def replacer(m):
        # Find the first non-None group set
        for i in range(1, len(m.groups()), 3):  # Step through each sub-pattern's groups
            if m.group(i):  # If this sub-pattern matched
                return f'<q>{m.group(i)}{m.group(i + 1)}{m.group(i + 2)}</q>'

        return m.group(0)  # Fallback (shouldn't happen)

    replaced_text = re.sub(pattern, replacer, text, flags=re.DOTALL)
    return replaced_text


def replace_blockquote(m):
    return m.group().replace('\n', '\n> ').replace('\\begin{blockquote}', '').replace('\\end{blockquote}', '')


def extract_thinking_block(string):
    """Extract thinking blocks from the beginning of a string."""
    if not string:
        return None, string

    THINK_START_TAG = "&lt;think&gt;"
    THINK_END_TAG = "&lt;/think&gt;"

    # Look for think tag first
    start_pos = string.find(THINK_START_TAG)
    end_pos = string.find(THINK_END_TAG)

    # If think tags found, use existing logic
    if start_pos != -1 or end_pos != -1:
        # handle missing start or end tags
        if start_pos == -1:
            thought_start = 0
        else:
            thought_start = start_pos + len(THINK_START_TAG)
        if end_pos == -1:
            thought_end = len(string)
            content_start = len(string)
        else:
            thought_end = end_pos
            content_start = end_pos + len(THINK_END_TAG)
        thinking_content = string[thought_start:thought_end]
        remaining_content = string[content_start:]
        return thinking_content, remaining_content

    # If think tags not found, try alternative format
    ALT_START = "&lt;|channel|&gt;analysis&lt;|message|&gt;"
    ALT_END = "&lt;|end|&gt;"
    ALT_CONTENT_START = "&lt;|start|&gt;assistant&lt;|channel|&gt;final&lt;|message|&gt;"

    alt_start_pos = string.find(ALT_START)
    alt_end_pos = string.find(ALT_END)
    alt_content_pos = string.find(ALT_CONTENT_START)

    if alt_start_pos != -1 or alt_end_pos != -1:
        if alt_start_pos == -1:
            thought_start = 0
        else:
            thought_start = alt_start_pos + len(ALT_START)

        # If no explicit end tag but content start exists, use content start as end
        if alt_end_pos == -1:
            if alt_content_pos != -1:
                thought_end = alt_content_pos
                content_start = alt_content_pos + len(ALT_CONTENT_START)
            else:
                thought_end = len(string)
                content_start = len(string)
        else:
            thought_end = alt_end_pos
            content_start = alt_content_pos + len(ALT_CONTENT_START) if alt_content_pos != -1 else alt_end_pos + len(ALT_END)

        thinking_content = string[thought_start:thought_end]
        remaining_content = string[content_start:]
        return thinking_content, remaining_content

    # Return if neither format is found
    return None, string


@functools.lru_cache(maxsize=None)
def convert_to_markdown(string, message_id=None):
    if not string:
        return ""

    # Use a default message ID if none provided
    if message_id is None:
        message_id = "unknown"

    # Extract thinking block if present
    thinking_content, remaining_content = extract_thinking_block(string)

    # Process the main content
    html_output = process_markdown_content(remaining_content)

    # If thinking content was found, process it using the same function
    if thinking_content is not None:
        thinking_html = process_markdown_content(thinking_content)

        # Generate unique ID for the thinking block
        block_id = f"thinking-{message_id}-0"

        # Check if thinking is complete or still in progress
        is_streaming = not remaining_content
        title_text = "Thinking..." if is_streaming else "Thought"

        thinking_block = f'''
        <details class="thinking-block" data-block-id="{block_id}" data-streaming="{str(is_streaming).lower()}">
            <summary class="thinking-header">
                {info_svg_small}
                <span class="thinking-title">{title_text}</span>
            </summary>
            <div class="thinking-content pretty_scrollbar">{thinking_html}</div>
        </details>
        '''

        # Prepend the thinking block to the message HTML
        html_output = thinking_block + html_output

    return html_output


def process_markdown_content(string):
    """Process a string through the markdown conversion pipeline."""
    if not string:
        return ""

    # Make \[ \]  LaTeX equations inline
    pattern = r'^\s*\\\[\s*\n([\s\S]*?)\n\s*\\\]\s*$'
    replacement = r'\\[ \1 \\]'
    string = re.sub(pattern, replacement, string, flags=re.MULTILINE)

    # Escape backslashes
    string = string.replace('\\', '\\\\')

    # Quote to <q></q>
    string = replace_quotes(string)

    # Blockquote
    string = re.sub(r'(^|[\n])&gt;', r'\1>', string)
    pattern = re.compile(r'\\begin{blockquote}(.*?)\\end{blockquote}', re.DOTALL)
    string = pattern.sub(replace_blockquote, string)

    # Code
    string = string.replace('\\begin{code}', '```')
    string = string.replace('\\end{code}', '```')
    string = string.replace('\\begin{align*}', '$$')
    string = string.replace('\\end{align*}', '$$')
    string = string.replace('\\begin{align}', '$$')
    string = string.replace('\\end{align}', '$$')
    string = string.replace('\\begin{equation}', '$$')
    string = string.replace('\\end{equation}', '$$')
    string = string.replace('\\begin{equation*}', '$$')
    string = string.replace('\\end{equation*}', '$$')
    string = re.sub(r"(.)```", r"\1\n```", string)

    result = ''
    is_code = False
    is_latex = False

    for line in string.split('\n'):
        stripped_line = line.strip()

        if stripped_line.startswith('```'):
            is_code = not is_code
        elif stripped_line.startswith('$$'):
            is_latex = not is_latex
        elif stripped_line.endswith('$$'):
            is_latex = False
        elif stripped_line.startswith('\\\\['):
            is_latex = True
        elif stripped_line.startswith('\\\\]'):
            is_latex = False
        elif stripped_line.endswith('\\\\]'):
            is_latex = False

        result += line

        # Don't add an extra \n for code, LaTeX, or tables
        if is_code or is_latex or line.startswith('|'):
            result += '\n'
        # Also don't add an extra \n for lists
        elif stripped_line.startswith('-') or stripped_line.startswith('*') or stripped_line.startswith('+') or stripped_line.startswith('>') or re.match(r'\d+\.', stripped_line):
            result += '  \n'
        else:
            result += '  \n'

    result = result.strip()
    if is_code:
        result += '\n```'  # Unfinished code block

    # Unfinished list, like "\n1.". A |delete| string is added and then
    # removed to force a <ol> or <ul> to be generated instead of a <p>.
    list_item_pattern = r'(\n\d+\.?|\n\s*[-*+]\s*([*_~]{1,3})?)$'
    if re.search(list_item_pattern, result):
        delete_str = '|delete|'

        if re.search(r'(\d+\.?)$', result) and not result.endswith('.'):
            result += '.'

        # Add the delete string after the list item
        result = re.sub(list_item_pattern, r'\g<1> ' + delete_str, result)

        # Convert to HTML using markdown
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables', SaneListExtension()])

        # Remove the delete string from the HTML output
        pos = html_output.rfind(delete_str)
        if pos > -1:
            html_output = html_output[:pos] + html_output[pos + len(delete_str):]
    else:
        # Convert to HTML using markdown
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables', SaneListExtension()])

    # Unescape code blocks
    pattern = re.compile(r'<code[^>]*>(.*?)</code>', re.DOTALL)
    html_output = pattern.sub(lambda x: html.unescape(x.group()), html_output)

    # Unescape backslashes
    html_output = html_output.replace('\\\\', '\\')

    return html_output


def convert_to_markdown_wrapped(string, message_id=None, use_cache=True):
    '''
    Used to avoid caching convert_to_markdown calls during streaming.
    '''

    if use_cache:
        return convert_to_markdown(string, message_id=message_id)

    return convert_to_markdown.__wrapped__(string, message_id=message_id)


def generate_basic_html(string):
    convert_to_markdown.cache_clear()
    string = convert_to_markdown(string)
    string = f'<style>{readable_css}</style><div class="readable-container">{string}</div>'
    return string


def make_thumbnail(image):
    image = image.resize((350, round(image.size[1] / image.size[0] * 350)), Image.Resampling.LANCZOS)
    if image.size[1] > 470:
        image = ImageOps.fit(image, (350, 470), Image.LANCZOS)

    return image


def get_image_cache(path):
    cache_folder = Path(shared.args.disk_cache_dir)
    if not cache_folder.exists():
        cache_folder.mkdir()

    mtime = os.stat(path).st_mtime
    if (path in image_cache and mtime != image_cache[path][0]) or (path not in image_cache):
        img = make_thumbnail(Image.open(path))

        old_p = Path(f'{cache_folder}/{path.name}_cache.png')
        p = Path(f'{cache_folder}/cache_{path.name}.png')
        if old_p.exists():
            old_p.rename(p)

        output_file = p
        img.convert('RGBA').save(output_file, format='PNG')
        image_cache[path] = [mtime, output_file.as_posix()]

    return image_cache[path][1]


copy_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-copy"><path d="M8 8m0 2a2 2 0 0 1 2 -2h8a2 2 0 0 1 2 2v8a2 2 0 0 1 -2 2h-8a2 2 0 0 1 -2 -2z"></path><path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path></svg>'''
refresh_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-repeat"><path d="M4 12v-3a3 3 0 0 1 3 -3h13m-3 -3l3 3l-3 3"></path><path d="M20 12v3a3 3 0 0 1 -3 3h-13m3 3l-3 -3l3 -3"></path></svg>'''
continue_svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="20"  height="20"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-player-play"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M7 4v16l13 -8z" /></svg>'''
remove_svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="20"  height="20"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-trash"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 7l16 0" /><path d="M10 11l0 6" /><path d="M14 11l0 6" /><path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12" /><path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3" /></svg>'''
branch_svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-git-branch"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M7 18m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M7 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M17 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M7 8l0 8" /><path d="M9 18h6a2 2 0 0 0 2 -2v-5" /><path d="M14 14l3 -3l3 3" /></svg>'''
edit_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-pencil"><path d="M4 20h4l10.5 -10.5a2.828 2.828 0 1 0 -4 -4l-10.5 10.5v4"></path><path d="M13.5 6.5l4 4"></path></svg>'''
info_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="thinking-icon tabler-icon tabler-icon-info-circle"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 2a10 10 0 0 1 0 20a10 10 0 0 1 0 -20z" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>'''
info_svg_small = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="thinking-icon tabler-icon tabler-icon-info-circle"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 2a10 10 0 0 1 0 20a10 10 0 0 1 0 -20z" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>'''
attachment_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.48-8.48l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>'''

copy_button = f'<button class="footer-button footer-copy-button" title="Copy" onclick="copyToClipboard(this)">{copy_svg}</button>'
branch_button = f'<button class="footer-button footer-branch-button" title="Branch here" onclick="branchHere(this)">{branch_svg}</button>'
edit_button = f'<button class="footer-button footer-edit-button" title="Edit" onclick="editHere(this)">{edit_svg}</button>'
refresh_button = f'<button class="footer-button footer-refresh-button" title="Regenerate" onclick="regenerateClick()">{refresh_svg}</button>'
continue_button = f'<button class="footer-button footer-continue-button" title="Continue" onclick="continueClick()">{continue_svg}</button>'
remove_button = f'<button class="footer-button footer-remove-button" title="Remove last reply" onclick="removeLastClick()">{remove_svg}</button>'
info_button = f'<button class="footer-button footer-info-button" title="message">{info_svg}</button>'


def format_message_timestamp(history, role, index, tooltip_include_timestamp=True):
    """Get a formatted timestamp HTML span for a message if available"""
    key = f"{role}_{index}"
    if 'metadata' in history and key in history['metadata'] and history['metadata'][key].get('timestamp'):
        timestamp = history['metadata'][key]['timestamp']
        tooltip_text = get_message_tooltip(history, role, index, include_timestamp=tooltip_include_timestamp)
        title_attr = f' title="{html.escape(tooltip_text)}"' if tooltip_text else ''
        return f"<span class='timestamp'{title_attr}>{timestamp}</span>"

    return ""


def format_message_attachments(history, role, index):
    """Get formatted HTML for message attachments if available"""
    key = f"{role}_{index}"
    if 'metadata' in history and key in history['metadata'] and 'attachments' in history['metadata'][key]:
        attachments = history['metadata'][key]['attachments']
        if not attachments:
            return ""

        attachments_html = '<div class="message-attachments">'
        for attachment in attachments:
            name = html.escape(attachment["name"])

            # Make clickable if URL exists
            if "url" in attachment:
                name = f'<a href="{html.escape(attachment["url"])}" target="_blank" rel="noopener noreferrer">{name}</a>'

            attachments_html += (
                f'<div class="attachment-box">'
                f'<div class="attachment-icon">{attachment_svg}</div>'
                f'<div class="attachment-name">{name}</div>'
                f'</div>'
            )
        attachments_html += '</div>'
        return attachments_html

    return ""


def get_message_tooltip(history, role, index, include_timestamp=True):
    """Get tooltip text combining timestamp and model name for a message"""
    key = f"{role}_{index}"
    if 'metadata' not in history or key not in history['metadata']:
        return ""

    meta = history['metadata'][key]
    tooltip_parts = []

    if include_timestamp and meta.get('timestamp'):
        tooltip_parts.append(meta['timestamp'])
    if meta.get('model_name'):
        tooltip_parts.append(f"Model: {meta['model_name']}")

    return " | ".join(tooltip_parts)


def get_version_navigation_html(history, i, role):
    """Generate simple navigation arrows for message versions"""
    key = f"{role}_{i}"
    metadata = history.get('metadata', {})

    if key not in metadata or 'versions' not in metadata[key]:
        return ""

    versions = metadata[key]['versions']
    # Default to the last version if current_version_index isn't set in metadata
    current_idx = metadata[key].get('current_version_index', len(versions) - 1 if versions else 0)

    if len(versions) <= 1:
        return ""

    left_disabled = ' disabled' if current_idx == 0 else ''
    right_disabled = ' disabled' if current_idx >= len(versions) - 1 else ''

    left_arrow = f'<button class="footer-button version-nav-button"{left_disabled} onclick="navigateVersion(this, \'left\')" title="Previous version">&lt;</button>'
    right_arrow = f'<button class="footer-button version-nav-button"{right_disabled} onclick="navigateVersion(this, \'right\')" title="Next version">&gt;</button>'
    position = f'<span class="version-position">{current_idx + 1}/{len(versions)}</span>'

    return f'<div class="version-navigation">{left_arrow}{position}{right_arrow}</div>'


def actions_html(history, i, role, info_message=""):
    action_buttons = ""
    version_nav_html = ""

    if role == "assistant":
        action_buttons = (
            f'{copy_button}'
            f'{edit_button}'
            f'{refresh_button if i == len(history["visible"]) - 1 else ""}'
            f'{continue_button if i == len(history["visible"]) - 1 else ""}'
            f'{remove_button if i == len(history["visible"]) - 1 else ""}'
            f'{branch_button}'
        )

        version_nav_html = get_version_navigation_html(history, i, "assistant")
    elif role == "user":
        action_buttons = (
            f'{copy_button}'
            f'{edit_button}'
        )

        version_nav_html = get_version_navigation_html(history, i, "user")

    return (f'<div class="message-actions">'
            f'{action_buttons}'
            f'{info_message}'
            f'</div>'
            f'{version_nav_html}')


def generate_instruct_html(history, last_message_only=False):
    if not last_message_only:
        output = f'<style>{instruct_css}</style><div class="chat" id="chat" data-mode="instruct"><div class="messages">'
    else:
        output = ""

    def create_message(role, content, raw_content):
        """Inner function that captures variables from outer scope."""
        class_name = "user-message" if role == "user" else "assistant-message"

        # Get role-specific data
        timestamp = format_message_timestamp(history, role, i)
        attachments = format_message_attachments(history, role, i)

        # Create info button if timestamp exists
        info_message = ""
        if timestamp:
            tooltip_text = get_message_tooltip(history, role, i)
            info_message = info_button.replace('title="message"', f'title="{html.escape(tooltip_text)}"')

        return (
            f'<div class="{class_name}" '
            f'data-raw="{html.escape(raw_content, quote=True)}"'
            f'data-index={i}>'
            f'<div class="text">'
            f'<div class="message-body">{content}</div>'
            f'{attachments}'
            f'{actions_html(history, i, role, info_message)}'
            f'</div>'
            f'</div>'
        )

    # Determine range
    start_idx = len(history['visible']) - 1 if last_message_only else 0
    end_idx = len(history['visible'])

    for i in range(start_idx, end_idx):
        row_visible = history['visible'][i]
        row_internal = history['internal'][i]

        # Convert content
        if last_message_only:
            converted_visible = [None, convert_to_markdown_wrapped(row_visible[1], message_id=i, use_cache=i != len(history['visible']) - 1)]
        else:
            converted_visible = [convert_to_markdown_wrapped(entry, message_id=i, use_cache=i != len(history['visible']) - 1) for entry in row_visible]

        # Generate messages
        if not last_message_only and converted_visible[0]:
            output += create_message("user", converted_visible[0], row_internal[0])

        output += create_message("assistant", converted_visible[1], row_internal[1])

    if not last_message_only:
        output += "</div></div>"

    return output


def get_character_image_with_cache_buster():
    """Get character image URL with cache busting based on file modification time"""
    cache_path = Path("user_data/cache/pfp_character_thumb.png")
    if cache_path.exists():
        mtime = int(cache_path.stat().st_mtime)
        return f'<img src="file/user_data/cache/pfp_character_thumb.png?{mtime}" class="pfp_character">'

    return ''


def generate_cai_chat_html(history, name1, name2, style, character, reset_cache=False, last_message_only=False):
    if not last_message_only:
        output = f'<style>{chat_styles[style]}</style><div class="chat" id="chat"><div class="messages">'
    else:
        output = ""

    img_bot = get_character_image_with_cache_buster()

    def create_message(role, content, raw_content):
        """Inner function for CAI-style messages."""
        circle_class = "circle-you" if role == "user" else "circle-bot"
        name = name1 if role == "user" else name2

        # Get role-specific data
        timestamp = format_message_timestamp(history, role, i, tooltip_include_timestamp=False)
        attachments = format_message_attachments(history, role, i)

        # Get appropriate image
        if role == "user":
            img = (f'<img src="file/user_data/cache/pfp_me.png?{time.time() if reset_cache else ""}">'
                   if Path("user_data/cache/pfp_me.png").exists() else '')
        else:
            img = img_bot

        return (
            f'<div class="message" '
            f'data-raw="{html.escape(raw_content, quote=True)}"'
            f'data-index={i}>'
            f'<div class="{circle_class}">{img}</div>'
            f'<div class="text">'
            f'<div class="username">{name}{timestamp}</div>'
            f'<div class="message-body">{content}</div>'
            f'{attachments}'
            f'{actions_html(history, i, role)}'
            f'</div>'
            f'</div>'
        )

    # Determine range
    start_idx = len(history['visible']) - 1 if last_message_only else 0
    end_idx = len(history['visible'])

    for i in range(start_idx, end_idx):
        row_visible = history['visible'][i]
        row_internal = history['internal'][i]

        # Convert content
        if last_message_only:
            converted_visible = [None, convert_to_markdown_wrapped(row_visible[1], message_id=i, use_cache=i != len(history['visible']) - 1)]
        else:
            converted_visible = [convert_to_markdown_wrapped(entry, message_id=i, use_cache=i != len(history['visible']) - 1) for entry in row_visible]

        # Generate messages
        if not last_message_only and converted_visible[0]:
            output += create_message("user", converted_visible[0], row_internal[0])

        output += create_message("assistant", converted_visible[1], row_internal[1])

    if not last_message_only:
        output += "</div></div>"

    return output


def time_greeting():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning!"
    elif 12 <= current_hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"


def chat_html_wrapper(history, name1, name2, mode, style, character, reset_cache=False, last_message_only=False):
    if len(history['visible']) == 0:
        greeting = f"<div class=\"welcome-greeting\">{time_greeting()} How can I help you today?</div>"
        result = f'<div class="chat" id="chat">{greeting}</div>'
    elif mode == 'instruct':
        result = generate_instruct_html(history, last_message_only=last_message_only)
    else:
        result = generate_cai_chat_html(history, name1, name2, style, character, reset_cache=reset_cache, last_message_only=last_message_only)

    return {'html': result, 'last_message_only': last_message_only}
