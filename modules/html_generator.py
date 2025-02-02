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


with open(Path(__file__).resolve().parent / '../css/html_readable_style.css', 'r') as f:
    readable_css = f.read()
with open(Path(__file__).resolve().parent / '../css/html_instruct_style.css', 'r') as f:
    instruct_css = f.read()

# Custom chat styles
chat_styles = {}
for k in get_available_chat_styles():
    chat_styles[k] = open(Path(f'css/chat_style-{k}.css'), 'r').read()

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


@functools.lru_cache(maxsize=None)
def convert_to_markdown(string):
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


def convert_to_markdown_wrapped(string, use_cache=True):
    '''
    Used to avoid caching convert_to_markdown calls during streaming.
    '''

    if use_cache:
        return convert_to_markdown(string)

    return convert_to_markdown.__wrapped__(string)


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

copy_button = f'<button class="footer-button footer-copy-button" title="Copy" onclick="copyToClipboard(this)">{copy_svg}</button>'
refresh_button = f'<button class="footer-button footer-refresh-button" title="Regenerate" onclick="regenerateClick()">{refresh_svg}</button>'
continue_button = f'<button class="footer-button footer-continue-button" title="Continue" onclick="continueClick()">{continue_svg}</button>'
remove_button = f'<button class="footer-button footer-remove-button" title="Remove last reply" onclick="removeLastClick()">{remove_svg}</button>'


def generate_instruct_html(history):
    output = f'<style>{instruct_css}</style><div class="chat" id="chat"><div class="messages">'

    for i in range(len(history['visible'])):
        row_visible = history['visible'][i]
        row_internal = history['internal'][i]
        converted_visible = [convert_to_markdown_wrapped(entry, use_cache=i != len(history['visible']) - 1) for entry in row_visible]

        if converted_visible[0]:  # Don't display empty user messages
            output += (
                f'<div class="user-message" '
                f'data-raw="{html.escape(row_internal[0], quote=True)}">'
                f'<div class="text">'
                f'<div class="message-body">{converted_visible[0]}</div>'
                f'{copy_button}'
                f'</div>'
                f'</div>'
            )

        output += (
            f'<div class="assistant-message" '
            f'data-raw="{html.escape(row_internal[1], quote=True)}">'
            f'<div class="text">'
            f'<div class="message-body">{converted_visible[1]}</div>'
            f'{copy_button}'
            f'{refresh_button if i == len(history["visible"]) - 1 else ""}'
            f'{continue_button if i == len(history["visible"]) - 1 else ""}'
            f'{remove_button if i == len(history["visible"]) - 1 else ""}'
            f'</div>'
            f'</div>'
        )

    output += "</div></div>"
    return output


def generate_cai_chat_html(history, name1, name2, style, character, reset_cache=False):
    output = f'<style>{chat_styles[style]}</style><div class="chat" id="chat"><div class="messages">'

    # We use ?character and ?time.time() to force the browser to reset caches
    img_bot = (
        f'<img src="file/cache/pfp_character_thumb.png?{character}" class="pfp_character">'
        if Path("cache/pfp_character_thumb.png").exists() else ''
    )

    img_me = (
        f'<img src="file/cache/pfp_me.png?{time.time() if reset_cache else ""}">'
        if Path("cache/pfp_me.png").exists() else ''
    )

    for i in range(len(history['visible'])):
        row_visible = history['visible'][i]
        row_internal = history['internal'][i]
        converted_visible = [convert_to_markdown_wrapped(entry, use_cache=i != len(history['visible']) - 1) for entry in row_visible]

        if converted_visible[0]:  # Don't display empty user messages
            output += (
                f'<div class="message" '
                f'data-raw="{html.escape(row_internal[0], quote=True)}">'
                f'<div class="circle-you">{img_me}</div>'
                f'<div class="text">'
                f'<div class="username">{name1}</div>'
                f'<div class="message-body">{converted_visible[0]}</div>'
                f'{copy_button}'
                f'</div>'
                f'</div>'
            )

        output += (
            f'<div class="message" '
            f'data-raw="{html.escape(row_internal[1], quote=True)}">'
            f'<div class="circle-bot">{img_bot}</div>'
            f'<div class="text">'
            f'<div class="username">{name2}</div>'
            f'<div class="message-body">{converted_visible[1]}</div>'
            f'{copy_button}'
            f'{refresh_button if i == len(history["visible"]) - 1 else ""}'
            f'{continue_button if i == len(history["visible"]) - 1 else ""}'
            f'{remove_button if i == len(history["visible"]) - 1 else ""}'
            f'</div>'
            f'</div>'
        )

    output += "</div></div>"
    return output


def generate_chat_html(history, name1, name2, reset_cache=False):
    output = f'<style>{chat_styles["wpp"]}</style><div class="chat" id="chat"><div class="messages">'

    for i in range(len(history['visible'])):
        row_visible = history['visible'][i]
        row_internal = history['internal'][i]
        converted_visible = [convert_to_markdown_wrapped(entry, use_cache=i != len(history['visible']) - 1) for entry in row_visible]

        if converted_visible[0]:  # Don't display empty user messages
            output += (
                f'<div class="message" '
                f'data-raw="{html.escape(row_internal[0], quote=True)}">'
                f'<div class="text-you">'
                f'<div class="message-body">{converted_visible[0]}</div>'
                f'{copy_button}'
                f'</div>'
                f'</div>'
            )

        output += (
            f'<div class="message" '
            f'data-raw="{html.escape(row_internal[1], quote=True)}">'
            f'<div class="text-bot">'
            f'<div class="message-body">{converted_visible[1]}</div>'
            f'{copy_button}'
            f'{refresh_button if i == len(history["visible"]) - 1 else ""}'
            f'{continue_button if i == len(history["visible"]) - 1 else ""}'
            f'{remove_button if i == len(history["visible"]) - 1 else ""}'
            f'</div>'
            f'</div>'
        )

    output += "</div></div>"
    return output


def chat_html_wrapper(history, name1, name2, mode, style, character, reset_cache=False):
    if mode == 'instruct':
        return generate_instruct_html(history)
    elif style == 'wpp':
        return generate_chat_html(history, name1, name2)
    else:
        return generate_cai_chat_html(history, name1, name2, style, character, reset_cache)
