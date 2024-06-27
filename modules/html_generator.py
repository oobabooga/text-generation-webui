import functools
import html
import os
import re
import time
from pathlib import Path

import markdown
from PIL import Image, ImageOps

from modules import shared
from modules.utils import get_available_chat_styles

# This is to store the paths to the thumbnails of the profile pictures
image_cache = {}

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


def fix_newlines(string):
    string = string.replace('\n', '\n\n')
    string = re.sub(r"\n{3,}", "\n\n", string)
    string = string.strip()
    return string


def replace_blockquote(m):
    return m.group().replace('\n', '\n> ').replace('\\begin{blockquote}', '').replace('\\end{blockquote}', '')


@functools.lru_cache(maxsize=4096)
def convert_to_markdown(string):

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
    for line in string.split('\n'):
        if line.lstrip(' ').startswith('```'):
            is_code = not is_code

        result += line
        if is_code or line.startswith('|'):  # Don't add an extra \n for tables or code
            result += '\n'
        else:
            result += '\n\n'

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
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])

        # Remove the delete string from the HTML output
        pos = html_output.rfind(delete_str)
        if pos > -1:
            html_output = html_output[:pos] + html_output[pos + len(delete_str):]
    else:
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])

    # Unescape code blocks
    pattern = re.compile(r'<code[^>]*>(.*?)</code>', re.DOTALL)
    html_output = pattern.sub(lambda x: html.unescape(x.group()), html_output)

    return html_output


def convert_to_markdown_wrapped(string, use_cache=True):
    '''
    Used to avoid caching convert_to_markdown calls during streaming.
    '''

    if use_cache:
        return convert_to_markdown(string)

    return convert_to_markdown.__wrapped__(string)


def generate_basic_html(string):
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


def generate_instruct_html(history):
    output = f'<style>{instruct_css}</style><div class="chat" id="chat"><div class="messages">'
    for i, _row in enumerate(history):
        row = [convert_to_markdown_wrapped(entry, use_cache=i != len(history) - 1) for entry in _row]

        if row[0]:  # don't display empty user messages
            output += f"""
                  <div class="user-message">
                    <div class="text">
                      <div class="message-body">
                        {row[0]}
                      </div>
                    </div>
                  </div>
                """

        output += f"""
              <div class="assistant-message">
                <div class="text">
                  <div class="message-body">
                    {row[1]}
                  </div>
                </div>
              </div>
            """

    output += "</div></div>"

    return output


def generate_cai_chat_html(history, name1, name2, style, character, reset_cache=False):
    output = f'<style>{chat_styles[style]}</style><div class="chat" id="chat"><div class="messages">'

    # We use ?character and ?time.time() to force the browser to reset caches
    img_bot = f'<img src="file/cache/pfp_character_thumb.png?{character}" class="pfp_character">' if Path("cache/pfp_character_thumb.png").exists() else ''
    img_me = f'<img src="file/cache/pfp_me.png?{time.time() if reset_cache else ""}">' if Path("cache/pfp_me.png").exists() else ''

    for i, _row in enumerate(history):
        row = [convert_to_markdown_wrapped(entry, use_cache=i != len(history) - 1) for entry in _row]

        if row[0]:  # don't display empty user messages
            output += f"""
                  <div class="message">
                    <div class="circle-you">
                      {img_me}
                    </div>
                    <div class="text">
                      <div class="username">
                        {name1}
                      </div>
                      <div class="message-body">
                        {row[0]}
                      </div>
                    </div>
                  </div>
                """

        output += f"""
              <div class="message">
                <div class="circle-bot">
                  {img_bot}
                </div>
                <div class="text">
                  <div class="username">
                    {name2}
                  </div>
                  <div class="message-body">
                    {row[1]}
                  </div>
                </div>
              </div>
            """

    output += "</div></div>"
    return output


def generate_chat_html(history, name1, name2, reset_cache=False):
    output = f'<style>{chat_styles["wpp"]}</style><div class="chat" id="chat"><div class="messages">'

    for i, _row in enumerate(history):
        row = [convert_to_markdown_wrapped(entry, use_cache=i != len(history) - 1) for entry in _row]

        if row[0]:  # don't display empty user messages
            output += f"""
              <div class="message">
                <div class="text-you">
                  <div class="message-body">
                    {row[0]}
                  </div>
                </div>
              </div>
            """

        output += f"""
          <div class="message">
            <div class="text-bot">
              <div class="message-body">
                {row[1]}
              </div>
            </div>
          </div>
        """

    output += "</div></div>"
    return output


def chat_html_wrapper(history, name1, name2, mode, style, character, reset_cache=False):
    if mode == 'instruct':
        return generate_instruct_html(history['visible'])
    elif style == 'wpp':
        return generate_chat_html(history['visible'], name1, name2)
    else:
        return generate_cai_chat_html(history['visible'], name1, name2, style, character, reset_cache)
