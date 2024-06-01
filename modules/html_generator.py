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

# Cache for profile picture thumbnails
image_cache = {}

# Load CSS files
def load_css(file_name):
    with open(Path(__file__).resolve().parent / f'../css/{file_name}', 'r') as f:
        return f.read()

readable_css = load_css('html_readable_style.css')
instruct_css = load_css('html_instruct_style.css')

# Load custom chat styles
chat_styles = {style: load_css(f'chat_style-{style}.css') for style in get_available_chat_styles()}

# Handle derived styles
for key, style_css in chat_styles.items():
    lines = style_css.split('\n')
    if match := re.match(r'chat_style-([a-z\-]*)\.css', lines[0]):
        base_style = match.group(1)
        chat_styles[key] = chat_styles.get(base_style, '') + '\n\n' + '\n'.join(lines[1:])


def fix_newlines(string):
    return re.sub(r"\n{3,}", "\n\n", string.replace('\n', '\n\n')).strip()


def replace_blockquote(match):
    return match.group().replace('\n', '\n> ').replace('\\begin{blockquote}', '').replace('\\end{blockquote}', '')


@functools.lru_cache(maxsize=4096)
def convert_to_markdown(string):
    try:
        # Blockquote and code replacements
        string = re.sub(r'(^|[\n])&gt;', r'\1>', string)
        string = re.sub(r'\\begin{blockquote}(.*?)\\end{blockquote}', replace_blockquote, string, flags=re.DOTALL)
        string = string.replace('\\begin{code}', '```').replace('\\end{code}', '```')
        for env in ['align*', 'align', 'equation', 'equation*']:
            string = string.replace(f'\\begin{{{env}}}', '$$').replace(f'\\end{{{env}}}', '$$')
        string = re.sub(r"(.)```", r"\1\n```", string)

        # Process lines and handle unfinished code blocks
        result = ''
        is_code = False
        for line in string.split('\n'):
            is_code = not is_code if line.lstrip(' ').startswith('```') else is_code
            result += line + ('\n' if is_code or line.startswith('|') else '\n\n')
        result = result.strip()
        result += '\n```' if is_code else ''

        # Markdown conversion and unescape code blocks
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])
        html_output = re.sub(r'<code[^>]*>(.*?)</code>', lambda x: html.unescape(x.group()), html_output, flags=re.DOTALL)

        return html_output
    except Exception as e:
        print(f"Error in convert_to_markdown: {e}")
        return html.escape(string)


def convert_to_markdown_wrapped(string, use_cache=True):
    return convert_to_markdown(string) if use_cache else convert_to_markdown.__wrapped__(string)


def generate_basic_html(string):
    try:
        return f'<style>{readable_css}</style><div class="readable-container">{convert_to_markdown(string)}</div>'
    except Exception as e:
        print(f"Error in generate_basic_html: {e}")
        return html.escape(string)


def make_thumbnail(image):
    try:
        image = image.resize((350, round(image.size[1] / image.size[0] * 350)), Image.Resampling.LANCZOS)
        return ImageOps.fit(image, (350, 470), Image.LANCZOS) if image.size[1] > 470 else image
    except Exception as e:
        print(f"Error in make_thumbnail: {e}")
        return image


def get_image_cache(path):
    try:
        cache_folder = Path(shared.args.disk_cache_dir)
        cache_folder.mkdir(exist_ok=True)

        mtime = os.stat(path).st_mtime
        if path not in image_cache or image_cache[path][0] != mtime:
            img = make_thumbnail(Image.open(path))
            output_file = cache_folder / f'cache_{path.name}.png'
            img.convert('RGBA').save(output_file, format='PNG')
            image_cache[path] = [mtime, output_file.as_posix()]

        return image_cache[path][1]
    except Exception as e:
        print(f"Error in get_image_cache: {e}")
        return ""


def generate_instruct_html(history):
    try:
        messages = ''.join([
            f'<div class="user-message"><div class="text"><div class="message-body">{convert_to_markdown_wrapped(row[0], use_cache=i != len(history) - 1)}</div></div></div>'
            if row[0] else ''
            f'<div class="assistant-message"><div class="text"><div class="message-body">{convert_to_markdown_wrapped(row[1], use_cache=i != len(history) - 1)}</div></div></div>'
            for i, row in enumerate(history)
        ])
        return f'<style>{instruct_css}</style><div class="chat" id="chat"><div class="messages">{messages}</div></div>'
    except Exception as e:
        print(f"Error in generate_instruct_html: {e}")
        return ""


def generate_chat_html(history, name1, name2, style, character, reset_cache=False):
    try:
        img_bot = f'<img src="file/cache/pfp_character_thumb.png?{character}" class="pfp_character">' if Path("cache/pfp_character_thumb.png").exists() else ''
        img_me = f'<img src="file/cache/pfp_me.png?{time.time() if reset_cache else ""}">' if Path("cache/pfp_me.png").exists() else ''

        messages = ''.join([
            f'<div class="message"><div class="circle-you">{img_me}</div><div class="text"><div class="username">{name1}</div><div class="message-body">{convert_to_markdown_wrapped(row[0], use_cache=i != len(history) - 1)}</div></div></div>'
            if row[0] else ''
            f'<div class="message"><div class="circle-bot">{img_bot}</div><div class="text"><div class="username">{name2}</div><div class="message-body">{convert_to_markdown_wrapped(row[1], use_cache=i != len(history) - 1)}</div></div></div>'
            for i, row in enumerate(history)
        ])

        return f'<style>{chat_styles[style]}</style><div class="chat" id="chat"><div class="messages">{messages}</div></div>'
    except Exception as e:
        print(f"Error in generate_chat_html: {e}")
        return ""


def chat_html_wrapper(history, name1, name2, mode, style, character, reset_cache=False):
    try:
        if mode == 'instruct':
            return generate_instruct_html(history['visible'])
        return generate_chat_html(history['visible'], name1, name2, style, character, reset_cache)
    except Exception as e:
        print(f"Error in chat_html_wrapper: {e}")
        return ""
