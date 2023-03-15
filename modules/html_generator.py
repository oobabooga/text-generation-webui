'''

This is a library for formatting GPT-4chan and chat outputs as nice HTML.

'''

import os
import re
from pathlib import Path

import markdown
from PIL import Image

# This is to store the paths to the thumbnails of the profile pictures
image_cache = {}

with open(Path(__file__).resolve().parent / '../css/html_readable_style.css', 'r') as f:
    readable_css = f.read()
with open(Path(__file__).resolve().parent / '../css/html_4chan_style.css', 'r') as css_f:
    _4chan_css = css_f.read()
with open(Path(__file__).resolve().parent / '../css/html_cai_style.css', 'r') as f:
    cai_css = f.read()

def generate_basic_html(s):
    s = '\n'.join([f'<p>{line}</p>' for line in s.split('\n')])
    s = f'<style>{readable_css}</style><div class="container">{s}</div>'
    return s

def process_post(post, c):
    t = post.split('\n')
    number = t[0].split(' ')[1]
    if len(t) > 1:
        src = '\n'.join(t[1:])
    else:
        src = ''
    src = re.sub('>', '&gt;', src)
    src = re.sub('(&gt;&gt;[0-9]*)', '<span class="quote">\\1</span>', src)
    src = re.sub('\n', '<br>\n', src)
    src = f'<blockquote class="message">{src}\n'
    src = f'<span class="name">Anonymous </span> <span class="number">No.{number}</span>\n{src}'
    return src

def generate_4chan_html(f):
    posts = []
    post = ''
    c = -2
    for line in f.splitlines():
        line += "\n"
        if line == '-----\n':
            continue
        elif line.startswith('--- '):
            c += 1
            if post != '':
                src = process_post(post, c)
                posts.append(src)
            post = line
        else:
            post += line
    if post != '':
        src = process_post(post, c)
        posts.append(src)

    for i in range(len(posts)):
        if i == 0:
            posts[i] = f'<div class="op">{posts[i]}</div>\n'
        else:
            posts[i] = f'<div class="reply">{posts[i]}</div>\n'
    
    output = ''
    output += f'<style>{_4chan_css}</style><div id="parent"><div id="container">'
    for post in posts:
        output += post
    output += '</div></div>'
    output = output.split('\n')
    for i in range(len(output)):
        output[i] = re.sub(r'^(&gt;(.*?)(<br>|</div>))', r'<span class="greentext">\1</span>', output[i])
        output[i] = re.sub(r'^<blockquote class="message">(&gt;(.*?)(<br>|</div>))', r'<blockquote class="message"><span class="greentext">\1</span>', output[i])
    output = '\n'.join(output)

    return output

def get_image_cache(path):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    mtime = os.stat(path).st_mtime
    if (path in image_cache and mtime != image_cache[path][0]) or (path not in image_cache):
        img = Image.open(path)
        img.thumbnail((200, 200))
        output_file = Path(f'cache/{path.name}_cache.png')
        img.convert('RGB').save(output_file, format='PNG')
        image_cache[path] = [mtime, output_file.as_posix()]

    return image_cache[path][1]

def load_html_image(paths):
    for str_path in paths:
          path = Path(str_path)
          if path.exists():
              return f'<img src="file/{get_image_cache(path)}">'
    return ''

def generate_chat_html(history, name1, name2, character):
    output = f'<style>{cai_css}</style><div class="chat" id="chat">'
    
    img_bot = load_html_image([f"characters/{character}.{ext}" for ext in ['png', 'jpg', 'jpeg']] + ["img_bot.png","img_bot.jpg","img_bot.jpeg"])
    img_me = load_html_image(["img_me.png", "img_me.jpg", "img_me.jpeg"])

    for i,_row in enumerate(history[::-1]):
        row = [markdown.markdown(re.sub(r"(.)```", r"\1\n```", entry), extensions=['fenced_code']) for entry in _row]
        
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

        if not (i == len(history)-1 and len(row[0]) == 0):
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

    output += "</div>"
    return output
