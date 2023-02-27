'''

This is a library for formatting GPT-4chan and chat outputs as nice HTML.

'''

import os
import re
from pathlib import Path

from PIL import Image

# This is to store the paths to the thumbnails of the profile pictures
image_cache = {}

def generate_basic_html(s):
    css = """
    .container {
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        background-color: rgb(31, 41, 55);
        padding:3em;
    }
    .container p {
        font-size: 16px !important;
        color: white !important;
        margin-bottom: 22px;
        line-height: 1.4 !important;
    }
    """
    s = '\n'.join([f'<p>{line}</p>' for line in s.split('\n')])
    s = f'<style>{css}</style><div class="container">{s}</div>'
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
    css = """

    #parent #container {
        background-color: #eef2ff;
        padding: 17px;
    }
    #parent #container .reply {
        background-color: rgb(214, 218, 240);
        border-bottom-color: rgb(183, 197, 217);
        border-bottom-style: solid;
        border-bottom-width: 1px;
        border-image-outset: 0;
        border-image-repeat: stretch;
        border-image-slice: 100%;
        border-image-source: none;
        border-image-width: 1;
        border-left-color: rgb(0, 0, 0);
        border-left-style: none;
        border-left-width: 0px;
        border-right-color: rgb(183, 197, 217);
        border-right-style: solid;
        border-right-width: 1px;
        border-top-color: rgb(0, 0, 0);
        border-top-style: none;
        border-top-width: 0px;
        color: rgb(0, 0, 0);
        display: table;
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        margin-bottom: 4px;
        margin-left: 0px;
        margin-right: 0px;
        margin-top: 4px;
        overflow-x: hidden;
        overflow-y: hidden;
        padding-bottom: 4px;
        padding-left: 2px;
        padding-right: 2px;
        padding-top: 4px;
    }

    #parent #container .number {
        color: rgb(0, 0, 0);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        width: 342.65px;
        margin-right: 7px;
    }

    #parent #container .op {
        color: rgb(0, 0, 0);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        margin-bottom: 8px;
        margin-left: 0px;
        margin-right: 0px;
        margin-top: 4px;
        overflow-x: hidden;
        overflow-y: hidden;
    }

    #parent #container .op blockquote {
        margin-left: 0px !important;
    }

    #parent #container .name {
        color: rgb(17, 119, 67);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        font-weight: 700;
        margin-left: 7px;
    }

    #parent #container .quote {
        color: rgb(221, 0, 0);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        text-decoration-color: rgb(221, 0, 0);
        text-decoration-line: underline;
        text-decoration-style: solid;
        text-decoration-thickness: auto;
    }

    #parent #container .greentext {
        color: rgb(120, 153, 34);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
    }

    #parent #container blockquote {
        margin: 0px !important;
        margin-block-start: 1em;
        margin-block-end: 1em;
        margin-inline-start: 40px;
        margin-inline-end: 40px;
        margin-top: 13.33px !important;
        margin-bottom: 13.33px !important;
        margin-left: 40px !important;
        margin-right: 40px !important;
    }

    #parent #container .message {
        color: black;
        border: none;
    }
    """

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
    output += f'<style>{css}</style><div id="parent"><div id="container">'
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

def generate_chat_html(history, name1, name2, character):
    css = """
    .chat {
      margin-left: auto;
      margin-right: auto;
      max-width: 800px;
      height: 66.67vh;
      overflow-y: auto;
      padding-right: 20px;
      display: flex;
      flex-direction: column-reverse;
    }       

    .message {
      display: grid;
      grid-template-columns: 60px 1fr;
      padding-bottom: 25px;
      font-size: 15px;
      font-family: Helvetica, Arial, sans-serif;
      line-height: 1.428571429;
    }   
        
    .circle-you {
      width: 50px;
      height: 50px;
      background-color: rgb(238, 78, 59);
      border-radius: 50%;
    }
          
    .circle-bot {
      width: 50px;
      height: 50px;
      background-color: rgb(59, 78, 244);
      border-radius: 50%;
    }

    .circle-bot img, .circle-you img {
      border-radius: 50%;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .text {
    }

    .text p {
      margin-top: 5px;
    }

    .username {
      font-weight: bold;
    }

    .message-body {
    }

    .message-body img {
      max-width: 300px;
      max-height: 300px;
      border-radius: 20px;
    }

    .message-body p {
      margin-bottom: 0 !important;
      font-size: 15px !important;
      line-height: 1.428571429 !important;
    }

    .dark .message-body p em {
      color: rgb(138, 138, 138) !important;
    }

    .message-body p em {
      color: rgb(110, 110, 110) !important;
    }

    """

    output = ''
    output += f'<style>{css}</style><div class="chat" id="chat">'
    img = ''

    for i in [
            f"characters/{character}.png",
            f"characters/{character}.jpg",
            f"characters/{character}.jpeg",
            "img_bot.png",
            "img_bot.jpg",
            "img_bot.jpeg"
            ]:

        path = Path(i)
        if path.exists():
            img = f'<img src="file/{get_image_cache(path)}">'
            break

    img_me = ''
    for i in ["img_me.png", "img_me.jpg", "img_me.jpeg"]:
        path = Path(i)
        if path.exists():
            img_me = f'<img src="file/{get_image_cache(path)}">'
            break

    for i,_row in enumerate(history[::-1]):
        row = _row.copy()
        row[0] = re.sub(r"(\*\*)([^\*\n]*)(\*\*)", r"<b>\2</b>", row[0])
        row[1] = re.sub(r"(\*\*)([^\*\n]*)(\*\*)", r"<b>\2</b>", row[1])
        row[0] = re.sub(r"(\*)([^\*\n]*)(\*)", r"<em>\2</em>", row[0])
        row[1] = re.sub(r"(\*)([^\*\n]*)(\*)", r"<em>\2</em>", row[1])
        p = '\n'.join([f"<p>{x}</p>" for x in row[1].split('\n')])
        output += f"""
              <div class="message">
                <div class="circle-bot">
                  {img}
                </div>
                <div class="text">
                  <div class="username">
                    {name2}
                  </div>
                  <div class="message-body">
                    {p}
                  </div>
                </div>
              </div>
            """

        if not (i == len(history)-1 and len(row[0]) == 0):
            p = '\n'.join([f"<p>{x}</p>" for x in row[0].split('\n')])
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
                        {p}
                      </div>
                    </div>
                  </div>
                """

    output += "</div>"
    return output
