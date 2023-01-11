'''

This is a library for formatting gpt4chan outputs as nice HTML.

'''

import re

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
    #container {
        background-color: #eef2ff;
        padding: 17px;
    }
    .reply {
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
        padding-bottom: 2px;
        padding-left: 2px;
        padding-right: 2px;
        padding-top: 2px;
    }

    .number {
        color: rgb(0, 0, 0);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        width: 342.65px;
    }

    .op {
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

    .op blockquote {
        margin-left:7px;
    }

    .name {
        color: rgb(17, 119, 67);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        font-weight: 700;
        margin-left: 7px;
    }

    .quote {
        color: rgb(221, 0, 0);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
        text-decoration-color: rgb(221, 0, 0);
        text-decoration-line: underline;
        text-decoration-style: solid;
        text-decoration-thickness: auto;
    }

    .greentext {
        color: rgb(120, 153, 34);
        font-family: arial, helvetica, sans-serif;
        font-size: 13.3333px;
    }

    blockquote {
        margin-block-start: 1em;
        margin-block-end: 1em;
        margin-inline-start: 40px;
        margin-inline-end: 40px;
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
    output += f'<style>{css}</style><div id="container">'
    for post in posts:
        output += post
    output += '</div>'
    output = output.split('\n')
    for i in range(len(output)):
        output[i] = re.sub(r'^(&gt;(.*?)(<br>|</div>))', r'<span class="greentext">\1</span>', output[i])
        output[i] = re.sub(r'^<blockquote class="message">(&gt;(.*?)(<br>|</div>))', r'<blockquote class="message"><span class="greentext">\1</span>', output[i])
    output = '\n'.join(output)

    return output
