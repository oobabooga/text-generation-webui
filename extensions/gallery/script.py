import json
import os
from pathlib import Path

import gradio as gr


def generate_html():
    css = """
    .character-gallery table {
      border-collapse: collapse;
      table-layout: fixed;
      width: 100%;
    }

    .character-gallery th, .character-gallery td {
      padding: 8px;
    }

    .character-gallery img {
      width: 150px;
      height: 200px;
      object-fit: cover;
    }

    .character-gallery .placeholder {
      width: 150px;
      height: 200px;
      background-color: gray;
    }

    .character-gallery td {
      text-align: center;
      vertical-align: middle;
    }
    """

    table_html = f'<style>{css}</style><div class="character-gallery"><table>'

    # Iterate through files in image folder
    for file in Path("characters").glob("*"):
        if file.name.endswith(".json"):
            json_name = file.name
            image_name = file.name.replace(".json", "")
            table_html += "<tr>"
            if Path(f"characters/{image_name}.png").exists():
                image_html = f'<img src="file/characters/{image_name}.png">'
            elif Path(f"characters/{image_name}.jpg").exists():
                image_html = f'<img src="file/characters/{image_name}.jpg">'
            else:
                image_html = "<div class='placeholder'></div>"
            table_html += f"<td>{image_html}</td><td>{image_name}</td>"
            table_html += "</tr>"
    table_html += "</table></div>"

    return table_html

def ui():
    with gr.Accordion("Character gallery"):
        update = gr.Button("Refresh")
        gallery = gr.HTML(value=generate_html())
    update.click(generate_html, [], gallery)
