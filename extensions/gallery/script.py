from pathlib import Path

import gradio as gr

from modules.html_generator import image_to_base64


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

    .character-gallery tr {
      cursor: pointer;
    }

    .character-gallery .image-td {
      width: 150px;
    }

    .character-gallery .character-td {
      text-align: center !important;
    }

    """

    table_html = f'<style>{css}</style><div class="character-gallery"><table>'

    # Iterate through files in image folder
    for file in Path("characters").glob("*"):
        if file.name.endswith(".json"):
            character = file.name.replace(".json", "")
            table_html += f'<tr onclick=\'document.getElementById("character-menu").children[1].children[1].value = "{character}"; document.getElementById("character-menu").children[1].children[1].dispatchEvent(new Event("change"));\'>'
            image_html = "<div class='placeholder'></div>"

            for i in [
                    f"characters/{character}.png",
                    f"characters/{character}.jpg",
                    f"characters/{character}.jpeg",
                    ]:

                path = Path(i)
                if path.exists():
                    try:
                        image_html = f'<img src="data:image/png;base64,{image_to_base64(path)}">'
                        break
                    except:
                        continue

            table_html += f'<td class="image-td">{image_html}</td><td class="character-td">{character}</td>'
            table_html += "</tr>"
    table_html += "</table></div>"

    return table_html

def ui():
    with gr.Accordion("Character gallery"):
        update = gr.Button("Refresh")
        gallery = gr.HTML(value=generate_html())
    update.click(generate_html, [], gallery)
