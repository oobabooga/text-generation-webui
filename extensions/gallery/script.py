from pathlib import Path

import gradio as gr

from modules.html_generator import get_image_cache


def generate_html():
    css = """
      .character-gallery {
        margin: 1rem 0;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        grid-column-gap: 0.4rem;
        grid-row-gap: 1.2rem;
      }

      .character-container {
        cursor: pointer;
        text-align: center;
        position: relative;
        opacity: 0.85;
      }

      .character-container:hover {
        opacity: 1;
      }

      .character-container .placeholder, .character-container img {
        width: 150px;
        height: 200px;
        background-color: gray;
        object-fit: cover;
        margin: 0 auto;
        border-radius: 1rem;
        border: 3px solid white;
        box-shadow: 3px 3px 6px 0px rgb(0 0 0 / 50%);
      }

      .character-name {
        margin-top: 0.3rem;
        display: block;
        font-size: 1.2rem;
        font-weight: 600;
        overflow-wrap: anywhere;
      }
    """

    container_html = f'<style>{css}</style><div class="character-gallery">'

    # Iterate through files in image folder
    for file in sorted(Path("characters").glob("*")):
        if file.suffix in [".json", ".yml", ".yaml"]:
            character = file.name.replace(file.suffix, "")
            container_html += f'<div class="character-container" onclick=\'document.getElementById("character-menu").children[1].children[1].value = "{character}"; document.getElementById("character-menu").children[1].children[1].dispatchEvent(new Event("change"));\'>'
            image_html = "<div class='placeholder'></div>"

            for i in [
                    f"characters/{character}.png",
                    f"characters/{character}.jpg",
                    f"characters/{character}.jpeg",
                    ]:

                path = Path(i)
                if path.exists():
                    try:
                        image_html = f'<img src="file/{get_image_cache(path)}">'
                        break
                    except:
                        continue

            container_html += f'{image_html} <span class="character-name">{character}</span>'
            container_html += "</div>"

    container_html += "</div>"
    return container_html

def ui():
    with gr.Accordion("Character gallery", open=False):
        update = gr.Button("Refresh")
        gallery = gr.HTML(value=generate_html())
    update.click(generate_html, [], gallery)
