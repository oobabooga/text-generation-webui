from pathlib import Path

import gradio as gr

from modules.html_generator import get_image_cache
from modules.shared import gradio


def load_js():
    js = """
    (() => {
        let galleryScript = document.createElement("script");
        galleryScript.src = "file/""" + Path('extensions/gallery/resources/gallery.js').as_posix() + """\";
        document.head.appendChild(galleryScript);
    })();
    """
    return js


def generate_html():
    cards = []
    # Iterate through files in image folder
    for file in sorted(Path("characters").glob("*")):
        if file.suffix in [".json", ".yml", ".yaml"]:
            character = file.stem
            container_html = f'<div class="character-container" item-data="{character}" onclick="checkActive(event)">'
            image_html = "<div class='placeholder'></div>"

            for path in [Path(f"characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
                if path.exists():
                    image_html = f'<img src="file/{get_image_cache(path)}"><div class="glow"></div>'
                    break

            container_html += f'{image_html} <div><span class="character-name">{character}</span></div>'
            container_html += "</div>"
            cards.append([container_html, character])

    return cards


def select_character(evt: gr.SelectData):
    return (evt.value[1])


def ui():
    with gr.Accordion("Character gallery", open=False):
        update = gr.Button("Refresh")
        gr.HTML(value=f"""<link rel='stylesheet' href='file/{Path('extensions/gallery/resources/gallery.css').as_posix()}'><img src onerror='{load_js()}'>""")
        gallery = gr.Dataset(components=[gr.HTML(visible=False)],
                             label="",
                             samples=generate_html(),
                             elem_classes=["character-gallery"],
                             samples_per_page=50
                             )
    update.click(generate_html, [], gallery)
    gallery.select(select_character, None, gradio['character_menu'])
