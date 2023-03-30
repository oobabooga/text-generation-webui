from pathlib import Path

import gradio as gr

from modules.html_generator import get_image_cache
import modules.shared as shared
import modules.chat as chat


def generate_html():
    # This is an ugly hack to deal with the preview window.
    # In theory, since we're using elem_id, it shouldn't mess with any other gallery previews in any other extensions.
    css = """
      div#extension_gallery div.preview {
        display: none !important;
      }
    """

    container_html = f'<style>{css}</style>'
    return container_html


def gallery_select(evt: gr.SelectData):
    result = chat.load_character(evt.value, shared.settings['name1'], shared.settings['name2'])
    return result + (gr.Dropdown.update(value=evt.value), gr.Gallery.update(value=build_gallery()), )


def build_gallery():
    imgs = [('extensions/gallery/no_image.png', 'None')]
    for file in sorted(Path("characters").glob("*")):
        if file.name.endswith(".json"):
            character = file.name.replace(".json", "")

            for i in [
                f"characters/{character}.png",
                f"characters/{character}.jpg",
                f"characters/{character}.jpeg",
            ]:

                path = Path(i)
                image_src = "extensions/gallery/no_image.png"
                if path.exists():
                    try:
                        image_src = get_image_cache(path)
                        break
                    except:
                        continue
            imgs.append((image_src, character))

    return imgs


def update_gallery():
    return gr.Gallery.update(value=build_gallery())


def ui():
    with gr.Accordion("Character gallery", open=False):
        update = gr.Button("Refresh")
        gallery = gr.HTML(value=generate_html())
        imgs = build_gallery()
        new_gallery = gr.Gallery(imgs, show_label=False, elem_id="extension_gallery")
    update.click(update_gallery, [], new_gallery)
    new_gallery.style(grid=5, height="100%")
    new_gallery.select(gallery_select, None,
                       [shared.gradio['name2'], shared.gradio['context'], shared.gradio['display'],
                        shared.gradio['character_menu'], new_gallery])
