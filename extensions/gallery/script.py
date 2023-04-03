from pathlib import Path

import gradio as gr

from modules.html_generator import get_image_cache
from modules.shared import gradio


def generate_css():
    css = """
      .character-gallery .gallery {
        margin: 1rem 0;
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        grid-column-gap: 0.4rem;
        grid-row-gap: 1.2rem;
        padding: 0;
      }

      .character-gallery > .label {
        display: none !important;
      }

      .character-gallery button.gallery-item {
        display: contents;
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
        font-size: 1rem;
        line-height: 1.2rem;
        font-weight: 600;
        overflow-wrap: anywhere;

        transition-duration: 300ms;
        transition-property: top font-size;
        transition-timing-function: ease-out;
        position: relative;
        top: 0rem;
        text-align: center;
        width: 100%;
      }

      .character-container > img {
        transition-duration: 300ms;
        transition-property: transform, box-shadow;
        transition-timing-function: ease-out;
        transform: rotate3d(1, 1, 1, 0deg) scale3d(1, 1, 1);
      }

      .character-container > div {
        position: relative;
        width: 100%;

      }

      .character-container > img:hover {
        transition-duration: 150ms;
        box-shadow: 0 5px 20px 5px #00000044;
        transform: rotate3d(1, 1, 1, 2deg);
      }

      .character-container > img.character-selected {
        transition-duration: 150ms;
        box-shadow: 0 5px 20px 5px #00000088;
        transform: scale3d(1.04, 1.04, 1.04)
      }

      .character-container > img.character-selected + div .character-name {
        transition-duration: 150ms;
        font-size: 1.05rem;
        top: 0.3rem;
      }
    """
    return css


def generate_js():
    js = """
    function loadjs() {
      document.querySelector("script#character-gallery-script")?.remove();

      var js = document.createElement("script");
      js.id = "character-gallery-script"
      js.textContent = `
        function checkActive(e) {
          let element = document.querySelector(".character-container > img.character-selected");
          element?.classList.remove("character-selected");
          e.currentTarget.querySelector("img").classList.add("character-selected");
        }
      `;
      document.head.appendChild(js);

      function update() {
        let element = document.querySelector(".character-container > img.character-selected");
        element?.classList.remove("character-selected");
        let currentCharacter = document.querySelector("div#character-menu span.single-select")?.innerText;
        document.querySelector(".character-container[item-data=\\""+currentCharacter+"\\"] img")?.classList.add("character-selected")
      }
      
      document.querySelectorAll("div.character-gallery div.paginate > button").forEach(
        e => e.addEventListener("click", update)
      )

      update();
      //alert("JS Inserted");
    }
    loadjs();
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

            container_html += f'{image_html} <div><span class="character-name">{character}</span></div>'
            container_html += "</div>"
            cards.append([container_html, character])

    return cards


def select_character(evt: gr.SelectData):
    return (evt.value[1])


def ui():
    with gr.Accordion("Character gallery", open=False):
        update = gr.Button("Refresh")
        gr.HTML(value="<style>"+generate_css()+"</style>"+f"<img src onerror='{generate_js()}'>")
        gallery = gr.Dataset(components=[gr.HTML(visible=False)],
            label="",
            samples=generate_html(),
            elem_classes=["character-gallery"],
            samples_per_page=50
        )
    update.click(generate_html, [], gallery)
    gallery.select(select_character, None, gradio['character_menu'])