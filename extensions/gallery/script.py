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
        perspective: 20px;
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
        transform: scale3d(1, 1, 1);
      }

      .character-container > div:has(.character-name) {
        position: relative;
        width: 100%;
      }

      .character-container > img.character-selected {
        transition-duration: 150ms;
        box-shadow: 0 4px 10px 4px #00000077;
        transform: scale3d(1.04, 1.04, 1.04)
      }

      .character-container > img.character-selected + div .character-name {
        transition-duration: 150ms;
        font-size: 1.05rem;
        top: 0.3rem;
      }

      .character-container .glow {
        position: absolute;
        width: 100%;
        height: 100%;
        left: 0;
        top: 0;
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

      function init() {
        let containers = document.querySelectorAll(".character-container")
        containers.forEach(container => {
          let inner = container.querySelector("img, div.placeholder");

          let counter = 0, updateRate = 5;
          let isTimeToUpdate = () => counter++ % updateRate === 0;

          let onMouseEnterHandler = event => update(event);
          let onMouseLeaveHandler = () => { inner.style = ""; container.querySelector(".glow").style.backgroundImage = ""; }
          let onMouseMoveHandler = event => isTimeToUpdate() ? update(event) : null;

          let update = function(event) {
            let x = ((-event.offsetY / container.offsetHeight) + 0.5).toFixed(2);
            let y = ((event.offsetX / container.offsetWidth) - 0.5).toFixed(2);
            inner.style.transform = "rotateX(" + x + "deg) rotateY(" + y + "deg) scale3d(1.03, 1.03, 1.03)";

            container.querySelector(".glow").style.backgroundImage = `radial-gradient(circle at ${event.offsetX}px ${event.offsetY}px, #ffffff55, #00000000 50%)`;
          };

          container.onmouseenter = onMouseEnterHandler;
          container.onmouseleave = onMouseLeaveHandler;
          container.onmousemove = onMouseMoveHandler;
        });
      }

      init();
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
        gr.HTML(value="<style>" + generate_css() + "</style>" + "<img src onerror='" + generate_js() + "'>")
        gallery = gr.Dataset(components=[gr.HTML(visible=False)],
                             label="",
                             samples=generate_html(),
                             elem_classes=["character-gallery"],
                             samples_per_page=50
                             )
    update.click(generate_html, [], gallery)
    gallery.select(select_character, None, gradio['character_menu'])
