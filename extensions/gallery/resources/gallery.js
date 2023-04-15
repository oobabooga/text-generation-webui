(() => {
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
      document.querySelector(".character-container[item-data=" + currentCharacter + "] img")?.classList.add("character-selected")
    }
  
    document.querySelectorAll("div.character-gallery div.paginate > button").forEach(
      e => e.addEventListener("click", update)
    )
  
    function init() {
      let containers = document.querySelectorAll(".character-container")
      containers.forEach(container => {
        let inner = container.querySelector("img, div.placeholder");
  
        let counter = 0, updateRate = 2;
        let isTimeToUpdate = () => counter++ % updateRate === 0;
  
        let onMouseEnterHandler = event => update(event);
        let onMouseLeaveHandler = () => { inner.style = ""; container.querySelector(".glow").style.backgroundImage = ""; }
        let onMouseMoveHandler = event => isTimeToUpdate() ? update(event) : null;
  
        let update = function (event) {
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
  })();
  