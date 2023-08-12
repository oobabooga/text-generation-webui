let chat_tab = document.getElementById('chat-tab');
let notebook_tab = document.getElementById('notebook-tab');
let default_tab = document.getElementById('default-tab');

let main_parent = chat_tab.parentNode;
let extensions = document.getElementById('extensions');

main_parent.childNodes[0].classList.add("header_bar");
main_parent.style = "padding: 0; margin: 0";
main_parent.parentNode.parentNode.style = "padding: 0";

// Add an event listener to the generation tabs
main_parent.addEventListener('click', function(e) {
    let chat_visible = (chat_tab.offsetHeight > 0 && chat_tab.offsetWidth > 0);
    let notebook_visible = (notebook_tab.offsetHeight > 0 && notebook_tab.offsetWidth > 0);
    let default_visible = (default_tab.offsetHeight > 0 && default_tab.offsetWidth > 0);

    // Check if one of the generation tabs is visible
    if (chat_visible || notebook_visible || default_visible) {
        extensions.style.display = 'flex';
        if (chat_visible) {
            extensions.style.maxWidth = "800px";
            extensions.style.padding = "0px";
        } else {
            extensions.style.maxWidth = "none";
            extensions.style.padding = "15px";
        }
    } else {
        extensions.style.display = 'none';
    }
});

// Add some scrollbars
const textareaElements = document.querySelectorAll('.add_scrollbar textarea');
for(i = 0; i < textareaElements.length; i++) {
    textareaElements[i].classList.remove('scroll-hide');
    textareaElements[i].classList.add('pretty_scrollbar');
    textareaElements[i].style.resize = "none";
}

// Stop generation on Esc pressed
document.addEventListener("keydown", function(event) {
  if (event.key === "Escape") {
    // Find the element with id 'stop' and click it
    var stopButton = document.getElementById("stop");
    if (stopButton) {
      stopButton.click();
    }
  }
});
