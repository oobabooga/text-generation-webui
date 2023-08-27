let main_parent = document.getElementById('chat-tab').parentNode;
let extensions = document.getElementById('extensions');

main_parent.childNodes[0].classList.add("header_bar");
main_parent.style = "padding: 0; margin: 0";
main_parent.parentNode.parentNode.style = "padding: 0";

document.querySelector('.header_bar').addEventListener('click', function(event) {
    if (event.target.tagName === 'BUTTON') {
        const buttonText = event.target.textContent.trim();

        let chat_visible = (buttonText == 'Chat');
        let default_visible = (buttonText == 'Default');
        let notebook_visible = (buttonText == 'Notebook');

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
    }
});

//------------------------------------------------
// Add some scrollbars
//------------------------------------------------
const textareaElements = document.querySelectorAll('.add_scrollbar textarea');
for(i = 0; i < textareaElements.length; i++) {
    textareaElements[i].classList.remove('scroll-hide');
    textareaElements[i].classList.add('pretty_scrollbar');
    textareaElements[i].style.resize = "none";
}

//------------------------------------------------
// Stop generation on Esc pressed
//------------------------------------------------
document.addEventListener("keydown", function(event) {
  if (event.key === "Escape") {
    // Find the element with id 'stop' and click it
    var stopButton = document.getElementById("stop");
    if (stopButton) {
      stopButton.click();
    }
  }
});

//------------------------------------------------
// Chat scrolling
//------------------------------------------------
const targetElement = document.getElementById('chat').parentNode.parentNode.parentNode;

// Create a MutationObserver instance
const observer = new MutationObserver(function(mutations) {
  mutations.forEach(function(mutation) {
    let childElement = targetElement.childNodes[2].childNodes[0].childNodes[1];
    childElement.scrollTop = childElement.scrollHeight;
  });
});

// Configure the observer to watch for changes in the subtree and attributes
const config = {
  childList: true,
  subtree: true,
  characterData: true,
  attributeOldValue: true,
  characterDataOldValue: true
};

// Start observing the target element
observer.observe(targetElement, config);

//------------------------------------------------
// Improve the looks of the chat input field
//------------------------------------------------
document.getElementById('chat-input').parentNode.style.background = 'transparent';
document.getElementById('chat-input').parentNode.style.border = 'none';

//------------------------------------------------
// Remove some backgrounds
//------------------------------------------------
const noBackgroundelements = document.querySelectorAll('.no-background');
for(i = 0; i < noBackgroundelements.length; i++) {
    noBackgroundelements[i].parentNode.style.border = 'none';
    noBackgroundelements[i].parentNode.parentNode.parentNode.style.alignItems = 'center';
}
