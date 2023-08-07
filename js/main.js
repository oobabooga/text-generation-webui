document.getElementById("main").parentNode.childNodes[0].classList.add("header_bar");
document.getElementById("main").parentNode.style = "padding: 0; margin: 0";
document.getElementById("main").parentNode.parentNode.parentNode.style = "padding: 0";

// Get references to the elements
let main = document.getElementById('main');
let main_parent = main.parentNode;
let extensions = document.getElementById('extensions');

// Add an event listener to the main element
main_parent.addEventListener('click', function(e) {
    // Check if the main element is visible
    if (main.offsetHeight > 0 && main.offsetWidth > 0) {
        extensions.style.display = 'flex';
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
