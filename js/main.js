let main_parent = document.getElementById('chat-tab').parentNode;
let extensions = document.getElementById('extensions');

main_parent.childNodes[0].classList.add("header_bar");
main_parent.style = "padding: 0; margin: 0";
main_parent.parentNode.style = "gap: 0";
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
// Keyboard shortcuts
//------------------------------------------------
document.addEventListener("keydown", function(event) {

  // Stop generation on Esc pressed
  if (event.key === "Escape") {
    // Find the element with id 'stop' and click it
    var stopButton = document.getElementById("stop");
    if (stopButton) {
      stopButton.click();
    }
  }

  // Show chat controls on Ctrl+S pressed
  else if (event.ctrlKey && event.key == "s") {
    event.preventDefault();

    var showControlsElement = document.getElementById('show-controls');
    if (showControlsElement && showControlsElement.childNodes.length >= 4) {
      showControlsElement.childNodes[3].click();

      var arr = document.getElementById('chat-input').childNodes[2].childNodes;
      arr[arr.length - 1].focus();
    }
  }

});

//------------------------------------------------
// Position the chat typing dots
//------------------------------------------------
typing = document.getElementById('typing-container');
typingParent = typing.parentNode;
typingSibling = typing.previousElementSibling;
typingSibling.insertBefore(typing, typingSibling.childNodes[2]);

//------------------------------------------------
// Chat scrolling
//------------------------------------------------
const targetElement = document.getElementById('chat').parentNode.parentNode.parentNode;
targetElement.classList.add('pretty_scrollbar');
targetElement.classList.add('chat-parent');
let isScrolled = false;

targetElement.addEventListener('scroll', function() {
  let diff = targetElement.scrollHeight - targetElement.clientHeight;
  if(Math.abs(targetElement.scrollTop - diff) <= 1 || diff == 0) {
    isScrolled = false;
  } else {
    isScrolled = true;
  }
});

// Create a MutationObserver instance
const observer = new MutationObserver(function(mutations) {
  mutations.forEach(function(mutation) {
    if(!isScrolled) {
      targetElement.scrollTop = targetElement.scrollHeight;
    }

    const firstChild = targetElement.children[0];
    if (firstChild.classList.contains('generating')) {
      typing.parentNode.classList.add('visible-dots');
    } else {
      typing.parentNode.classList.remove('visible-dots');
    }

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
// Notebook box scrolling
//------------------------------------------------

const notebookElement = document.querySelector('#textbox-notebook textarea');
let notebookScrolled = false;

notebookElement.addEventListener('scroll', function() {
  let diff = notebookElement.scrollHeight - notebookElement.clientHeight;
  if(Math.abs(notebookElement.scrollTop - diff) <= 1 || diff == 0) {
    notebookScrolled = false;
  } else {
    notebookScrolled = true;
  }
});

const notebookObserver = new MutationObserver(function(mutations) {
  mutations.forEach(function(mutation) {
    if(!notebookScrolled) {
      notebookElement.scrollTop = notebookElement.scrollHeight;
    }
  });
});

notebookObserver.observe(notebookElement.parentNode.parentNode.parentNode, config);

//------------------------------------------------
// Default box scrolling
//------------------------------------------------

const defaultElement = document.querySelector('#textbox-default textarea');
let defaultScrolled = false;

defaultElement.addEventListener('scroll', function() {
  let diff = defaultElement.scrollHeight - defaultElement.clientHeight;
  if(Math.abs(defaultElement.scrollTop - diff) <= 1 || diff == 0) {
    defaultScrolled = false;
  } else {
    defaultScrolled = true;
  }
});

const defaultObserver = new MutationObserver(function(mutations) {
  mutations.forEach(function(mutation) {
    if(!defaultScrolled) {
      defaultElement.scrollTop = defaultElement.scrollHeight;
    }
  });
});

defaultObserver.observe(defaultElement.parentNode.parentNode.parentNode, config);

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
