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
                extensions.style.maxWidth = "880px";
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

  // Show chat controls on Ctrl + S
  else if (event.ctrlKey && event.key == "s") {
    event.preventDefault();

    var showControlsElement = document.getElementById('show-controls');
    if (showControlsElement && showControlsElement.childNodes.length >= 4) {
      showControlsElement.childNodes[3].click();

      var arr = document.getElementById('chat-input').childNodes[2].childNodes;
      arr[arr.length - 1].focus();
    }
  }

  // Regenerate on Ctrl + Enter
  else if (event.ctrlKey && event.key === 'Enter') {
    event.preventDefault();
    document.getElementById('Regenerate').click();
  }

  // Continue on Alt + Enter
  else if (event.altKey && event.key === 'Enter') {
    event.preventDefault();
    document.getElementById('Continue').click();
  }

  // Remove last on Ctrl + Shift + Backspace
  else if (event.ctrlKey && event.shiftKey && event.key === 'Backspace') {
    event.preventDefault();
    document.getElementById('Remove-last').click();
  }

  // Copy last on Ctrl + Shift + K
  else if (event.ctrlKey && event.shiftKey && event.key === 'K') {
    event.preventDefault();
    document.getElementById('Copy-last').click();
  }

  // Replace last on Ctrl + Shift + L
  else if (event.ctrlKey && event.shiftKey && event.key === 'L') {
    event.preventDefault();
    document.getElementById('Replace-last').click();
  }

  // Impersonate on Ctrl + Shift + M
  else if (event.ctrlKey && event.shiftKey && event.key === 'M') {
    event.preventDefault();
    document.getElementById('Impersonate').click();
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
  if(Math.abs(targetElement.scrollTop - diff) <= 10 || diff == 0) {
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
      document.getElementById('stop').style.display = 'flex';
      document.getElementById('Generate').style.display = 'none';
    } else {
      typing.parentNode.classList.remove('visible-dots');
      document.getElementById('stop').style.display = 'none';
      document.getElementById('Generate').style.display = 'flex';
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
  if(Math.abs(notebookElement.scrollTop - diff) <= 10 || diff == 0) {
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
  if(Math.abs(defaultElement.scrollTop - diff) <= 10 || diff == 0) {
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
// Remove some backgrounds
//------------------------------------------------
const noBackgroundelements = document.querySelectorAll('.no-background');
for(i = 0; i < noBackgroundelements.length; i++) {
    noBackgroundelements[i].parentNode.style.border = 'none';
    noBackgroundelements[i].parentNode.parentNode.parentNode.style.alignItems = 'center';
}

//------------------------------------------------
// Create the hover menu in the chat tab
// The show/hide events were adapted from:
// https://github.com/SillyTavern/SillyTavern/blob/6c8bd06308c69d51e2eb174541792a870a83d2d6/public/script.js
//------------------------------------------------
var buttonsInChat = document.querySelectorAll("#chat-tab:not(.old-ui) #chat-buttons button");
var button = document.getElementById('hover-element-button');
var menu = document.getElementById('hover-menu');

function showMenu() {
    menu.style.display = 'flex'; // Show the menu
}

function hideMenu() {
    menu.style.display = 'none'; // Hide the menu
}

if (buttonsInChat.length > 0) {
    for (let i = buttonsInChat.length - 1; i >= 0; i--) {
        const thisButton = buttonsInChat[i];
        menu.appendChild(thisButton);

        if(i != 8) {
            thisButton.addEventListener("click", () => {
                hideMenu();
            });
        }

        const buttonText = thisButton.textContent;
        const matches = buttonText.match(/(\(.*?\))/);

        if (matches && matches.length > 1) {
            // Apply the transparent-substring class to the matched substring
            const substring = matches[1];
            const newText = buttonText.replace(substring, `&nbsp;<span class="transparent-substring">${substring.slice(1, -1)}</span>`);
            thisButton.innerHTML = newText;
        }
    }
} else {
    buttonsInChat = document.querySelectorAll("#chat-tab.old-ui #chat-buttons button");
    for (let i = 0; i < buttonsInChat.length; i++) {
        buttonsInChat[i].textContent = buttonsInChat[i].textContent.replace(/ \(.*?\)/, '');
    }
    document.getElementById('gr-hover-container').style.display = 'none';
}

function isMouseOverButtonOrMenu() {
    return menu.matches(':hover') || button.matches(':hover');
}

button.addEventListener('mouseenter', function () {
    showMenu();
});

button.addEventListener('click', function () {
    showMenu();
});

// Add event listener for mouseleave on the button
button.addEventListener('mouseleave', function () {
    // Delay to prevent menu hiding when the mouse leaves the button into the menu
    setTimeout(function () {
        if (!isMouseOverButtonOrMenu()) {
            hideMenu();
        }
    }, 100);
});

// Add event listener for mouseleave on the menu
menu.addEventListener('mouseleave', function () {
    // Delay to prevent menu hide when the mouse leaves the menu into the button
    setTimeout(function () {
        if (!isMouseOverButtonOrMenu()) {
            hideMenu();
        }
    }, 100);
});

// Add event listener for click anywhere in the document
document.addEventListener('click', function (event) {
    // Check if the click is outside the button/menu and the menu is visible
    if (!isMouseOverButtonOrMenu() && menu.style.display === 'flex') {
        hideMenu();
    }
});

//------------------------------------------------
// Relocate the "Show controls" checkbox
//------------------------------------------------
var elementToMove = document.getElementById('show-controls');
var parent = elementToMove.parentNode;
for (var i = 0; i < 2; i++) {
  parent = parent.parentNode;
}

parent.insertBefore(elementToMove, parent.firstChild);

//------------------------------------------------
// Make the chat input grow upwards instead of downwards
//------------------------------------------------
document.getElementById('show-controls').parentNode.style.position = 'absolute';
document.getElementById('show-controls').parentNode.style.bottom = '0px';

//------------------------------------------------
// Focus on the chat input
//------------------------------------------------
document.querySelector('#chat-input textarea').focus()
