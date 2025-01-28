let main_parent = document.getElementById("chat-tab").parentNode;
let extensions = document.getElementById("extensions");

main_parent.childNodes[0].classList.add("header_bar");
main_parent.style = "padding: 0; margin: 0";
main_parent.parentNode.style = "gap: 0";
main_parent.parentNode.parentNode.style = "padding: 0";

document.querySelector(".header_bar").addEventListener("click", function(event) {
  if (event.target.tagName !== "BUTTON") return;

  const buttonText = event.target.textContent.trim();
  const extensionsVisible = ["Chat", "Default", "Notebook"].includes(buttonText);
  const chatVisible = buttonText === "Chat";
  const showControlsChecked = document.querySelector("#show-controls input").checked;
  const extensions = document.querySelector("#extensions");

  if (extensionsVisible) {
    if (extensions) {
      extensions.style.display = "flex";
    }

    this.style.marginBottom = chatVisible ? "0px" : "19px";

    if (chatVisible && !showControlsChecked) {
      document.querySelectorAll(
        "#chat-tab > div > :nth-child(1), #chat-tab > div > :nth-child(3), #chat-tab > div > :nth-child(4), #extensions"
      ).forEach(element => {
        element.style.display = "none";
      });
    }

  } else {
    this.style.marginBottom = "19px";
    if (extensions) extensions.style.display = "none";
  }
});

//------------------------------------------------
// Keyboard shortcuts
//------------------------------------------------
let previousTabId = "chat-tab-button";
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

    var showControlsElement = document.getElementById("show-controls");
    if (showControlsElement && showControlsElement.childNodes.length >= 4) {
      showControlsElement.childNodes[3].click();

      var arr = document.getElementById("chat-input").childNodes[2].childNodes;
      arr[arr.length - 1].focus();
    }
  }

  // Regenerate on Ctrl + Enter
  else if (event.ctrlKey && event.key === "Enter") {
    event.preventDefault();
    document.getElementById("Regenerate").click();
  }

  // Continue on Alt + Enter
  else if (event.altKey && event.key === "Enter") {
    event.preventDefault();
    document.getElementById("Continue").click();
  }

  // Remove last on Ctrl + Shift + Backspace
  else if (event.ctrlKey && event.shiftKey && event.key === "Backspace") {
    event.preventDefault();
    document.getElementById("Remove-last").click();
  }

  // Copy last on Ctrl + Shift + K
  else if (event.ctrlKey && event.shiftKey && event.key === "K") {
    event.preventDefault();
    document.getElementById("Copy-last").click();
  }

  // Replace last on Ctrl + Shift + L
  else if (event.ctrlKey && event.shiftKey && event.key === "L") {
    event.preventDefault();
    document.getElementById("Replace-last").click();
  }

  // Impersonate on Ctrl + Shift + M
  else if (event.ctrlKey && event.shiftKey && event.key === "M") {
    event.preventDefault();
    document.getElementById("Impersonate").click();
  }

});

//------------------------------------------------
// Position the chat typing dots
//------------------------------------------------
typing = document.getElementById("typing-container");
typingParent = typing.parentNode;
typingSibling = typing.previousElementSibling;
typingSibling.insertBefore(typing, typingSibling.childNodes[2]);

//------------------------------------------------
// Chat scrolling
//------------------------------------------------
const targetElement = document.getElementById("chat").parentNode.parentNode.parentNode;
targetElement.classList.add("pretty_scrollbar");
targetElement.classList.add("chat-parent");
let isScrolled = false;

targetElement.addEventListener("scroll", function() {
  let diff = targetElement.scrollHeight - targetElement.clientHeight;
  if(Math.abs(targetElement.scrollTop - diff) <= 10 || diff == 0) {
    isScrolled = false;
  } else {
    isScrolled = true;
  }

  doSyntaxHighlighting();

});

// Create a MutationObserver instance
const observer = new MutationObserver(function(mutations) {
  updateCssProperties();

  if (targetElement.classList.contains("_generating")) {
    typing.parentNode.classList.add("visible-dots");
    document.getElementById("stop").style.display = "flex";
    document.getElementById("Generate").style.display = "none";
  } else {
    typing.parentNode.classList.remove("visible-dots");
    document.getElementById("stop").style.display = "none";
    document.getElementById("Generate").style.display = "flex";
  }


  doSyntaxHighlighting();

  if (!isScrolled && targetElement.scrollTop !== targetElement.scrollHeight) {
    targetElement.scrollTop = targetElement.scrollHeight;
  }
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
// Handle syntax highlighting / LaTeX
//------------------------------------------------
function isElementVisibleOnScreen(element) {
  const rect = element.getBoundingClientRect();
  return (
    rect.left < window.innerWidth &&
    rect.right > 0 &&
    rect.top < window.innerHeight &&
    rect.bottom > 0
  );
}

function doSyntaxHighlighting() {
  const messageBodies = document.querySelectorAll(".message-body");

  if (messageBodies.length > 0) {
    observer.disconnect();

    messageBodies.forEach((messageBody) => {
      if (isElementVisibleOnScreen(messageBody)) {
        // Handle both code and math in a single pass through each message
        const codeBlocks = messageBody.querySelectorAll("pre code:not([data-highlighted])");
        codeBlocks.forEach((codeBlock) => {
          hljs.highlightElement(codeBlock);
          codeBlock.setAttribute("data-highlighted", "true");
        });

        renderMathInElement(messageBody, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true },
          ],
        });
      }
    });

    observer.observe(targetElement, config);
  }
}

//------------------------------------------------
// Add some scrollbars
//------------------------------------------------
const textareaElements = document.querySelectorAll(".add_scrollbar textarea");
for(i = 0; i < textareaElements.length; i++) {
  textareaElements[i].classList.remove("scroll-hide");
  textareaElements[i].classList.add("pretty_scrollbar");
  textareaElements[i].style.resize = "none";
}

//------------------------------------------------
// Remove some backgrounds
//------------------------------------------------
const noBackgroundelements = document.querySelectorAll(".no-background");
for(i = 0; i < noBackgroundelements.length; i++) {
  noBackgroundelements[i].parentNode.style.border = "none";
  noBackgroundelements[i].parentNode.parentNode.parentNode.style.alignItems = "center";
}

const slimDropdownElements = document.querySelectorAll(".slim-dropdown");
for (i = 0; i < slimDropdownElements.length; i++) {
  const parentNode = slimDropdownElements[i].parentNode;
  parentNode.style.background = "transparent";
  parentNode.style.border = "0";
}

//------------------------------------------------
// Create the hover menu in the chat tab
// The show/hide events were adapted from:
// https://github.com/SillyTavern/SillyTavern/blob/6c8bd06308c69d51e2eb174541792a870a83d2d6/public/script.js
//------------------------------------------------
var buttonsInChat = document.querySelectorAll("#chat-tab #chat-buttons button");
var button = document.getElementById("hover-element-button");
var menu = document.getElementById("hover-menu");
var istouchscreen = (navigator.maxTouchPoints > 0) || "ontouchstart" in document.documentElement;

function showMenu() {
  menu.style.display = "flex"; // Show the menu
}

function hideMenu() {
  menu.style.display = "none"; // Hide the menu
  if (!istouchscreen) {
    document.querySelector("#chat-input textarea").focus(); // Focus on the chat input
  }
}

if (buttonsInChat.length > 0) {
  for (let i = buttonsInChat.length - 1; i >= 0; i--) {
    const thisButton = buttonsInChat[i];
    menu.appendChild(thisButton);

    thisButton.addEventListener("click", () => {
      hideMenu();
    });

    const buttonText = thisButton.textContent;
    const matches = buttonText.match(/(\(.*?\))/);

    if (matches && matches.length > 1) {
      // Apply the transparent-substring class to the matched substring
      const substring = matches[1];
      const newText = buttonText.replace(substring, `&nbsp;<span class="transparent-substring">${substring.slice(1, -1)}</span>`);
      thisButton.innerHTML = newText;
    }
  }
}

function isMouseOverButtonOrMenu() {
  return menu.matches(":hover") || button.matches(":hover");
}

button.addEventListener("mouseenter", function () {
  if (!istouchscreen) {
    showMenu();
  }
});

button.addEventListener("click", function () {
  if (menu.style.display === "flex") {
    hideMenu();
  }
  else {
    showMenu();
  }
});

// Add event listener for mouseleave on the button
button.addEventListener("mouseleave", function () {
  // Delay to prevent menu hiding when the mouse leaves the button into the menu
  setTimeout(function () {
    if (!isMouseOverButtonOrMenu()) {
      hideMenu();
    }
  }, 100);
});

// Add event listener for mouseleave on the menu
menu.addEventListener("mouseleave", function () {
  // Delay to prevent menu hide when the mouse leaves the menu into the button
  setTimeout(function () {
    if (!isMouseOverButtonOrMenu()) {
      hideMenu();
    }
  }, 100);
});

// Add event listener for click anywhere in the document
document.addEventListener("click", function (event) {
  const target = event.target;

  // Check if the click is outside the button/menu and the menu is visible
  if (!isMouseOverButtonOrMenu() && menu.style.display === "flex") {
    hideMenu();
  }

  if (event.target.classList.contains("pfp_character")) {
    toggleBigPicture();
  }

  // Handle sidebar clicks on mobile
  if (isMobile()) {
  // Check if the click did NOT originate from any of the specified toggle buttons or elements
    if (
      target.closest("#navigation-toggle") !== navigationToggle &&
    target.closest("#past-chats-toggle") !== pastChatsToggle &&
    target.closest("#chat-controls-toggle") !== chatControlsToggle &&
    target.closest(".header_bar") !== headerBar &&
    target.closest("#past-chats-row") !== pastChatsRow &&
    target.closest("#chat-controls") !== chatControlsRow
    ) {
      handleIndividualSidebarClose(event);
    }
  }
});

//------------------------------------------------
// Relocate the "Show controls" checkbox
//------------------------------------------------
var elementToMove = document.getElementById("show-controls");
var parent = elementToMove.parentNode;
for (var i = 0; i < 2; i++) {
  parent = parent.parentNode;
}

parent.insertBefore(elementToMove, parent.firstChild);

//------------------------------------------------
// Position the chat input
//------------------------------------------------
document.getElementById("show-controls").parentNode.classList.add("chat-input-positioned");

//------------------------------------------------
// Focus on the chat input
//------------------------------------------------
const chatTextArea = document.getElementById("chat-input").querySelector("textarea");

function respondToChatInputVisibility(element, callback) {
  var options = {
    root: document.documentElement,
  };

  var observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      callback(entry.intersectionRatio > 0);
    });
  }, options);

  observer.observe(element);
}

function handleChatInputVisibilityChange(isVisible) {
  if (isVisible) {
    chatTextArea.focus();
  }
}

respondToChatInputVisibility(chatTextArea, handleChatInputVisibilityChange);

//------------------------------------------------
// Show enlarged character picture when the profile
// picture is clicked on
//------------------------------------------------
let bigPictureVisible = false;

function addBigPicture() {
  var imgElement = document.createElement("img");
  var timestamp = new Date().getTime();
  imgElement.src = "/file/cache/pfp_character.png?time=" + timestamp;
  imgElement.classList.add("bigProfilePicture");
  imgElement.addEventListener("load", function () {
    this.style.visibility = "visible";
  });
  imgElement.addEventListener("error", function () {
    this.style.visibility = "hidden";
  });

  var imgElementParent = document.getElementById("chat").parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
  imgElementParent.appendChild(imgElement);
}

function deleteBigPicture() {
  var bigProfilePictures = document.querySelectorAll(".bigProfilePicture");
  bigProfilePictures.forEach(function (element) {
    element.parentNode.removeChild(element);
  });
}

function toggleBigPicture() {
  if(bigPictureVisible) {
    deleteBigPicture();
    bigPictureVisible = false;
  } else {
    addBigPicture();
    bigPictureVisible = true;
  }
}

//------------------------------------------------
// Handle the chat input box growth
//------------------------------------------------

// Cache DOM elements
const chatContainer = document.getElementById("chat").parentNode.parentNode.parentNode;
const chatInput = document.querySelector("#chat-input textarea");

// Variables to store current dimensions
let currentChatInputHeight = chatInput.clientHeight;

// Update chat layout based on chat and input dimensions
function updateCssProperties() {
  const chatInputHeight = chatInput.clientHeight;

  // Check if the chat container is visible
  if (chatContainer.clientHeight > 0) {
    const chatContainerParentHeight = chatContainer.parentNode.clientHeight;
    const newChatHeight = `${chatContainerParentHeight - chatInputHeight - 80}px`;

    document.documentElement.style.setProperty("--chat-height", newChatHeight);
    document.documentElement.style.setProperty("--input-delta", `${chatInputHeight - 40}px`);

    // Adjust scrollTop based on input height change
    if (chatInputHeight !== currentChatInputHeight) {
      const deltaHeight = chatInputHeight - currentChatInputHeight;
      if (!isScrolled && deltaHeight < 0) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      } else {
        chatContainer.scrollTop += deltaHeight;
      }

      currentChatInputHeight = chatInputHeight;
    }
  }
}

// Observe textarea size changes and call update function
new ResizeObserver(updateCssProperties).observe(document.querySelector("#chat-input textarea"));

// Handle changes in window size
window.addEventListener("resize", updateCssProperties);

//------------------------------------------------
// Focus on the rename text area when it becomes visible
//------------------------------------------------
const renameTextArea = document.getElementById("rename-row").querySelector("textarea");

function respondToRenameVisibility(element, callback) {
  var options = {
    root: document.documentElement,
  };

  var observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      callback(entry.intersectionRatio > 0);
    });
  }, options);

  observer.observe(element);
}


function handleVisibilityChange(isVisible) {
  if (isVisible) {
    renameTextArea.focus();
  }
}

respondToRenameVisibility(renameTextArea, handleVisibilityChange);

//------------------------------------------------
// Adjust the chat tab margin if no extension UI
// is present at the bottom
//------------------------------------------------

if (document.getElementById("extensions") === null) {
  document.getElementById("chat-tab").style.marginBottom = "-29px";
}

//------------------------------------------------
// Focus on the chat input after starting a new chat
//------------------------------------------------

document.querySelectorAll(".focus-on-chat-input").forEach(element => {
  element.addEventListener("click", function() {
    document.querySelector("#chat-input textarea").focus();
  });
});

//------------------------------------------------
// Fix a border around the "past chats" menu
//------------------------------------------------
document.getElementById("past-chats").parentNode.style.borderRadius = "0px";

//------------------------------------------------
// Allow the character dropdown to coexist at the
// Chat tab and the Parameters > Character tab
//------------------------------------------------

const headerBar = document.querySelector(".header_bar");
let originalParent;
let originalIndex; // To keep track of the original position
let movedElement;

function moveToChatTab() {
  const characterMenu = document.getElementById("character-menu");
  const grandParent = characterMenu.parentElement.parentElement;

  // Save the initial location for the character dropdown
  if (!originalParent) {
    originalParent = grandParent.parentElement;
    originalIndex = Array.from(originalParent.children).indexOf(grandParent);
    movedElement = grandParent;
  }

  // Do not show the Character dropdown in the Chat tab when "instruct" mode is selected
  const instructRadio = document.querySelector("#chat-mode input[value=\"instruct\"]");
  if (instructRadio && instructRadio.checked) {
    grandParent.style.display = "none";
  }

  grandParent.children[0].style.minWidth = "100%";

  const chatControlsFirstChild = document.querySelector("#chat-controls").firstElementChild;
  const newParent = chatControlsFirstChild;
  let newPosition = newParent.children.length - 2;

  newParent.insertBefore(grandParent, newParent.children[newPosition]);
  document.getElementById("save-character").style.display = "none";
}

function restoreOriginalPosition() {
  if (originalParent && movedElement) {
    if (originalIndex >= originalParent.children.length) {
      originalParent.appendChild(movedElement);
    } else {
      originalParent.insertBefore(movedElement, originalParent.children[originalIndex]);
    }

    document.getElementById("save-character").style.display = "";
    movedElement.style.display = "";
    movedElement.children[0].style.minWidth = "";
  }
}

headerBar.addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") {
    const tabName = e.target.textContent.trim();
    if (tabName === "Chat") {
      moveToChatTab();
    } else {
      restoreOriginalPosition();
    }
  }
});

//------------------------------------------------
// Add a confirmation dialog when leaving the page
// Useful to avoid data loss
//------------------------------------------------
window.addEventListener("beforeunload", function (event) {
  // Cancel the event
  event.preventDefault();
  // Chrome requires returnValue to be set
  event.returnValue = "";
});

moveToChatTab();

//------------------------------------------------
// Buttons to toggle the sidebars
//------------------------------------------------

const leftArrowSVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-arrow-bar-left">
  <path d="M4 12l10 0"></path>
  <path d="M4 12l4 4"></path>
  <path d="M4 12l4 -4"></path>
  <path d="M20 4l0 16"></path>
</svg>`;

const rightArrowSVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-arrow-bar-right">
  <path d="M20 12l-10 0"></path>
  <path d="M20 12l-4 4"></path>
  <path d="M20 12l-4 -4"></path>
  <path d="M4 4l0 16"></path>
</svg>`;

const hamburgerMenuSVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-hamburger-menu">
  <line x1="3" y1="12" x2="21" y2="12"></line>
  <line x1="3" y1="6" x2="21" y2="6"></line>
  <line x1="3" y1="18" x2="21" y2="18"></line>
</svg>`;

const closeMenuSVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-close-menu">
  <line x1="18" y1="6" x2="6" y2="18"></line>
  <line x1="6" y1="6" x2="18" y2="18"></line>
</svg>`;

const chatTab = document.getElementById("chat-tab");
const pastChatsRow = document.getElementById("past-chats-row");
const chatControlsRow = document.getElementById("chat-controls");

if (chatTab) {
  // Create past-chats-toggle div
  const pastChatsToggle = document.createElement("div");
  pastChatsToggle.id = "past-chats-toggle";
  pastChatsToggle.innerHTML = leftArrowSVG; // Set initial icon to left arrow
  pastChatsToggle.classList.add("past-chats-open"); // Set initial position

  // Create chat-controls-toggle div
  const chatControlsToggle = document.createElement("div");
  chatControlsToggle.id = "chat-controls-toggle";
  chatControlsToggle.innerHTML = rightArrowSVG; // Set initial icon to right arrow
  chatControlsToggle.classList.add("chat-controls-open"); // Set initial position

  // Append both elements to the chat-tab
  chatTab.appendChild(pastChatsToggle);
  chatTab.appendChild(chatControlsToggle);
}

// Create navigation toggle div
const navigationToggle = document.createElement("div");
navigationToggle.id = "navigation-toggle";
navigationToggle.innerHTML = leftArrowSVG; // Set initial icon to right arrow
navigationToggle.classList.add("navigation-left"); // Set initial position
headerBar.appendChild(navigationToggle);

// Retrieve the dynamically created toggle buttons
const pastChatsToggle = document.getElementById("past-chats-toggle");
const chatControlsToggle = document.getElementById("chat-controls-toggle");

function handleIndividualSidebarClose(event) {
  const target = event.target;

  // Close navigation bar if click is outside and it is open
  if (!headerBar.contains(target) && !headerBar.classList.contains("sidebar-hidden")) {
    toggleSidebar(headerBar, navigationToggle, true);
  }

  // Close past chats row if click is outside and it is open
  if (!pastChatsRow.contains(target) && !pastChatsRow.classList.contains("sidebar-hidden")) {
    toggleSidebar(pastChatsRow, pastChatsToggle, true);
  }

  // Close chat controls row if click is outside and it is open
  if (!chatControlsRow.contains(target) && !chatControlsRow.classList.contains("sidebar-hidden")) {
    toggleSidebar(chatControlsRow, chatControlsToggle, true);
  }
}

function toggleSidebar(sidebar, toggle, forceClose = false) {
  const isCurrentlyHidden = sidebar.classList.contains("sidebar-hidden");
  const shouldClose = !isCurrentlyHidden;

  // Apply visibility classes
  sidebar.classList.toggle("sidebar-hidden", shouldClose);
  sidebar.classList.toggle("sidebar-shown", !shouldClose);

  if (sidebar === headerBar) {
    // Special handling for header bar
    document.documentElement.style.setProperty("--header-width", shouldClose ? "0px" : "112px");
    pastChatsRow.classList.toggle("negative-header", shouldClose);
    pastChatsToggle.classList.toggle("negative-header", shouldClose);
    toggle.innerHTML = shouldClose ? hamburgerMenuSVG : closeMenuSVG;
  } else if (sidebar === pastChatsRow) {
    // Past chats sidebar
    toggle.classList.toggle("past-chats-closed", shouldClose);
    toggle.classList.toggle("past-chats-open", !shouldClose);
    toggle.innerHTML = shouldClose ? rightArrowSVG : leftArrowSVG;
  } else if (sidebar === chatControlsRow) {
    // Chat controls sidebar
    toggle.classList.toggle("chat-controls-closed", shouldClose);
    toggle.classList.toggle("chat-controls-open", !shouldClose);
    toggle.innerHTML = shouldClose ? leftArrowSVG : rightArrowSVG;
  }

  // Mobile handling
  if (isMobile()) {
    sidebar.classList.toggle("sidebar-shown", !shouldClose);
  }
}

// Function to check if the device is mobile
function isMobile() {
  return window.innerWidth <= 924;
}

// Function to initialize sidebars
function initializeSidebars() {
  const isOnMobile = isMobile();
  
  if (isOnMobile) {
    // Mobile state: Hide sidebars and set closed states
    [pastChatsRow, chatControlsRow, headerBar].forEach(el => {
      el.classList.add("sidebar-hidden");
      el.classList.remove("sidebar-shown");
    });

    document.documentElement.style.setProperty("--header-width", "0px");
    pastChatsRow.classList.add("negative-header");
    pastChatsToggle.classList.add("negative-header", "past-chats-closed");
    pastChatsToggle.classList.remove("past-chats-open");

    [chatControlsToggle, navigationToggle].forEach(el => {
      el.classList.add("chat-controls-closed");
      el.classList.remove("chat-controls-open");
    });

    pastChatsToggle.innerHTML = rightArrowSVG;
    chatControlsToggle.innerHTML = leftArrowSVG;
    navigationToggle.innerHTML = hamburgerMenuSVG;
  } else {
    // Desktop state: Show sidebars and set open states
    [pastChatsRow, chatControlsRow].forEach(el => {
      el.classList.remove("sidebar-hidden", "sidebar-shown");
    });

    pastChatsToggle.classList.add("past-chats-open");
    pastChatsToggle.classList.remove("past-chats-closed");

    [chatControlsToggle, navigationToggle].forEach(el => {
      el.classList.add("chat-controls-open");
      el.classList.remove("chat-controls-closed");
    });

    pastChatsToggle.innerHTML = leftArrowSVG;
    chatControlsToggle.innerHTML = rightArrowSVG;
    navigationToggle.innerHTML = closeMenuSVG;
  }
}

// Run the initializer when the page loads
initializeSidebars();

// Add click event listeners to toggle buttons
pastChatsToggle.addEventListener("click", () => {
  toggleSidebar(pastChatsRow, pastChatsToggle);
});

chatControlsToggle.addEventListener("click", () => {
  toggleSidebar(chatControlsRow, chatControlsToggle);
});

navigationToggle.addEventListener("click", () => {
  toggleSidebar(headerBar, navigationToggle);
});

//------------------------------------------------
// Fixes #chat-input textarea height issue
// for devices with width <= 924px
//------------------------------------------------

if (isMobile()) {
  // Target the textarea
  const textarea = document.querySelector("#chat-input textarea");

  if (textarea) {
    // Simulate adding and removing a newline
    textarea.value += "\n";
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    textarea.value = textarea.value.slice(0, -1);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  }
}

//------------------------------------------------
// Create a top navigation bar on mobile
//------------------------------------------------

function createMobileTopBar() {
  const chatTab = document.getElementById("chat-tab");

  // Only create the top bar if it doesn't already exist
  if (chatTab && !chatTab.querySelector(".mobile-top-bar")) {
    const topBar = document.createElement("div");
    topBar.classList.add("mobile-top-bar");

    // Insert the top bar as the first child of chat-tab
    chatTab.appendChild(topBar);
  }
}

createMobileTopBar();
