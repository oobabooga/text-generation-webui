(function() {
  "use strict";

  //------------------------------------------------
  // Chat scrolling
  //------------------------------------------------
  const targetElement = document.getElementById("chat").parentNode.parentNode.parentNode;
  targetElement.classList.add("pretty_scrollbar");
  targetElement.classList.add("chat-parent");
  window.isScrolled = false;
  let scrollTimeout;
  let bigPictureVisible = false;

  // Private variables for input height management
  let wasAtBottom = false;
  let preservedDistance = 0;

  targetElement.addEventListener("scroll", function() {
    // Add scrolling class to disable hover effects
    targetElement.classList.add("scrolling");

    let diff = targetElement.scrollHeight - targetElement.clientHeight;
    if(Math.abs(targetElement.scrollTop - diff) <= 10 || diff == 0) {
      window.isScrolled = false;
    } else {
      window.isScrolled = true;
    }

    // Clear previous timeout and set new one
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      targetElement.classList.remove("scrolling");
      doSyntaxHighlighting(); // Only run after scrolling stops
    }, 150);
  });

  // Create a MutationObserver instance
  const observer = new MutationObserver(function(mutations) {
    // Check if this is just the scrolling class being toggled
    const isScrollingClassOnly = mutations.every(mutation =>
      mutation.type === "attributes" &&
            mutation.attributeName === "class" &&
            mutation.target === targetElement
    );

    const typing = document.getElementById("typing-container");
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

    if (!window.isScrolled && !isScrollingClassOnly && targetElement.scrollTop !== targetElement.scrollHeight) {
      targetElement.scrollTop = targetElement.scrollHeight;
    }

    const chatElement = document.getElementById("chat");
    if (chatElement && chatElement.getAttribute("data-mode") === "instruct") {
      const messagesContainer = chatElement.querySelector(".messages");
      const lastChild = messagesContainer?.lastElementChild;
      const prevSibling = lastChild?.previousElementSibling;
      if (lastChild && prevSibling) {
        lastChild.style.setProperty("margin-bottom",
          `max(0px, calc(max(70vh, 100vh - ${prevSibling.offsetHeight}px - 84px) - ${lastChild.offsetHeight}px))`,
          "important"
        );
      }
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
    const messageBodies = document.getElementById("chat").querySelectorAll(".message-body");

    if (messageBodies.length > 0) {
      observer.disconnect();

      let hasSeenVisible = false;

      // Go from last message to first
      for (let i = messageBodies.length - 1; i >= 0; i--) {
        const messageBody = messageBodies[i];

        if (isElementVisibleOnScreen(messageBody)) {
          hasSeenVisible = true;

          // Handle both code and math in a single pass through each message
          const codeBlocks = messageBody.querySelectorAll("pre code:not([data-highlighted])");
          codeBlocks.forEach((codeBlock) => {
            hljs.highlightElement(codeBlock);
            codeBlock.setAttribute("data-highlighted", "true");
            codeBlock.classList.add("pretty_scrollbar");
          });

          renderMathInElement(messageBody, {
            delimiters: [
              { left: "$$", right: "$$", display: true },
              { left: "$", right: "$", display: false },
              { left: "\\(", right: "\\)", display: false },
              { left: "\\[", right: "\\]", display: true },
            ],
          });
        } else if (hasSeenVisible) {
          // We've seen visible messages but this one is not visible
          // Since we're going from last to first, we can break
          break;
        }
      }

      observer.observe(targetElement, config);
    }
  }

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
  function addBigPicture() {
    var imgElement = document.createElement("img");
    var timestamp = new Date().getTime();
    imgElement.src = "/file/user_data/cache/pfp_character.png?time=" + timestamp;
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

  // Global function (called from other files)
  window.toggleBigPicture = function() {
    if(bigPictureVisible) {
      deleteBigPicture();
      bigPictureVisible = false;
    } else {
      addBigPicture();
      bigPictureVisible = true;
    }
  };

  //------------------------------------------------
  // Create the hover menu in the chat tab
  //------------------------------------------------
  document.addEventListener("DOMContentLoaded", function() {
    var buttonsInChat = document.querySelectorAll("#chat-tab #chat-buttons button, #chat-tab #chat-buttons #show-controls");
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

        // Only apply transformations to button elements
        if (thisButton.tagName.toLowerCase() === "button") {
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
  });

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
  // Character dropdown positioning
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
    document.getElementById("restore-character").style.display = "none";
  }

  function restoreOriginalPosition() {
    if (originalParent && movedElement) {
      if (originalIndex >= originalParent.children.length) {
        originalParent.appendChild(movedElement);
      } else {
        originalParent.insertBefore(movedElement, originalParent.children[originalIndex]);
      }

      document.getElementById("save-character").style.display = "";
      document.getElementById("restore-character").style.display = "";
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

  moveToChatTab();

  //------------------------------------------------
  // Paste Handler for Long Text
  //------------------------------------------------
  const MAX_PLAIN_TEXT_LENGTH = 2500;

  function setupPasteHandler() {
    const textbox = document.querySelector("#chat-input textarea[data-testid=\"textbox\"]");
    const fileInput = document.querySelector("#chat-input input[data-testid=\"file-upload\"]");

    if (!textbox || !fileInput) {
      setTimeout(setupPasteHandler, 500);
      return;
    }

    textbox.addEventListener("paste", async (event) => {
      const text = event.clipboardData?.getData("text");

      if (text && text.length > MAX_PLAIN_TEXT_LENGTH && document.querySelector("#paste_to_attachment input[data-testid=\"checkbox\"]")?.checked) {
        event.preventDefault();

        const file = new File([text], "pasted_text.txt", {
          type: "text/plain",
          lastModified: Date.now()
        });

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        fileInput.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupPasteHandler);
  } else {
    setupPasteHandler();
  }

  //------------------------------------------------
  // Inline icons for deleting past chats
  //------------------------------------------------
  function addMiniDeletes() {
    document.querySelectorAll("#past-chats label:not(.has-delete)").forEach(label => {
      const container = document.createElement("span");
      container.className = "delete-container";

      label.classList.add("chat-label-with-delete");

      const trashBtn = document.createElement("button");
      trashBtn.innerHTML = "🗑️";
      trashBtn.className = "trash-btn";

      const cancelBtn = document.createElement("button");
      cancelBtn.innerHTML = "✕";
      cancelBtn.className = "cancel-btn";

      const confirmBtn = document.createElement("button");
      confirmBtn.innerHTML = "✓";
      confirmBtn.className = "confirm-btn";

      label.addEventListener("mouseenter", () => {
        container.style.opacity = "1";
      });

      label.addEventListener("mouseleave", () => {
        container.style.opacity = "0";
      });

      trashBtn.onclick = (e) => {
        e.stopPropagation();
        label.querySelector("input").click();
        document.querySelector("#delete_chat").click();
        trashBtn.style.display = "none";
        cancelBtn.style.display = "flex";
        confirmBtn.style.display = "flex";
      };

      cancelBtn.onclick = (e) => {
        e.stopPropagation();
        document.querySelector("#delete_chat-cancel").click();
        resetButtons();
      };

      confirmBtn.onclick = (e) => {
        e.stopPropagation();
        document.querySelector("#delete_chat-confirm").click();
        resetButtons();
      };

      function resetButtons() {
        trashBtn.style.display = "inline";
        cancelBtn.style.display = "none";
        confirmBtn.style.display = "none";
      }

      container.append(trashBtn, cancelBtn, confirmBtn);
      label.appendChild(container);
      label.classList.add("has-delete");
    });
  }

  new MutationObserver(() => addMiniDeletes()).observe(
    document.querySelector("#past-chats"),
    {childList: true, subtree: true}
  );
  addMiniDeletes();

  //------------------------------------------------
  // Maintain distance from bottom when input height changes
  //------------------------------------------------
  const chatInput = document.querySelector("#chat-input textarea");

  function checkIfAtBottom() {
    const distanceFromBottom = targetElement.scrollHeight - targetElement.scrollTop - targetElement.clientHeight;
    wasAtBottom = distanceFromBottom <= 1; // Allow for rounding errors
  }

  function preserveScrollPosition() {
    preservedDistance = targetElement.scrollHeight - targetElement.scrollTop - targetElement.clientHeight;
  }

  function restoreScrollPosition() {
    if (wasAtBottom) {
      // Force to bottom
      targetElement.scrollTop = targetElement.scrollHeight - targetElement.clientHeight;
    } else {
      // Restore original distance
      targetElement.scrollTop = targetElement.scrollHeight - targetElement.clientHeight - preservedDistance;
    }
  }

  // Check position before input
  chatInput.addEventListener("beforeinput", () => {
    checkIfAtBottom();
    preserveScrollPosition();
  });

  // Restore after input
  chatInput.addEventListener("input", () => {
    requestAnimationFrame(() => restoreScrollPosition());
  });

  // Update wasAtBottom when user scrolls
  targetElement.addEventListener("scroll", checkIfAtBottom);

})();
