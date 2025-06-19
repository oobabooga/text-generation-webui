// -------------------------------------------------
// Event handlers
// -------------------------------------------------

function copyToClipboard(element) {
  if (!element) return;

  const messageElement = element.closest(".message, .user-message, .assistant-message");
  if (!messageElement) return;

  const rawText = messageElement.getAttribute("data-raw");
  if (!rawText) return;

  navigator.clipboard.writeText(rawText).then(function() {
    const originalSvg = element.innerHTML;
    element.innerHTML = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" class=\"text-green-500 dark:text-green-400\"><path d=\"M5 12l5 5l10 -10\"></path></svg>";
    setTimeout(() => {
      element.innerHTML = originalSvg;
    }, 1000);
  }).catch(function(err) {
    console.error("Failed to copy text: ", err);
  });
}

function branchHere(element) {
  if (!element) return;

  const messageElement = element.closest(".message, .user-message, .assistant-message");
  if (!messageElement) return;

  const index = messageElement.getAttribute("data-index");
  if (!index) return;

  const branchIndexInput = document.getElementById("Branch-index").querySelector("input");
  if (!branchIndexInput) {
    console.error("Element with ID 'Branch-index' not found.");
    return;
  }
  const branchButton = document.getElementById("Branch");

  if (!branchButton) {
    console.error("Required element 'Branch' not found.");
    return;
  }

  branchIndexInput.value = index;

  // Trigger any 'change' or 'input' events Gradio might be listening for
  const event = new Event("input", { bubbles: true });
  branchIndexInput.dispatchEvent(event);

  branchButton.click();
}

// -------------------------------------------------
// Message Editing Functions
// -------------------------------------------------

function editHere(buttonElement) {
  if (!buttonElement) return;

  const messageElement = buttonElement.closest(".message, .user-message, .assistant-message");
  if (!messageElement) return;

  const messageBody = messageElement.querySelector(".message-body");
  if (!messageBody) return;

  // If already editing, focus the textarea
  const existingTextarea = messageBody.querySelector(".editing-textarea");
  if (existingTextarea) {
    existingTextarea.focus();
    return;
  }

  // Determine role based on message element - handle different chat modes
  const isUserMessage = messageElement.classList.contains("user-message") ||
                       messageElement.querySelector(".text-you") !== null ||
                       messageElement.querySelector(".circle-you") !== null;

  startEditing(messageElement, messageBody, isUserMessage);
}

function startEditing(messageElement, messageBody, isUserMessage) {
  const rawText = messageElement.getAttribute("data-raw") || messageBody.textContent;
  const originalHTML = messageBody.innerHTML;

  // Create editing interface
  const editingInterface = createEditingInterface(rawText);

  // Replace message content
  messageBody.innerHTML = "";
  messageBody.appendChild(editingInterface.textarea);
  messageBody.appendChild(editingInterface.controls);

  editingInterface.textarea.focus();
  editingInterface.textarea.setSelectionRange(rawText.length, rawText.length);

  // Temporarily mark as scrolled to prevent auto-scroll
  const wasScrolled = window.isScrolled;
  window.isScrolled = true;

  // Scroll the textarea into view
  editingInterface.textarea.scrollIntoView({
    behavior: "smooth",
    block: "center"
  });

  // Restore the original scroll state after animation
  setTimeout(() => {
    window.isScrolled = wasScrolled;
  }, 500);

  // Setup event handlers
  setupEditingHandlers(editingInterface.textarea, messageElement, originalHTML, messageBody, isUserMessage);
}

function createEditingInterface(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.className = "editing-textarea";
  textarea.rows = Math.max(3, text.split("\n").length);

  const controls = document.createElement("div");
  controls.className = "edit-controls-container";

  const saveButton = document.createElement("button");
  saveButton.textContent = "Save";
  saveButton.className = "edit-control-button";
  saveButton.type = "button";

  const cancelButton = document.createElement("button");
  cancelButton.textContent = "Cancel";
  cancelButton.className = "edit-control-button edit-cancel-button";
  cancelButton.type = "button";

  controls.appendChild(saveButton);
  controls.appendChild(cancelButton);

  return { textarea, controls, saveButton, cancelButton };
}

function setupEditingHandlers(textarea, messageElement, originalHTML, messageBody, isUserMessage) {
  const saveButton = messageBody.querySelector(".edit-control-button:not(.edit-cancel-button)");
  const cancelButton = messageBody.querySelector(".edit-cancel-button");

  const submitEdit = () => {
    const index = messageElement.getAttribute("data-index");
    if (!index || !submitMessageEdit(index, textarea.value, isUserMessage)) {
      cancelEdit();
    }
  };

  const cancelEdit = () => {
    messageBody.innerHTML = originalHTML;
  };

  // Event handlers
  saveButton.onclick = submitEdit;
  cancelButton.onclick = cancelEdit;

  textarea.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitEdit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancelEdit();
    }
  };
}

function submitMessageEdit(index, newText, isUserMessage) {
  const editIndexInput = document.getElementById("Edit-message-index")?.querySelector("input");
  const editTextInput = document.getElementById("Edit-message-text")?.querySelector("textarea");
  const editRoleInput = document.getElementById("Edit-message-role")?.querySelector("textarea");
  const editButton = document.getElementById("Edit-message");

  if (!editIndexInput || !editTextInput || !editRoleInput || !editButton) {
    console.error("Edit elements not found");
    return false;
  }

  editIndexInput.value = index;
  editTextInput.value = newText;
  editRoleInput.value = isUserMessage ? "user" : "assistant";

  editIndexInput.dispatchEvent(new Event("input", { bubbles: true }));
  editTextInput.dispatchEvent(new Event("input", { bubbles: true }));
  editRoleInput.dispatchEvent(new Event("input", { bubbles: true }));

  editButton.click();
  return true;
}

function navigateVersion(element, direction) {
  if (!element) return;

  const messageElement = element.closest(".message, .user-message, .assistant-message");
  if (!messageElement) return;

  const index = messageElement.getAttribute("data-index");
  if (!index) return;

  // Determine role based on message element classes
  let role = "assistant"; // Default role
  if (messageElement.classList.contains("user-message") ||
      messageElement.querySelector(".text-you") ||
      messageElement.querySelector(".circle-you")) {
    role = "user";
  }

  const indexInput = document.getElementById("Navigate-message-index")?.querySelector("input");
  const directionInput = document.getElementById("Navigate-direction")?.querySelector("textarea");
  const roleInput = document.getElementById("Navigate-message-role")?.querySelector("textarea");
  const navigateButton = document.getElementById("Navigate-version");

  if (!indexInput || !directionInput || !roleInput || !navigateButton) {
    console.error("Navigation control elements (index, direction, role, or button) not found.");
    return;
  }

  indexInput.value = index;
  directionInput.value = direction;
  roleInput.value = role;

  // Trigger 'input' events for Gradio to pick up changes
  const event = new Event("input", { bubbles: true });
  indexInput.dispatchEvent(event);
  directionInput.dispatchEvent(event);
  roleInput.dispatchEvent(event);

  navigateButton.click();
}

function regenerateClick() {
  document.getElementById("Regenerate").click();
}

function continueClick() {
  document.getElementById("Continue").click();
}

function removeLastClick() {
  document.getElementById("Remove-last").click();
}

function handleMorphdomUpdate(data) {
  // Determine target element and use it as query scope
  var target_element, target_html;
  if (data.last_message_only) {
    const childNodes = document.getElementsByClassName("messages")[0].childNodes;
    target_element = childNodes[childNodes.length - 1];
    target_html = data.html;
  } else {
    target_element = document.getElementById("chat").parentNode;
    target_html =  "<div class=\"prose svelte-1ybaih5\">" + data.html + "</div>";
  }

  const queryScope = target_element;

  // Track open blocks
  const openBlocks = new Set();
  queryScope.querySelectorAll(".thinking-block").forEach(block => {
    const blockId = block.getAttribute("data-block-id");
    // If block exists and is open, add to open set
    if (blockId && block.hasAttribute("open")) {
      openBlocks.add(blockId);
    }
  });

  // Store scroll positions for any open blocks
  const scrollPositions = {};
  queryScope.querySelectorAll(".thinking-block[open]").forEach(block => {
    const content = block.querySelector(".thinking-content");
    const blockId = block.getAttribute("data-block-id");
    if (content && blockId) {
      const isAtBottom = Math.abs((content.scrollHeight - content.scrollTop) - content.clientHeight) < 5;
      scrollPositions[blockId] = {
        position: content.scrollTop,
        isAtBottom: isAtBottom
      };
    }
  });

  morphdom(
    target_element,
    target_html,
    {
      onBeforeElUpdated: function(fromEl, toEl) {
        // Preserve code highlighting
        if (fromEl.tagName === "PRE" && fromEl.querySelector("code[data-highlighted]")) {
          const fromCode = fromEl.querySelector("code");
          const toCode = toEl.querySelector("code");

          if (fromCode && toCode && fromCode.textContent === toCode.textContent) {
            toEl.className = fromEl.className;
            toEl.innerHTML = fromEl.innerHTML;
            return false;
          }
        }

        // For thinking blocks, assume closed by default
        if (fromEl.classList && fromEl.classList.contains("thinking-block") &&
           toEl.classList && toEl.classList.contains("thinking-block")) {
          const blockId = toEl.getAttribute("data-block-id");
          // Remove open attribute by default
          toEl.removeAttribute("open");
          // If this block was explicitly opened by user, keep it open
          if (blockId && openBlocks.has(blockId)) {
            toEl.setAttribute("open", "");
          }
        }

        return !fromEl.isEqualNode(toEl);
      },

      onElUpdated: function(el) {
        // Restore scroll positions for open thinking blocks
        if (el.classList && el.classList.contains("thinking-block") && el.hasAttribute("open")) {
          const blockId = el.getAttribute("data-block-id");
          const content = el.querySelector(".thinking-content");

          if (content && blockId && scrollPositions[blockId]) {
            setTimeout(() => {
              if (scrollPositions[blockId].isAtBottom) {
                content.scrollTop = content.scrollHeight;
              } else {
                content.scrollTop = scrollPositions[blockId].position;
              }
            }, 0);
          }
        }
      }
    }
  );

  // Add toggle listeners for new blocks
  queryScope.querySelectorAll(".thinking-block").forEach(block => {
    if (!block._hasToggleListener) {
      block.addEventListener("toggle", function(e) {
        if (this.open) {
          const content = this.querySelector(".thinking-content");
          if (content) {
            setTimeout(() => {
              content.scrollTop = content.scrollHeight;
            }, 0);
          }
        }
      });
      block._hasToggleListener = true;
    }
  });
}

// Wait for Gradio to finish setting its styles, then force dark theme
const observer = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.type === "attributes" &&
        mutation.target.tagName === "GRADIO-APP" &&
        mutation.attributeName === "style") {

      // Gradio just set its styles, now force dark theme
      document.body.classList.add("dark");
      observer.disconnect();
    }
  });
});

// Start observing
observer.observe(document.documentElement, {
  attributes: true,
  subtree: true,
  attributeFilter: ["style"]
});
