// ------------------------------------------------
// Helper functions
// ------------------------------------------------

// Get Gradio app root (if available, otherwise use document)
function gradioApp() {
  const elems = document.querySelectorAll('gradio-app');
  const gradioShadowRoot = elems.length > 0 ? elems[0].shadowRoot : null;
  return gradioShadowRoot || document;
}

// Update Gradio element value
function updateGradioInput(element, value) {
    if (element) {
      element.value = value;
      element.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
      console.warn("Attempted to update a null Gradio input element.");
    }
}

// Check message classes and children to determine if it's a bot message
function isBotMessage(messageElement) {
  if (!messageElement) return null;

  if (messageElement.classList.contains('assistant-message')) return true;
  if (messageElement.querySelector('.circle-bot, .text-bot')) return true;

  return false;
}

// Prepare and stringify payload for sending to the backend
function preparePayload(action, payload) {
  if (typeof payload === 'object') {
    return JSON.stringify({ action: action, payload: payload });
  } else if (typeof payload === 'string') {
    return JSON.stringify({ action: action, payload: { text: payload } });
  } else {
    console.error("Invalid payload type:", typeof payload);
    return null;
  }
}

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

  const jsonStrInput = document.querySelector("#temporary-json-str textarea");
  if (!jsonStrInput) {
    console.error("Element with ID 'Temporary-index' not found.");
    return;
  }
  const branchButton = document.querySelector("#Branch");

  if (!branchButton) {
    console.error("Required element 'Branch' not found.");
    return;
  }

  updateGradioInput(jsonStrInput, preparePayload("branch_message", { messageIndex: parseInt(index) }));
  branchButton.click();
}

function editHere(buttonElement) {
  if (!buttonElement) return;

  const messageElement = buttonElement.closest(".message, .user-message, .assistant-message");
  if (!messageElement) return;

  let messageBody = messageElement.querySelector(".message-body");
  if (!messageBody) return;

  // If already editing, do nothing (or perhaps focus the textarea)
  if (messageBody.tagName === 'TEXTAREA' && messageBody.classList.contains('editing-textarea')) {
    messageBody.focus();
    return;
  }

  const rawText = messageElement.getAttribute("data-raw") || messageBody.textContent;
  const originalHTML = messageBody.innerHTML;

  const gradio = gradioApp();

  const textarea = document.createElement("textarea");
  textarea.value = rawText;
  textarea.classList.add("editing-textarea");
  textarea.rows = Math.max(3, rawText.split('\n').length); // Adjust rows based on content

  // Replace messageBody with textarea
  messageBody.parentNode.replaceChild(textarea, messageBody);
  textarea.focus();
  textarea.selectionStart = textarea.selectionEnd = textarea.value.length; // Move cursor to end

  // Create a container for the buttons
  const editControlsContainer = document.createElement('div');
  editControlsContainer.classList.add('edit-controls-container');

  // Create "Branch and Edit" button
  const branchAndEditButton = document.createElement('button');
  branchAndEditButton.textContent = 'Branch & Edit';
  branchAndEditButton.classList.add('edit-control-button');
  branchAndEditButton.onclick = () => submitEdit(true);

  // Create "Cancel" button
  const cancelButton = document.createElement('button');
  cancelButton.textContent = 'Cancel';
  cancelButton.classList.add('edit-control-button', 'edit-cancel-button');
  cancelButton.onclick = cancelEdit;

  editControlsContainer.appendChild(branchAndEditButton);
  editControlsContainer.appendChild(cancelButton);

  // Insert button container after the textarea
  textarea.parentNode.insertBefore(editControlsContainer, textarea.nextSibling);

  function submitEdit(doBranch = false) {
    try {
      const newText = textarea.value;
      // const doBranch = branchCheckbox.checked; // This is now passed as a parameter and used directly in the payload

      const indexStr = messageElement.getAttribute("data-index");
      if (indexStr === null) {
        console.error("Message index (data-index) not found.");
        cleanup();
        return;
      }
      const index = parseInt(indexStr);
      const type = isBotMessage(messageElement) ? 1 : 0;
      console.debug(`Editing message at index: ${index}, type: ${type}, doBranch: ${doBranch}`);

      if (!isNaN(index)) {
        const jsonStrInput = gradio.querySelector("#temporary-json-str textarea");
        if (!jsonStrInput) {
            console.error("Element with ID 'temporary-json-str textarea' not found.");
            cleanup();
            return;
        }
        
        const submitEditButton = gradio.querySelector("#edit");
        if (!submitEditButton) {
            console.error("Hidden submit button for edits (i.e., #edit) not found.");
            cleanup();
            return;
        }

        const payload = preparePayload("edit_message", {
          messageIndex: index,
          newText: newText,
          messageType: type,
          doBranch: doBranch
        });

        updateGradioInput(jsonStrInput, payload);
        
        submitEditButton.click();
      } else {
        console.error("Invalid message index for edit:", indexStr);
      }
    } catch (error) {
        console.error("Error during submitEdit:", error);
    } finally {
      cleanup(); // Ensure cleanup happens, submitEditButton.click() is async from Gradio's perspective
    }
  };

  function cancelEdit() {
    const originalBody = document.createElement("div");
    originalBody.classList.add("message-body");
    originalBody.innerHTML = originalHTML;

    textarea.parentNode.replaceChild(originalBody, textarea);
    cleanup();
  };


  function eventListener(event) {
    if (event.type === 'blur') {
      // Delay slightly to allow click on potential submit/cancel buttons if they were part_of the textarea
      setTimeout(() => submitEdit(), 100);
    } else if (event.type === 'keydown') {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitEdit();
      } else if (event.key === "Escape") {
        event.preventDefault();
        cancelEdit();
      }
    }
  };

  textarea.addEventListener("blur", eventListener);
  textarea.addEventListener("keydown", eventListener);

  function cleanup() {
    textarea.removeEventListener("blur", eventListener);
    textarea.removeEventListener("keydown", eventListener);
    if (editControlsContainer && editControlsContainer.parentNode) {
        editControlsContainer.parentNode.removeChild(editControlsContainer);
    }
  }
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

function handleMorphdomUpdate(text) {
  // Track open blocks
  const openBlocks = new Set();
  document.querySelectorAll(".thinking-block").forEach(block => {
    const blockId = block.getAttribute("data-block-id");
    // If block exists and is open, add to open set
    if (blockId && block.hasAttribute("open")) {
      openBlocks.add(blockId);
    }
  });

  // Store scroll positions for any open blocks
  const scrollPositions = {};
  document.querySelectorAll(".thinking-block[open]").forEach(block => {
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
    document.getElementById("chat").parentNode,
    "<div class=\"prose svelte-1ybaih5\">" + text + "</div>",
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
  document.querySelectorAll(".thinking-block").forEach(block => {
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
