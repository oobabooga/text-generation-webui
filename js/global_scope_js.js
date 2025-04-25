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
  // Store references to any open thinking blocks and their scroll positions before update
  const openBlocks = {};
  document.querySelectorAll(".thinking-block[open]").forEach(block => {
    const content = block.querySelector(".thinking-content");
    const blockId = block.getAttribute("data-block-id");
    if (content) {
      // Check if user was scrolled to bottom (with small tolerance)
      const isAtBottom = Math.abs((content.scrollHeight - content.scrollTop) - content.clientHeight) < 5;
      openBlocks[blockId] = {
        element: block,
        isAtBottom: isAtBottom
      };
    }
  });

  morphdom(
    document.getElementById("chat").parentNode,
    "<div class=\"prose svelte-1ybaih5\">" + text + "</div>",
    {
      onBeforeElUpdated: function(fromEl, toEl) {
        if (fromEl.tagName === "PRE" && fromEl.querySelector("code[data-highlighted]")) {
          const fromCode = fromEl.querySelector("code");
          const toCode = toEl.querySelector("code");

          if (fromCode && toCode && fromCode.textContent === toCode.textContent) {
            toEl.className = fromEl.className;
            toEl.innerHTML = fromEl.innerHTML;
            return false; // Skip updating the <pre> element
          }
        }

        // Preserve open/closed state for thinking blocks
        if (fromEl.classList && fromEl.classList.contains("thinking-block") &&
            toEl.classList && toEl.classList.contains("thinking-block")) {
          // Check if IDs match exactly (handles streaming updates)
          if (fromEl.getAttribute("data-block-id") === toEl.getAttribute("data-block-id") &&
              fromEl.hasAttribute("open")) {
            toEl.setAttribute("open", "");
          }
        }

        return !fromEl.isEqualNode(toEl); // Update only if nodes differ
      },

      // Add this callback to handle after element updates
      onElUpdated: function(el) {
        // Check if this is a thinking-block that was open before
        if (el.classList && el.classList.contains("thinking-block") && el.hasAttribute("open")) {
          const blockId = el.getAttribute("data-block-id");
          const content = el.querySelector(".thinking-content");

          if (content) {
            // If this is a newly opened block or was at the bottom before, scroll to bottom
            if (!openBlocks[blockId] || openBlocks[blockId].isAtBottom) {
              setTimeout(() => {
                content.scrollTop = content.scrollHeight;
              }, 0);
            }
          }
        }
      }
    }
  );

  // Also add event listener for when details are opened manually
  document.querySelectorAll(".thinking-block").forEach(block => {
    if (!block._hasOpenListener) {
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
      block._hasOpenListener = true;
    }
  });
}
