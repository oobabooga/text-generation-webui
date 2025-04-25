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
  // Track closed blocks
  const closedBlocks = new Set();
  document.querySelectorAll(".thinking-block").forEach(block => {
    const blockId = block.getAttribute("data-block-id");
    // If block exists and is not open, add to closed set
    if (blockId && !block.hasAttribute("open")) {
      closedBlocks.add(blockId);
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

        // For thinking blocks, respect closed state
        if (fromEl.classList && fromEl.classList.contains("thinking-block") &&
            toEl.classList && toEl.classList.contains("thinking-block")) {
          const blockId = toEl.getAttribute("data-block-id");
          // If this block was closed by user, keep it closed
          if (blockId && closedBlocks.has(blockId)) {
            toEl.removeAttribute("open");
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
