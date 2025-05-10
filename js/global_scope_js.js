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
  const event = new Event('input', { bubbles: true }); // 'change' might also work
  branchIndexInput.dispatchEvent(event);

  branchButton.click(); // Gradio will now pick up the 'index'

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
  morphdom(
    document.getElementById("chat").parentNode,
    "<div class=\"prose svelte-1ybaih5\">" + text + "</div>",
    {
      onBeforeElUpdated: function(fromEl, toEl) {
        if (fromEl.tagName === "PRE" && fromEl.querySelector("code[data-highlighted]")) {
          const fromCode = fromEl.querySelector("code");
          const toCode = toEl.querySelector("code");

          if (fromCode && toCode && fromCode.textContent === toCode.textContent) {
            // If the <code> content is the same, preserve the entire <pre> element
            toEl.className = fromEl.className;
            toEl.innerHTML = fromEl.innerHTML;
            return false; // Skip updating the <pre> element
          }
        }
        return !fromEl.isEqualNode(toEl); // Update only if nodes differ
      }
    }
  );
}
