(function() {
  "use strict";

  //------------------------------------------------
  // Keyboard shortcuts
  //------------------------------------------------

  // Private helper functions
  function isModifiedKeyboardEvent() {
    return (event instanceof KeyboardEvent &&
            event.shiftKey ||
            event.ctrlKey ||
            event.altKey ||
            event.metaKey);
  }

  function isFocusedOnEditableTextbox() {
    if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") {
      return !!event.target.value;
    }
  }

  // Global function (called from other files)
  window.navigateLastAssistantMessage = function(direction) {
    const chat = document.querySelector("#chat");
    if (!chat) return false;

    const messages = chat.querySelectorAll("[data-index]");
    if (messages.length === 0) return false;

    // Find the last assistant message (starting from the end)
    let lastAssistantMessage = null;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (
        msg.classList.contains("assistant-message") ||
                msg.querySelector(".circle-bot") ||
                msg.querySelector(".text-bot")
      ) {
        lastAssistantMessage = msg;
        break;
      }
    }

    if (!lastAssistantMessage) return false;

    const buttons = lastAssistantMessage.querySelectorAll(".version-nav-button");

    for (let i = 0; i < buttons.length; i++) {
      const button = buttons[i];
      const onclick = button.getAttribute("onclick");
      const disabled = button.hasAttribute("disabled");

      const isLeft = onclick && onclick.includes("'left'");
      const isRight = onclick && onclick.includes("'right'");

      if (!disabled) {
        if (direction === "left" && isLeft) {
          navigateVersion(button, direction);
          return true;
        }
        if (direction === "right" && isRight) {
          navigateVersion(button, direction);
          return true;
        }
      }
    }

    return false;
  };

  // Main keyboard event listener
  document.addEventListener("keydown", function(event) {
    // Stop generation on Esc pressed
    if (event.key === "Escape") {
      // Find the element with id 'stop' and click it
      var stopButton = document.getElementById("stop");
      if (stopButton) {
        stopButton.click();
      }
      return;
    }

    if (!document.querySelector("#chat-tab").checkVisibility() ) {
      return;
    }

    // Show chat controls on Ctrl + S
    if (event.ctrlKey && event.key == "s") {
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

    // Impersonate on Ctrl + Shift + M
    else if (event.ctrlKey && event.shiftKey && event.key === "M") {
      event.preventDefault();
      document.getElementById("Impersonate").click();
    }

    // --- Simple version navigation --- //
    if (!isFocusedOnEditableTextbox()) {
      // Version navigation on Arrow keys (horizontal)
      if (!isModifiedKeyboardEvent() && event.key === "ArrowLeft") {
        event.preventDefault();
        window.navigateLastAssistantMessage("left");
      }

      else if (!isModifiedKeyboardEvent() && event.key === "ArrowRight") {
        event.preventDefault();
        if (!window.navigateLastAssistantMessage("right")) {
          // If can't navigate right (last version), regenerate
          document.getElementById("Regenerate").click();
        }
      }
    }
  });

})();
