(function() {
  "use strict";

  // ------------------------------------------------
  // Main
  // ------------------------------------------------

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
  // Position the chat typing dots
  //------------------------------------------------
  const typing = document.getElementById("typing-container");
  const typingParent = typing.parentNode;
  const typingSibling = typing.previousElementSibling;
  typingSibling.insertBefore(typing, typingSibling.childNodes[2]);

  //------------------------------------------------
  // Add some scrollbars
  //------------------------------------------------
  const textareaElements = document.querySelectorAll(".add_scrollbar textarea");
  for(let i = 0; i < textareaElements.length; i++) {
    textareaElements[i].classList.remove("scroll-hide");
    textareaElements[i].classList.add("pretty_scrollbar");
    textareaElements[i].style.resize = "none";
  }

  //------------------------------------------------
  // Remove some backgrounds
  //------------------------------------------------
  const noBackgroundelements = document.querySelectorAll(".no-background");
  for(let i = 0; i < noBackgroundelements.length; i++) {
    noBackgroundelements[i].parentNode.style.border = "none";
    noBackgroundelements[i].parentNode.parentNode.parentNode.style.alignItems = "center";
  }

  const slimDropdownElements = document.querySelectorAll(".slim-dropdown");
  for (let i = 0; i < slimDropdownElements.length; i++) {
    const parentNode = slimDropdownElements[i].parentNode;
    parentNode.style.background = "transparent";
    parentNode.style.border = "0";
  }

  //------------------------------------------------
  // Position the chat input
  //------------------------------------------------
  document.getElementById("chat-input-row").classList.add("chat-input-positioned");

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
  // Add a confirmation dialog when leaving the page
  // Useful to avoid data loss
  //------------------------------------------------
  window.addEventListener("beforeunload", function (event) {
    // Cancel the event
    event.preventDefault();
    // Chrome requires returnValue to be set
    event.returnValue = "";
  });

  //------------------------------------------------
  // Tooltips
  //------------------------------------------------
  // File upload button
  document.querySelector("#chat-input .upload-button").title = "Upload text files, PDFs, and DOCX documents";

  // Activate web search
  document.getElementById("web-search").title = "Search the internet with DuckDuckGo";

})();
