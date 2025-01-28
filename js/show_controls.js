const belowChatInput = document.querySelectorAll(
  "#chat-tab > div > :nth-child(1), #chat-tab > div > :nth-child(3), #chat-tab > div > :nth-child(4), #extensions"
);
const chatParent = document.querySelector(".chat-parent");

function toggle_controls(value) {
  if (value) {
    belowChatInput.forEach(element => {
      element.style.display = "inherit";
    });

    chatParent.classList.remove("bigchat");
    document.getElementById("chat-input-row").classList.remove("bigchat");
    document.getElementById("chat-col").classList.remove("bigchat");
    document.getElementById("chat-tab").style.paddingBottom = "";

    let gallery_element = document.getElementById("gallery-extension");
    if (gallery_element) {
      gallery_element.style.display = "block";
    }

  } else {
    belowChatInput.forEach(element => {
      element.style.display = "none";
    });

    chatParent.classList.add("bigchat");
    document.getElementById("chat-input-row").classList.add("bigchat");
    document.getElementById("chat-col").classList.add("bigchat");
    document.getElementById("chat-tab").style.paddingBottom = "0px";
  }
}
