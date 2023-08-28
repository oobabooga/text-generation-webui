const belowChatInput = document.querySelectorAll("#chat-tab > div > :nth-child(n+3), #extensions");
const chatParent = document.getElementById("chat").parentNode.parentNode.parentNode;

function toggle_controls(value) {
    if (value) {
        belowChatInput.forEach(element => {
          element.style.display = "inherit";
        });

        chatParent.classList.remove("bigchat");
    } else {
        belowChatInput.forEach(element => {
          element.style.display = "none";
        });

        chatParent.classList.add("bigchat");
    }
}
