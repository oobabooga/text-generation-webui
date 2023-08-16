const belowChatInput = document.querySelectorAll("#chat-tab > div > :nth-child(n+3), #extensions");
const chatDisplayElement = document.getElementById("chat-display");

function toggle_controls(value) {
    if (value) {
        belowChatInput.forEach(element => {
          element.style.display = "inherit";
        });

        chatDisplayElement.classList.remove("bigchat");
    } else {
        belowChatInput.forEach(element => {
          element.style.display = "none";
        });

        chatDisplayElement.classList.add("bigchat");
    }
}
