const belowChatInput = document.querySelectorAll("#chat-tab > div > :nth-child(n+3), #extensions");
const chatElement = document.getElementById("chat");

function toggle_controls(value) {
    if (value) {
        belowChatInput.forEach(element => {
          element.style.display = "inherit";
        });

        chatElement.classList.remove("bigchat");
    } else {
        belowChatInput.forEach(element => {
          element.style.display = "none";
        });

        chatElement.classList.add("bigchat");
    }
}
