const belowChatInput = document.querySelectorAll("#chat-tab > div > :nth-child(n+2), #extensions");
const chatParent = document.getElementById("chat").parentNode.parentNode.parentNode;

function toggle_controls(value) {
    if (value) {
        belowChatInput.forEach(element => {
          element.style.display = "inherit";
        });

        chatParent.classList.remove("bigchat");
        document.getElementById('show-controls').parentNode.parentNode.style.paddingBottom = '115px';
        document.querySelector('.chat-parent').style.height = 'calc(100dvh - 240px)';
    } else {
        belowChatInput.forEach(element => {
          element.style.display = "none";
        });

        chatParent.classList.add("bigchat");
        document.getElementById('show-controls').parentNode.parentNode.style.paddingBottom = '95px';
        document.querySelector('.chat-parent').style.height = 'calc(100dvh - 181px)';
    }
}
