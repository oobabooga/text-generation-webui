const belowChatInput = document.querySelectorAll("#chat-tab > div > :nth-child(n+2), #extensions");
const chatParent = document.getElementById("chat").parentNode.parentNode.parentNode;

function toggle_controls(value) {
    if (value) {
        belowChatInput.forEach(element => {
          element.style.display = "inherit";
        });

        chatParent.classList.remove("bigchat");
        document.getElementById('stop').parentElement.parentElement.parentElement.style.paddingBottom = '20px';
        document.getElementById('show-controls').parentNode.parentNode.style.paddingBottom = '115px';
    } else {
        belowChatInput.forEach(element => {
          element.style.display = "none";
        });

        chatParent.classList.add("bigchat");
        document.getElementById('stop').parentElement.parentElement.parentElement.style.paddingBottom = '0px';
        document.getElementById('show-controls').parentNode.parentNode.style.paddingBottom = '95px';
    }
}
