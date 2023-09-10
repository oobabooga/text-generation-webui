const belowChatInput = document.querySelectorAll("#chat-tab > div > :nth-child(n+3), #extensions");
const chatParent = document.getElementById("chat").parentNode.parentNode.parentNode;
const chatcontrols = document.getElementById("chat-controls")

function toggle_controls(value) {
    chatcontrols.style.display = value ? "inherit" : "none"
    belowChatInput.forEach(element => {
        element.style.display = value ? "inherit" : "none";
    });
}