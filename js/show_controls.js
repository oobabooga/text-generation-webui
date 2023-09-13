const belowChatInput = document.querySelectorAll("#chat-tab > div > :first-child > :nth-child(n+3), #extensions");
const chatParent = document.getElementById("chat").parentNode.parentNode.parentNode;

function toggle_controls(value) {
    belowChatInput.forEach(element => {
        element.style.display = value ? "inherit" : "none";
    });
	document.querySelector("#chat-tab > div > :nth-child(2)").style.display = value ? "initial" : "none";
	document.getElementById('stop').parentElement.parentElement.parentElement.style.paddingBottom = value ? '20px' : "0px";
}