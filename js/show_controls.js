const belowChatInput = document.querySelectorAll("#chat-tab > div > :nth-child(n+2), #extensions");
const chatParent = document.querySelector(".chat-parent.old-ui");

function toggle_controls(value) {
    if (value) {
        belowChatInput.forEach(element => {
          element.style.display = "inherit";
        });
        
        if (chatParent) {
          chatParent.classList.remove("bigchat");
        }
        document.getElementById('stop').parentElement.parentElement.parentElement.style.paddingBottom = '20px';
        document.querySelector('#show-controls:not(.old-ui)').parentNode.parentNode.style.paddingBottom = '115px';
    } else {
        belowChatInput.forEach(element => {
          element.style.display = "none";
        });

        if (chatParent) {
          chatParent.classList.add("bigchat");
        }
        document.getElementById('stop').parentElement.parentElement.parentElement.style.paddingBottom = '0px';
        document.querySelector('#show-controls:not(.old-ui)').parentNode.parentNode.style.paddingBottom = '95px';
    }
}
