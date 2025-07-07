function toggleDarkMode() {
  document.body.classList.toggle("dark");
  var currentCSS = document.getElementById("highlight-css");
  if (currentCSS.getAttribute("href") === "file/css/highlightjs/github-dark.min.css") {
    currentCSS.setAttribute("href", "file/css/highlightjs/github.min.css");
  } else {
    currentCSS.setAttribute("href", "file/css/highlightjs/github-dark.min.css");
  }

  // Re-highlight all code blocks once stylesheet loads
  currentCSS.onload = function() {
    const messageBodies = document.getElementById("chat").querySelectorAll(".message-body");
    messageBodies.forEach((messageBody) => {
      const codeBlocks = messageBody.querySelectorAll("pre code");
      codeBlocks.forEach((codeBlock) => {
        hljs.highlightElement(codeBlock);
      });
    });
  };
}
