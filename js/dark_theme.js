function toggleDarkMode() {
  document.body.classList.toggle("dark");
  const currentCSS = document.getElementById("highlight-css");
  if (currentCSS.getAttribute("href") === "file/css/highlightjs/github-dark.min.css") {
    currentCSS.setAttribute("href", "file/css/highlightjs/github.min.css");
  } else {
    currentCSS.setAttribute("href", "file/css/highlightjs/github-dark.min.css");
  }

  // Re-highlight all code blocks once stylesheet loads
  currentCSS.onload = function() {
    // Clear data-highlighted so hljs will re-process with the new theme
    document.querySelectorAll("#chat .message-body pre code[data-highlighted]").forEach((codeBlock) => {
      delete codeBlock.dataset.highlighted;
    });
    doSyntaxHighlighting();
  };
}
