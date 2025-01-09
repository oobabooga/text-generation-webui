function toggleDarkMode() {
  document.body.classList.toggle("dark");
  var currentCSS = document.getElementById("highlight-css");
  if (currentCSS.getAttribute("href") === "file/css/highlightjs/github-dark.min.css") {
    currentCSS.setAttribute("href", "file/css/highlightjs/github.min.css");
  } else {
    currentCSS.setAttribute("href", "file/css/highlightjs/github-dark.min.css");
  }
}
