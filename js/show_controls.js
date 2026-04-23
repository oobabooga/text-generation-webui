if (window._controlsInitialized === undefined) {
  window._controlsInitialized = false;
}

function toggle_controls(value) {
  const extensions = document.querySelector("#extensions");
  const galleryExtension = document.getElementById("gallery-extension");

  if (window._controlsInitialized) {
    window.SIDEBARS.forEach(({ element, toggle, key }) => {
      if (value) {
        if (element && element.classList.contains("sidebar-hidden")) {
          window.toggleSidebar(element, toggle);
        }
        localStorage.removeItem(key);
      } else {
        if (element && !element.classList.contains("sidebar-hidden")) {
          window.toggleSidebar(element, toggle);
        }
        localStorage.setItem(key, "true");
      }
    });
  }

  if (value) {
    if (extensions) extensions.style.display = "inherit";
    if (galleryExtension) galleryExtension.style.display = "block";
  } else {
    if (extensions) extensions.style.display = "none";
    if (galleryExtension) galleryExtension.style.display = "none";
  }

  window._controlsInitialized = true;
}
