function toggle_controls(value) {
  const navToggle = document.getElementById("navigation-toggle");
  const pastChatsToggle = document.getElementById("past-chats-toggle");
  const extensions = document.querySelector("#extensions");
  const galleryExtension = document.getElementById("gallery-extension");

  if (value) {
    // SHOW MODE: Click toggles to show hidden sidebars
    if (navToggle && document.querySelector(".header_bar")?.classList.contains("sidebar-hidden")) {
      navToggle.click();
    }
    if (pastChatsToggle && document.getElementById("past-chats-row")?.classList.contains("sidebar-hidden")) {
      pastChatsToggle.click();
    }

    // Show extensions only
    if (extensions) {
      extensions.style.display = "inherit";
    }
    if (galleryExtension) {
      galleryExtension.style.display = "block";
    }
  } else {
    // HIDE MODE: Click toggles to hide visible sidebars
    if (navToggle && !document.querySelector(".header_bar")?.classList.contains("sidebar-hidden")) {
      navToggle.click();
    }
    if (pastChatsToggle && !document.getElementById("past-chats-row")?.classList.contains("sidebar-hidden")) {
      pastChatsToggle.click();
    }

    // Hide extensions only
    if (extensions) {
      extensions.style.display = "none";
    }
    if (galleryExtension) {
      galleryExtension.style.display = "none";
    }
  }
}
