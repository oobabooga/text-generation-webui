const chatParent = document.querySelector(".chat-parent");

function toggle_controls(value) {
  const extensions = document.querySelector("#extensions");

  if (value) {
    // SHOW MODE: Click toggles to show hidden sidebars
    const navToggle = document.getElementById("navigation-toggle");
    const pastChatsToggle = document.getElementById("past-chats-toggle");

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

    let gallery_element = document.getElementById("gallery-extension");
    if (gallery_element) {
      gallery_element.style.display = "block";
    }

  } else {
    // HIDE MODE: Click toggles to hide visible sidebars
    const navToggle = document.getElementById("navigation-toggle");
    const pastChatsToggle = document.getElementById("past-chats-toggle");

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
  }
}
