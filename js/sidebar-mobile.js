(function() {
  "use strict";

  //------------------------------------------------
  // Buttons to toggle the sidebars
  //------------------------------------------------

  // Private constants
  const leftArrowSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-arrow-bar-left">
      <path d="M4 12l10 0"></path>
      <path d="M4 12l4 4"></path>
      <path d="M4 12l4 -4"></path>
      <path d="M20 4l0 16"></path>
    </svg>`;

  const rightArrowSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-arrow-bar-right">
      <path d="M20 12l-10 0"></path>
      <path d="M20 12l-4 4"></path>
      <path d="M20 12l-4 -4"></path>
      <path d="M4 4l0 16"></path>
    </svg>`;

  const hamburgerMenuSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-hamburger-menu">
      <line x1="3" y1="12" x2="21" y2="12"></line>
      <line x1="3" y1="6" x2="21" y2="6"></line>
      <line x1="3" y1="18" x2="21" y2="18"></line>
    </svg>`;

  const closeMenuSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-close-menu">
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>`;

  // Private variables
  const chatTab = document.getElementById("chat-tab");
  const pastChatsRow = document.getElementById("past-chats-row");
  const chatControlsRow = document.getElementById("chat-controls");
  const headerBar = document.querySelector(".header_bar");

  // Function to check if the device is mobile
  function isMobile() {
    return window.innerWidth <= 924;
  }

  function handleIndividualSidebarClose(event) {
    const target = event.target;
    const navigationToggle = document.getElementById("navigation-toggle");
    const pastChatsToggle = document.getElementById("past-chats-toggle");
    const chatControlsToggle = document.getElementById("chat-controls-toggle");

    // Close navigation bar if click is outside and it is open
    if (!headerBar.contains(target) && !headerBar.classList.contains("sidebar-hidden")) {
      toggleSidebar(headerBar, navigationToggle, true);
    }

    // Close past chats row if click is outside and it is open
    if (!pastChatsRow.contains(target) && !pastChatsRow.classList.contains("sidebar-hidden")) {
      toggleSidebar(pastChatsRow, pastChatsToggle, true);
    }

    // Close chat controls row if click is outside and it is open
    if (!chatControlsRow.contains(target) && !chatControlsRow.classList.contains("sidebar-hidden")) {
      toggleSidebar(chatControlsRow, chatControlsToggle, true);
    }
  }

  function toggleSidebar(sidebar, toggle, forceClose = false) {
    const isCurrentlyHidden = sidebar.classList.contains("sidebar-hidden");
    const shouldClose = !isCurrentlyHidden;

    // Apply visibility classes
    sidebar.classList.toggle("sidebar-hidden", shouldClose);
    sidebar.classList.toggle("sidebar-shown", !shouldClose);

    if (sidebar === headerBar) {
      // Special handling for header bar
      document.documentElement.style.setProperty("--header-width", shouldClose ? "0px" : "112px");
      pastChatsRow.classList.toggle("negative-header", shouldClose);
      document.getElementById("past-chats-toggle").classList.toggle("negative-header", shouldClose);
      toggle.innerHTML = shouldClose ? hamburgerMenuSVG : closeMenuSVG;
    } else if (sidebar === pastChatsRow) {
      // Past chats sidebar
      toggle.classList.toggle("past-chats-closed", shouldClose);
      toggle.classList.toggle("past-chats-open", !shouldClose);
      toggle.innerHTML = shouldClose ? rightArrowSVG : leftArrowSVG;
    } else if (sidebar === chatControlsRow) {
      // Chat controls sidebar
      toggle.classList.toggle("chat-controls-closed", shouldClose);
      toggle.classList.toggle("chat-controls-open", !shouldClose);
      toggle.innerHTML = shouldClose ? leftArrowSVG : rightArrowSVG;
    }

    // Mobile handling
    if (isMobile()) {
      sidebar.classList.toggle("sidebar-shown", !shouldClose);
    }
  }

  // Function to initialize sidebars
  function initializeSidebars() {
    const isOnMobile = isMobile();
    const pastChatsToggle = document.getElementById("past-chats-toggle");
    const chatControlsToggle = document.getElementById("chat-controls-toggle");
    const navigationToggle = document.getElementById("navigation-toggle");

    if (isOnMobile) {
      // Mobile state: Hide sidebars and set closed states
      [pastChatsRow, chatControlsRow, headerBar].forEach(el => {
        el.classList.add("sidebar-hidden");
        el.classList.remove("sidebar-shown");
      });

      document.documentElement.style.setProperty("--header-width", "0px");
      pastChatsRow.classList.add("negative-header");
      pastChatsToggle.classList.add("negative-header", "past-chats-closed");
      pastChatsToggle.classList.remove("past-chats-open");

      [chatControlsToggle, navigationToggle].forEach(el => {
        el.classList.add("chat-controls-closed");
        el.classList.remove("chat-controls-open");
      });

      pastChatsToggle.innerHTML = rightArrowSVG;
      chatControlsToggle.innerHTML = leftArrowSVG;
      navigationToggle.innerHTML = hamburgerMenuSVG;
    } else {
      // Desktop state: Show sidebars and set open states
      [pastChatsRow, chatControlsRow].forEach(el => {
        el.classList.remove("sidebar-hidden", "sidebar-shown");
      });

      pastChatsToggle.classList.add("past-chats-open");
      pastChatsToggle.classList.remove("past-chats-closed");

      [chatControlsToggle, navigationToggle].forEach(el => {
        el.classList.add("chat-controls-open");
        el.classList.remove("chat-controls-closed");
      });

      pastChatsToggle.innerHTML = leftArrowSVG;
      chatControlsToggle.innerHTML = rightArrowSVG;
      navigationToggle.innerHTML = closeMenuSVG;
    }
  }

  //------------------------------------------------
  // Create a top navigation bar on mobile
  //------------------------------------------------
  function createMobileTopBar() {
    const chatTab = document.getElementById("chat-tab");

    // Only create the top bar if it doesn't already exist
    if (chatTab && !chatTab.querySelector(".mobile-top-bar")) {
      const topBar = document.createElement("div");
      topBar.classList.add("mobile-top-bar");

      // Insert the top bar as the first child of chat-tab
      chatTab.appendChild(topBar);
    }
  }

  //------------------------------------------------
  // Fixes #chat-input textarea height issue
  // for devices with width <= 924px
  //------------------------------------------------
  if (isMobile()) {
    // Target the textarea
    const textarea = document.querySelector("#chat-input textarea");

    if (textarea) {
      // Simulate adding and removing a newline
      textarea.value += "\n";
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.value = textarea.value.slice(0, -1);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  // Initialize everything
  document.addEventListener("DOMContentLoaded", function() {
    // Create toggle buttons
    if (chatTab) {
      // Create past-chats-toggle div
      const pastChatsToggle = document.createElement("div");
      pastChatsToggle.id = "past-chats-toggle";
      pastChatsToggle.innerHTML = leftArrowSVG;
      pastChatsToggle.classList.add("past-chats-open");

      // Create chat-controls-toggle div
      const chatControlsToggle = document.createElement("div");
      chatControlsToggle.id = "chat-controls-toggle";
      chatControlsToggle.innerHTML = rightArrowSVG;
      chatControlsToggle.classList.add("chat-controls-open");

      // Append both elements to the chat-tab
      chatTab.appendChild(pastChatsToggle);
      chatTab.appendChild(chatControlsToggle);
    }

    // Create navigation toggle div
    const navigationToggle = document.createElement("div");
    navigationToggle.id = "navigation-toggle";
    navigationToggle.innerHTML = leftArrowSVG;
    navigationToggle.classList.add("navigation-left");
    headerBar.appendChild(navigationToggle);

    // Initialize sidebars
    initializeSidebars();
    createMobileTopBar();

    // Add event listeners
    const pastChatsToggle = document.getElementById("past-chats-toggle");
    const chatControlsToggle = document.getElementById("chat-controls-toggle");

    pastChatsToggle.addEventListener("click", () => {
      const isCurrentlyOpen = !pastChatsRow.classList.contains("sidebar-hidden");
      toggleSidebar(pastChatsRow, pastChatsToggle);

      // On desktop, open/close both sidebars at the same time
      if (!isMobile()) {
        if (isCurrentlyOpen) {
          // If we just closed the left sidebar, also close the right sidebar
          if (!chatControlsRow.classList.contains("sidebar-hidden")) {
            toggleSidebar(chatControlsRow, chatControlsToggle, true);
          }
        } else {
          // If we just opened the left sidebar, also open the right sidebar
          if (chatControlsRow.classList.contains("sidebar-hidden")) {
            toggleSidebar(chatControlsRow, chatControlsToggle, false);
          }
        }
      }
    });

    chatControlsToggle.addEventListener("click", () => {
      const isCurrentlyOpen = !chatControlsRow.classList.contains("sidebar-hidden");
      toggleSidebar(chatControlsRow, chatControlsToggle);

      // On desktop, open/close both sidebars at the same time
      if (!isMobile()) {
        if (isCurrentlyOpen) {
          // If we just closed the right sidebar, also close the left sidebar
          if (!pastChatsRow.classList.contains("sidebar-hidden")) {
            toggleSidebar(pastChatsRow, pastChatsToggle, true);
          }
        } else {
          // If we just opened the right sidebar, also open the left sidebar
          if (pastChatsRow.classList.contains("sidebar-hidden")) {
            toggleSidebar(pastChatsRow, pastChatsToggle, false);
          }
        }
      }
    });

    navigationToggle.addEventListener("click", () => {
      toggleSidebar(headerBar, navigationToggle);
    });

    // Add global click handler for mobile sidebar closing
    document.addEventListener("click", function (event) {
      const target = event.target;

      // Check if the click is outside the button/menu and the menu is visible
      const menu = document.getElementById("hover-menu");
      if (menu && !isMouseOverButtonOrMenu() && menu.style.display === "flex") {
        hideMenu();
      }

      if (event.target.classList.contains("pfp_character")) {
        toggleBigPicture();
      }

      // Handle sidebar clicks on mobile
      if (isMobile()) {
        // Check if the click did NOT originate from any of the specified toggle buttons or elements
        if (
          target.closest("#navigation-toggle") !== navigationToggle &&
                    target.closest("#past-chats-toggle") !== pastChatsToggle &&
                    target.closest("#chat-controls-toggle") !== chatControlsToggle &&
                    target.closest(".header_bar") !== headerBar &&
                    target.closest("#past-chats-row") !== pastChatsRow &&
                    target.closest("#chat-controls") !== chatControlsRow
        ) {
          handleIndividualSidebarClose(event);
        }
      }
    });
  });

  // Helper functions for hover menu (defined elsewhere)
  function isMouseOverButtonOrMenu() {
    const menu = document.getElementById("hover-menu");
    const button = document.getElementById("hover-element-button");
    return menu && button && (menu.matches(":hover") || button.matches(":hover"));
  }

  function hideMenu() {
    const menu = document.getElementById("hover-menu");
    if (menu) {
      menu.style.display = "none";
    }
  }

  function toggleBigPicture() {
    // This function is defined in chat-features.js
    if (window.toggleBigPicture) {
      window.toggleBigPicture();
    }
  }

})();
