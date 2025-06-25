function scrollToTop() {
  window.scrollTo({ top: 0 });
}

function findButtonsByText(buttonText) {
  const buttons = document.getElementsByTagName("button");
  const matchingButtons = [];

  for (let i = 0; i < buttons.length; i++) {
    if (buttons[i].textContent.trim() === buttonText) {
      matchingButtons.push(buttons[i]);
    }
  }

  return matchingButtons;
}

function switch_to_chat() {
  document.getElementById("chat-tab-button").click();
  scrollToTop();
}

function switch_to_notebook() {
  document.getElementById("notebook-parent-tab-button").click();
  findButtonsByText("Raw")[1].click();
  scrollToTop();
}

function switch_to_generation_parameters() {
  document.getElementById("parameters-button").click();
  findButtonsByText("Generation")[0].click();
  scrollToTop();
}

function switch_to_character() {
  document.getElementById("character-tab-button").click();
  scrollToTop();
}
