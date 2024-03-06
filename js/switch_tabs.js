let chat_tab = document.getElementById("chat-tab");
let main_parent = chat_tab.parentNode;

function scrollToTop() {
  window.scrollTo({
    top: 0,
    // behavior: 'smooth'
  });
}

function findButtonsByText(buttonText) {
  const buttons = document.getElementsByTagName("button");
  const matchingButtons = [];
  buttonText = buttonText.trim();

  for (let i = 0; i < buttons.length; i++) {
    const button = buttons[i];
    const buttonInnerText = button.textContent.trim();

    if (buttonInnerText === buttonText) {
      matchingButtons.push(button);
    }
  }

  return matchingButtons;
}

function switch_to_chat() {
  let chat_tab_button = main_parent.childNodes[0].childNodes[1];
  chat_tab_button.click();
  scrollToTop();
}

function switch_to_default() {
  let default_tab_button = main_parent.childNodes[0].childNodes[4];
  default_tab_button.click();
  scrollToTop();
}

function switch_to_notebook() {
  let notebook_tab_button = main_parent.childNodes[0].childNodes[7];
  notebook_tab_button.click();
  findButtonsByText("Raw")[1].click();
  scrollToTop();
}

function switch_to_generation_parameters() {
  let parameters_tab_button = main_parent.childNodes[0].childNodes[10];
  parameters_tab_button.click();
  findButtonsByText("Generation")[0].click();
  scrollToTop();
}

function switch_to_character() {
  let parameters_tab_button = main_parent.childNodes[0].childNodes[10];
  parameters_tab_button.click();
  findButtonsByText("Character")[0].click();
  scrollToTop();
}
