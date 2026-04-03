function scrollToTop() {
  window.scrollTo({ top: 0 });
}

function findButtonsByText(buttonText, container = document) {
  return Array.from(container.getElementsByTagName("button"))
    .filter(btn => btn.textContent.trim() === buttonText);
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

function switch_to_image_ai_generate() {
  const container = document.querySelector("#image-ai-tab");
  const generateBtn = findButtonsByText("Generate", container)[0];
  if (generateBtn) {
    generateBtn.click();
  }

  scrollToTop();
}
