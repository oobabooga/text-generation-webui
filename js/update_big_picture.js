function updateBigPicture() {
  var existingElement = document.querySelector(".bigProfilePicture");
  if (existingElement) {
    var timestamp = new Date().getTime();
    existingElement.src = "/gradio_api/file=user_data/cache/pfp_character.png?time=" + timestamp;
  }
}
