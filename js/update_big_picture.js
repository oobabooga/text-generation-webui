function updateBigPicture() {
  var existingElement = document.querySelector(".bigProfilePicture");
  if (existingElement) {
    var timestamp = new Date().getTime();
    existingElement.src = "/file/cache/pfp_character.png?time=" + timestamp;
  }
}
