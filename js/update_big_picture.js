function updateBigPicture() {
  var existingElement = document.querySelector(".bigProfilePicture");
  if (existingElement) {
    existingElement.src = getProfilePictureUrl();
  }
}
