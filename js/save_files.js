// Functions for downloading JSON files
function getCurrentTimestamp() {
  const now = new Date();
  const timezoneOffset = now.getTimezoneOffset() * 60000; // Convert minutes to milliseconds
  const localTime = new Date(now.getTime() - timezoneOffset);
  return localTime.toISOString().replace(/[-:]/g, "").slice(0, 15);
}

function saveFile(contents, filename) {
  const element = document.createElement("a");
  element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(contents));
  element.setAttribute("download", filename);
  element.style.display = "none";
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}

function saveHistory(history, character, mode) {
  let path;

  if (["chat", "chat-instruct"].includes(mode) && character && character.trim() !== "") {
    path = `history_${character}_${getCurrentTimestamp()}.json`;
  } else {
    path = `history_${mode || "unknown"}_${getCurrentTimestamp()}.json`;
  }

  saveFile(history, path);
}

function saveSession(session) {
  const path = `session_${getCurrentTimestamp()}.json`;
  saveFile(session, path);
}
