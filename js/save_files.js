// Functions for downloading JSON files
function getCurrentTimestamp() {
  const now = new Date();
  const timezoneOffset = now.getTimezoneOffset() * 60000; // Convert to milliseconds
  const localTime = new Date(now.getTime() - timezoneOffset);
  const formattedTimestamp = localTime.toISOString().replace(/[-:]/g, "").slice(0, 15);
  return formattedTimestamp;
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
  let path = null;

  if (["chat", "chat-instruct"].includes(mode) && character && character.trim() !== "") {
    path = `history_${character}_${getCurrentTimestamp()}.json`;
  } else {
    try {
      path = `history_${mode}_${getCurrentTimestamp()}.json`;
    } catch (error) {
      path = `history_${getCurrentTimestamp()}.json`;
    }
  }
  saveFile(history, path);
}

function saveSession(session) {
  let path = null;

  path = `session_${getCurrentTimestamp()}.json`;
  saveFile(session, path);
}
