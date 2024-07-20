console.log("Whisper STT script loaded");

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

window.startStopRecording = function() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.error("getUserMedia not supported on your browser!");
    return;
  }

  if (isRecording == false) {
    //console.log("Start recording function called");
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        //console.log("Got audio stream");
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = []; // Reset audio chunks
        mediaRecorder.start();
        //console.log("MediaRecorder started");
        recButton.icon;
        recordButton.innerHTML = recButton.innerHTML = "Stop";
        isRecording = true;

        mediaRecorder.addEventListener("dataavailable", event => {
          //console.log("Data available event, data size: ", event.data.size);
          audioChunks.push(event.data);
        });
                
        mediaRecorder.addEventListener("stop", () => {
          //console.log("MediaRecorder stopped");
          if (audioChunks.length > 0) {
            const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
            //console.log("Audio blob created, size: ", audioBlob.size);
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = function() {
              const base64data = reader.result;
              //console.log("Audio converted to base64, length: ", base64data.length);
                            
              const audioBase64Input = document.querySelector("#audio-base64 textarea");
              if (audioBase64Input) {
                audioBase64Input.value = base64data;
                audioBase64Input.dispatchEvent(new Event("input", { bubbles: true }));
                audioBase64Input.dispatchEvent(new Event("change", { bubbles: true }));
                //console.log("Updated textarea with base64 data");
              } else {
                console.error("Could not find audio-base64 textarea");
              }
            };
          } else {
            console.error("No audio data recorded for Whisper");
          }
        });
      });
  } else {
    //console.log("Stopping MediaRecorder");
    recordButton.innerHTML = recButton.innerHTML = "Rec.";
    isRecording = false;
    mediaRecorder.stop();
  }
};

const recordButton = gradioApp().querySelector("#record-button");
recordButton.addEventListener("click", window.startStopRecording);


function gradioApp() {
  const elems = document.getElementsByTagName("gradio-app");
  const gradioShadowRoot = elems.length == 0 ? null : elems[0].shadowRoot;
  return gradioShadowRoot ? gradioShadowRoot : document;
}


// extra rec button next to generate button
var recButton = recordButton.cloneNode(true);
var generate_button = document.getElementById("Generate");
generate_button.insertAdjacentElement("afterend", recButton);

recButton.style.setProperty("margin-left", "-10px");
recButton.innerHTML = "Rec.";

recButton.addEventListener("click", function() {
  recordButton.click();
});
