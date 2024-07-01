console.log("Whisper STT script loaded");

let mediaRecorder;
let audioChunks = [];

window.startRecording = function() {
    console.log("Start recording function called");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("getUserMedia not supported on your browser!");
        return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            console.log("Got audio stream");
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = []; // Reset audio chunks
            mediaRecorder.start();
            console.log("MediaRecorder started");

            mediaRecorder.addEventListener("dataavailable", event => {
                console.log("Data available event, data size: ", event.data.size);
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                console.log("MediaRecorder stopped");
                if (audioChunks.length > 0) {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    console.log("Audio blob created, size: ", audioBlob.size);
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = function() {
                        const base64data = reader.result;
                        console.log("Audio converted to base64, length: ", base64data.length);
                        
                        const audioBase64Input = gradioApp().querySelector('#audio-base64 textarea');
                        if (audioBase64Input) {
                            console.log("Found audio-base64 textarea");
                            audioBase64Input.value = base64data;
                            audioBase64Input.dispatchEvent(new Event("input", { bubbles: true }));
                            audioBase64Input.dispatchEvent(new Event("change", { bubbles: true }));
                            console.log("Updated textarea with base64 data");
                        } else {
                            console.error("Could not find audio-base64 textarea");
                        }
                    }
                } else {
                    console.error("No audio data recorded");
                }
            });

            recButton.style.display = "none";
            stopRecButton.style.display = "inline-block";
        })
        .catch(error => {
            console.error("Error accessing the microphone", error);
        });
}

window.stopRecording = function() {
    console.log("Stop recording function called");
    if (mediaRecorder && mediaRecorder.state === "recording") {
        console.log("Stopping MediaRecorder");
        mediaRecorder.stop();
        
        recButton.style.display = "inline-block";
        stopRecButton.style.display = "none";
    } else {
        console.log("MediaRecorder not recording, no action taken");
    }
}

function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const gradioShadowRoot = elems.length == 0 ? null : elems[0].shadowRoot;
    return gradioShadowRoot ? gradioShadowRoot : document;
}

// Create and add the Rec button
var recButton = document.createElement('button');
recButton.innerText = "Rec.";
recButton.style.marginLeft = "-10px";
recButton.style.padding = "5px 10px";
recButton.style.border = "none";
recButton.style.borderRadius = "5px";
recButton.style.backgroundColor = "#4CAF50";
recButton.style.color = "white";
recButton.style.cursor = "pointer";
recButton.addEventListener('click', window.startRecording);

// Create the stop button
var stopRecButton = document.createElement('button');
stopRecButton.innerText = "Stop";
stopRecButton.style.marginLeft = "5px";
stopRecButton.style.padding = "5px 10px";
stopRecButton.style.border = "none";
stopRecButton.style.borderRadius = "5px";
stopRecButton.style.backgroundColor = "#f44336";
stopRecButton.style.color = "white";
stopRecButton.style.cursor = "pointer";
stopRecButton.style.display = "none";
stopRecButton.addEventListener('click', window.stopRecording);

// Wait for Gradio to finish loading
function onGradioLoaded() {
    console.log("Gradio loaded, setting up button listeners");
    var generate_button = gradioApp().querySelector("#Generate");
    if (generate_button) {
        generate_button.insertAdjacentElement("afterend", stopRecButton);
        generate_button.insertAdjacentElement("afterend", recButton);
    } else {
        console.log("Generate button not found");
    }
}

// Check periodically if Gradio has loaded
const gradioLoadedCheckInterval = setInterval(() => {
    if (gradioApp().querySelector("#Generate")) {
        clearInterval(gradioLoadedCheckInterval);
        onGradioLoaded();
    }
}, 100);