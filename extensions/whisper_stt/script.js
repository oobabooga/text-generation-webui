var recButton = document.getElementsByClassName("record-button")[0].cloneNode(true);
var generate_button = document.getElementById("Generate");
generate_button.insertAdjacentElement("afterend", recButton);

recButton.style.setProperty("margin-left", "-10px");
recButton.innerText = "Rec."


recButton.addEventListener('click', function() {
    var originalRecordButton = document.getElementsByClassName("record-button")[1];
    originalRecordButton.click();

    var stopRecordButtons = document.getElementsByClassName("stop-button");
    if (stopRecordButtons.length > 1) generate_button.parentElement.removeChild(stopRecordButtons[0]);
    var stopRecordButton = document.getElementsByClassName("stop-button")[0];
    generate_button.insertAdjacentElement("afterend", stopRecordButton);

    //stopRecordButton.style.setProperty("margin-left", "-10px");
    stopRecordButton.style.setProperty("padding-right", "10px");
    recButton.style.display = "none";

    stopRecordButton.addEventListener('click', function() {
        recButton.style.display = "flex";
    });
});