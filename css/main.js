document.getElementById("main").parentNode.childNodes[0].classList.add("header_bar");
document.getElementById("main").parentNode.style = "padding: 0; margin: 0";
document.getElementById("main").parentNode.parentNode.parentNode.style = "padding: 0";

// Get references to the elements
let main = document.getElementById('main');
let main_parent = main.parentNode;
let extensions = document.getElementById('extensions');

// Add an event listener to the main element
main_parent.addEventListener('click', function(e) {
    // Check if the main element is visible
    if (main.offsetHeight > 0 && main.offsetWidth > 0) {
        extensions.style.display = 'flex';
    } else {
        extensions.style.display = 'none';
    }
});
