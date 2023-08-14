let chat_tab = document.getElementById('chat-tab');
let main_parent = chat_tab.parentNode;

function scrollToTop() {
    window.scrollTo({
        top: 0,
        // behavior: 'smooth'
    });
}

function switch_to_chat() {
    let chat_tab_button = main_parent.childNodes[0].childNodes[1];
    chat_tab_button.click();
    scrollToTop();
}

function switch_to_default() {
    let default_tab_button = main_parent.childNodes[0].childNodes[4];
    default_tab_button.click();
    scrollToTop();
}

function switch_to_notebook() {
    let notebook_tab_button = main_parent.childNodes[0].childNodes[7];
    notebook_tab_button.click();
    scrollToTop();
}

function switch_to_generation_parameters() {
    let parameters_tab_button = main_parent.childNodes[0].childNodes[10];
    let generation_tab_button = document.getElementById('character-menu').parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.childNodes[0].childNodes[1];
    parameters_tab_button.click();
    generation_tab_button.click();
    scrollToTop();
}

function switch_to_character() {
    let parameters_tab_button = main_parent.childNodes[0].childNodes[10];
    let character_tab_button = document.getElementById('character-menu').parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.childNodes[0].childNodes[4];
    parameters_tab_button.click();
    character_tab_button.click();
    scrollToTop();
}
