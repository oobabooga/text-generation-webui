let gallery_element = document.getElementById('gallery-extension');
let chat_mode_element = document.getElementById('chat-mode');

let extensions_block = document.getElementById('extensions');
let extensions_block_size = extensions_block.childNodes.length;
let gallery_only = (extensions_block_size == 5);

function gotoFirstPage() {
    const firstPageButton = gallery_element.querySelector('.paginate > button');
    if (firstPageButton) {
        firstPageButton.click();
    }
}

document.querySelector('.header_bar').addEventListener('click', function(event) {
    if (event.target.tagName === 'BUTTON') {
        const buttonText = event.target.textContent.trim();

        let chat_visible = (buttonText == 'Chat');
        let default_visible = (buttonText == 'Default');
        let notebook_visible = (buttonText == 'Notebook');
        let chat_mode_visible = (chat_mode_element.offsetHeight > 0 && chat_mode_element.offsetWidth > 0);

        // Only show this extension in the Chat tab
        if (chat_visible) {
            if (chat_mode_visible) {
                gallery_element.style.display = 'block';
                extensions_block.style.display = '';
            } else {
                gallery_element.style.display = 'none';
                extensions_block.style.display = 'none';
            }
        } else {
            gallery_element.style.display = 'none';
            if (gallery_only) {
                extensions_block.style.display = 'none';
            }
        }
    }
});
