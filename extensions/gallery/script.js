let gallery_element = document.getElementById('gallery-extension');
let chat_mode_element = document.getElementById('chat-mode');

let extensions_block = gallery_element.parentElement;
let extensions_block_size = extensions_block.childNodes.length;
let gallery_only = (extensions_block_size == 5);

main_parent.addEventListener('click', function(e) {
    let chat_visible = (chat_tab.offsetHeight > 0 && chat_tab.offsetWidth > 0);
    let chat_mode_visible = (chat_mode_element.offsetHeight > 0 && chat_mode_element.offsetWidth > 0);
    let notebook_visible = (notebook_tab.offsetHeight > 0 && notebook_tab.offsetWidth > 0);
    let default_visible = (default_tab.offsetHeight > 0 && default_tab.offsetWidth > 0);

    // Only show this extension in the Chat tab
    if (chat_visible) {
        if (chat_mode_visible) {
            gallery_element.style.display = 'block';
            if (gallery_only) {
                extensions_block.style.display = '';
            }
        } else {
            gallery_element.style.display = 'none';
            extensions_block.style.display = 'none';
        }
    } else {
        gallery_element.style.display = 'none';
        if (gallery_only) {
            extensions_block.style.display = 'none';
        }
        else {
            extensions_block.style.display = '';
        }
    }
});
