import gradio as gr

refresh_symbol = '\U0001f504'  # 🔄

css = """
.tabs.svelte-710i53 {
    margin-top: 0
}
.py-6 {
    padding-top: 2.5rem
}
.dark #refresh-button {
    background-color: #ffffff1f;
}
#refresh-button {
    flex: none;
     margin: 0;
     padding: 0;
     min-width: 50px;
     border: none;
     box-shadow: none;
     border-radius: 10px;
     background-color: #0000000d;
}
#download-label, #upload-label {
    min-height: 0
}
#accordion {
}
"""

chat_css = """
.h-\[40vh\], .wrap.svelte-byatnx.svelte-byatnx.svelte-byatnx {
    height: 66.67vh
}
.gradio-container {
    max-width: 800px;
     margin-left: auto;
     margin-right: auto
}
.w-screen {
    width: unset
}
div.svelte-362y77>*, div.svelte-362y77>.form>* {
    flex-wrap: nowrap
}
/* fixes the API documentation in chat mode */
.api-docs.svelte-1iguv9h.svelte-1iguv9h.svelte-1iguv9h {
    display: grid;
}
"""

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button
