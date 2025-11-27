import gradio as gr
import os


def create_ui():
    with gr.Tab("Image AI", elem_id="image-ai-tab"):
        with gr.Tabs():
            # TAB 1: GENERATION STUDIO
            with gr.TabItem("Generate Images"):
                with gr.Row():
                    
                    # === LEFT COLUMN: CONTROLS ===
                    with gr.Column(scale=4, min_width=350):
                        
                        # 1. PROMPT
                        prompt = gr.Textbox(label="Prompt", placeholder="Describe your imagination...", lines=3, autofocus=True)
                        neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="Low quality...", lines=3)

                        # 2. GENERATE BUTTON
                        generate_btn = gr.Button("✨ GENERATE", variant="primary", size="lg", elem_id="gen-btn")
                        gr.HTML("<hr style='border-top: 1px solid #444; margin: 20px 0;'>")

                        # 3. DIMENSIONS
                        gr.Markdown("### 📐 Dimensions")
                        with gr.Row():
                            with gr.Column():
                                width_slider = gr.Slider(256, 2048, value=1024, step=32, label="Width")

                            with gr.Column():
                                height_slider = gr.Slider(256, 2048, value=1024, step=32, label="Height")

                        preset_radio = gr.Radio(
                            choices=["1:1 Square", "16:9 Cinema", "9:16 Mobile", "4:3 Photo", "Custom"],
                            value="1:1 Square",
                            label="Aspect Ratio",
                            interactive=True
                        )

                        # 4. SETTINGS & BATCHING
                        gr.Markdown("### ⚙️  Config")
                        with gr.Row():
                            with gr.Column():
                                steps_slider = gr.Slider(1, 15, value=9, step=1, label="Steps")
                                cfg_slider = gr.Slider(value=0.0, label="Guidance", interactive=False, info="Locked")
                                seed_input = gr.Number(label="Seed", value=-1, precision=0, info="-1 = Random")

                            with gr.Column():
                                batch_size_parallel = gr.Slider(1, 32, value=1, step=1, label="Batch Size (VRAM Heavy)", info="Generates N images at once.")
                                batch_count_seq = gr.Slider(1, 128, value=1, step=1, label="Sequential Count (Loop)", info="Repeats the generation N times.")
                        
                    # === RIGHT COLUMN: VIEWPORT ===
                    with gr.Column(scale=6, min_width=500):
                        with gr.Column(elem_classes=["viewport-container"]):
                            output_gallery = gr.Gallery(
                                label="Output", show_label=False, columns=2, rows=2, height="80vh", object_fit="contain", preview=True 
                            )
                        with gr.Row():
                            used_seed = gr.Markdown(label="Info", interactive=False, lines=3)

            # TAB 2: HISTORY VIEWER
            with gr.TabItem("Gallery"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Gallery", elem_classes="refresh-button")

                history_gallery = gr.Gallery(
                    label="History", show_label=False, columns=6, object_fit="cover", height="auto", allow_preview=True
                )

        # === WIRING ===
        
        # Aspect Buttons
        # btn_sq.click(lambda: set_dims(1024, 1024), outputs=[width_slider, height_slider])
        # btn_port.click(lambda: set_dims(720, 1280), outputs=[width_slider, height_slider])
        # btn_land.click(lambda: set_dims(1280, 720), outputs=[width_slider, height_slider])
        # btn_wide.click(lambda: set_dims(1536, 640), outputs=[width_slider, height_slider])

        # Generation
        inputs = [prompt, neg_prompt, width_slider, height_slider, steps_slider, seed_input, batch_size_parallel, batch_count_seq]
        outputs = [output_gallery, used_seed]

        # generate_btn.click(fn=generate, inputs=inputs, outputs=outputs)
        # prompt.submit(fn=generate, inputs=inputs, outputs=outputs)
        # neg_prompt.submit(fn=generate, inputs=inputs, outputs=outputs)

        # System
        # load_btn.click(fn=load_pipeline, inputs=[backend_drop, compile_check, offload_check, gr.State("bfloat16")], outputs=None)
        
        # History
        # refresh_btn.click(fn=get_history_images, inputs=None, outputs=history_gallery)
        # Load history on app launch
        # demo.load(fn=get_history_images, inputs=None, outputs=history_gallery)


def create_event_handlers():
    pass
