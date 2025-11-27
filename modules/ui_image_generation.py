import gradio as gr
import os
from modules.utils import resolve_model_path

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

        generate_btn.click(fn=generate, inputs=inputs, outputs=outputs)
        prompt.submit(fn=generate, inputs=inputs, outputs=outputs)
        neg_prompt.submit(fn=generate, inputs=inputs, outputs=outputs)

        # System
        # load_btn.click(fn=load_pipeline, inputs=[backend_drop, compile_check, offload_check, gr.State("bfloat16")], outputs=None)
        
        # History
        # refresh_btn.click(fn=get_history_images, inputs=None, outputs=history_gallery)
        # Load history on app launch
        # demo.load(fn=get_history_images, inputs=None, outputs=history_gallery)


def generate(prompt, neg_prompt, width, height, steps, seed, batch_size_parallel, batch_count_seq):
    if engine.pipe is None:
        load_pipeline("SDPA", False, False, "bfloat16")

    if seed == -1: seed = np.random.randint(0, 2**32 - 1)

    # We use a base generator. For sequential batches, we might increment seed if desired,
    # but here we keep the base seed logic consistent.
    generator = torch.Generator("cuda").manual_seed(int(seed))

    all_images = []

    # SEQUENTIAL LOOP (Easy on VRAM)
    for i in range(batch_count_seq):
        # Update seed for subsequent batches so they aren't identical
        current_seed = seed + i
        generator.manual_seed(int(current_seed))

        # PARALLEL GENERATION (Fast, Heavy VRAM)
        # diffusers handles 'num_images_per_prompt' for parallel execution
        batch_results = engine.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            num_images_per_prompt=int(batch_size_parallel),
            generator=generator,
        ).images

        all_images.extend(batch_results)

    # Save to disk
    save_generated_images(all_images, prompt, seed)

    return all_images, seed


# --- File Saving Logic ---
def save_generated_images(images, prompt, seed):
    # Create folder structure: outputs/YYYY-MM-DD/
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("outputs", date_str)
    os.makedirs(folder_path, exist_ok=True)

    saved_paths = []

    for idx, img in enumerate(images):
        timestamp = datetime.now().strftime("%H-%M-%S")
        # Filename: Time_Seed_Index.png
        filename = f"{timestamp}_{seed}_{idx}.png"
        full_path = os.path.join(folder_path, filename)

        # Save image
        img.save(full_path)
        saved_paths.append(full_path)

        # Optional: Save prompt metadata in a text file next to it? 
        # For now, we just save the image.

    return saved_paths


# --- History Logic ---
def get_history_images():
    """Scans the outputs folder and returns all images, newest first"""
    if not os.path.exists("outputs"):
        return []

    image_files = []
    for root, dirs, files in os.walk("outputs"):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                # Get creation time for sorting
                mtime = os.path.getmtime(full_path)
                image_files.append((full_path, mtime))

    # Sort by time, newest first
    image_files.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in image_files]


def load_pipeline(attn_backend, compile_model, offload_cpu, dtype_str):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    target_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    if engine.pipe is not None and engine.config["backend"] == attn_backend:
        return gr.Info("Pipeline ready.")
    
    try:
        gr.Info(f"Loading Model ({attn_backend})...")
        pipe = ZImagePipeline.from_pretrained(
            engine.config["model_id"],
            torch_dtype=target_dtype,
            low_cpu_mem_usage=False,
        )
        if not offload_cpu: pipe.to("cuda")

        if attn_backend == "Flash Attention 2":
            pipe.transformer.set_attention_backend("flash")
        elif attn_backend == "Flash Attention 3":
            pipe.transformer.set_attention_backend("_flash_3")

        if compile_model:
            gr.Warning("Compiling... First run will be slow.")
            pipe.transformer.compile()

        if offload_cpu: pipe.enable_model_cpu_offload()

        engine.pipe = pipe
        engine.config["backend"] = attn_backend
        return gr.Success("System Ready.")
    except Exception as e:
        return gr.Error(f"Init Failed: {str(e)}")

