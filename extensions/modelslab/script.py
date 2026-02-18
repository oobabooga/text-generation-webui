import base64
import io
import re
import time
import json
from datetime import date
from pathlib import Path

import gradio as gr
import requests
from PIL import Image

from modules import shared
from modules.ui import create_refresh_button

# ModelsLab API parameters - can be customized in settings.json
params = {
    # API Configuration
    'api_key': '',
    'base_url': 'https://modelslab.com/api/v6',
    'model': 'flux',
    
    # Generation Parameters
    'prompt_prefix': '(Masterpiece:1.1), detailed, intricate, colorful',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'width': 1024,
    'height': 1024,
    'steps': 25,
    'cfg_scale': 7.5,
    'seed': -1,
    
    # Behavior Settings
    'mode': 1,  # 0=Manual, 1=Interactive, 2=Always-on
    'save_images': True,
    'enhance_prompt': True,
    'safety_checker': True,
    
    # Text Integration
    'textgen_prefix': 'Please provide a detailed and vivid description of [subject]',
    'trigger_words': ['image', 'picture', 'photo', 'generate', 'draw', 'create', 'show me'],
    
    # Available Models
    'available_models': ['flux', 'sdxl', 'playground-v2', 'stable-diffusion'],
    'model_info': {
        'flux': {'name': 'Flux', 'desc': 'Best quality, prompt adherence (~$0.018)', 'speed': '2-4s'},
        'sdxl': {'name': 'Stable Diffusion XL', 'desc': 'Artistic, creative style (~$0.015)', 'speed': '3-5s'},
        'playground-v2': {'name': 'Playground v2.5', 'desc': 'UI mockups, designs (~$0.016)', 'speed': '2-3s'},
        'stable-diffusion': {'name': 'Stable Diffusion', 'desc': 'General purpose, fast (~$0.012)', 'speed': '2-4s'}
    }
}

# Global state
picture_response = False
current_generation_task = None

def load_settings():
    """Load settings from webui settings.json"""
    try:
        settings_file = Path('settings.json')
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                data = json.load(f)
                modelslab_settings = data.get('modelslab_api_pictures', {})
                for key, value in modelslab_settings.items():
                    if key in params:
                        params[key] = value
    except Exception as e:
        print(f"ModelsLab: Could not load settings: {e}")

def save_settings():
    """Save current params to webui settings.json"""
    try:
        settings_file = Path('settings.json')
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        data['modelslab_api_pictures'] = params.copy()
        
        with open(settings_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"ModelsLab: Could not save settings: {e}")

class ModelsLabClient:
    """ModelsLab API client with async support"""
    
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_image(self, prompt, **kwargs):
        """Generate image using ModelsLab API"""
        if not self.api_key:
            raise Exception("ModelsLab API key not configured")
        
        payload = {
            "key": self.api_key,
            "model_id": kwargs.get('model', 'flux'),
            "prompt": prompt,
            "negative_prompt": kwargs.get('negative_prompt', ''),
            "width": kwargs.get('width', 1024),
            "height": kwargs.get('height', 1024),
            "samples": 1,
            "num_inference_steps": kwargs.get('steps', 25),
            "guidance_scale": kwargs.get('cfg_scale', 7.5),
            "enhance_prompt": kwargs.get('enhance_prompt', True),
            "safety_checker": kwargs.get('safety_checker', True)
        }
        
        # Add seed if specified
        if kwargs.get('seed') and kwargs.get('seed') != -1:
            payload["seed"] = kwargs.get('seed')
        
        try:
            response = self.session.post(
                f"{self.base_url}/images/text2img",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
            
            return self.handle_response(response.json())
            
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error. Check your internet connection.")
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")
    
    def handle_response(self, data):
        """Handle API response including async polling"""
        if data.get('status') == 'success':
            output = data.get('output', [])
            if output:
                return output[0]  # Return first image URL
            else:
                raise Exception("No image generated")
                
        elif data.get('status') == 'processing':
            task_id = data.get('id')
            if task_id:
                return self.poll_result(task_id)
            else:
                raise Exception("Generation started but no task ID received")
                
        elif data.get('status') == 'error':
            raise Exception(f"Generation failed: {data.get('message', 'Unknown error')}")
        
        else:
            # Direct image URL response
            output = data.get('output', [])
            if output:
                return output[0]
            else:
                raise Exception("Unexpected API response format")
    
    def poll_result(self, task_id):
        """Poll for async generation completion"""
        max_attempts = 30  # Maximum 1 minute wait
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = self.session.post(
                    f"{self.base_url}/images/fetch/{task_id}",
                    json={"key": self.api_key},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('status') == 'success':
                        output = data.get('output', [])
                        if output:
                            return output[0]
                        else:
                            raise Exception("Generation completed but no image received")
                            
                    elif data.get('status') == 'failed':
                        raise Exception(f"Generation failed: {data.get('message', 'Unknown error')}")
                    
                    # Still processing, continue polling
                    time.sleep(2)
                    attempt += 1
                
                else:
                    raise Exception(f"Polling failed: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                attempt += 1
                time.sleep(2)
                continue
        
        raise Exception("Generation timed out after 1 minute")

def remove_surrounded_chars(string):
    """Remove text between asterisks (actions)"""
    return re.sub(r'\*[^\*]*?(\*|$)', '', string)

def contains_trigger_words(string):
    """Check if string contains image generation trigger words"""
    string = remove_surrounded_chars(string).lower()
    
    # Check for trigger word patterns
    for trigger in params['trigger_words']:
        if trigger.lower() in string:
            return True
    
    # More sophisticated patterns
    patterns = [
        r'\b(send|mail|message|me)\b.+?\b(image|pic(ture)?|photo|snap(shot)?|selfie)\b',
        r'\b(generate|create|draw|make)\b.+?\b(image|picture|art|drawing)\b',
        r'\b(show|display)\s+me\b.+?\b(image|picture|photo)\b'
    ]
    
    for pattern in patterns:
        if re.search(pattern, string, re.IGNORECASE):
            return True
    
    return False

def extract_subject(string):
    """Extract subject from user request"""
    string = remove_surrounded_chars(string).strip()
    
    # Try various extraction patterns
    patterns = [
        r'(?:image|picture|photo|drawing)\s+of\s+(.+?)(?:\.|$|,)',
        r'(?:generate|create|draw|make)\s+(?:an?|the)?\s*(.+?)(?:\.|$|,)',
        r'(?:show|display)\s+me\s+(?:an?|the)?\s*(.+?)(?:\.|$|,)',
        r'(?:want|like)\s+to\s+see\s+(?:an?|the)?\s*(.+?)(?:\.|$|,)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            if subject:
                return subject
    
    # Fallback: return the whole string cleaned up
    return re.sub(r'\b(image|picture|photo|generate|create|draw|show|me|of|an?|the)\b', '', 
                  string, flags=re.IGNORECASE).strip()

def schedule_image_generation(prompt):
    """Mark that next response should include an image"""
    global picture_response, current_generation_task
    picture_response = True
    current_generation_task = prompt
    print(f"ModelsLab: Scheduled image generation for: {prompt[:100]}...")

def state_modifier(state):
    """Modify generation state when image generation is scheduled"""
    global picture_response
    if picture_response:
        state['stream'] = False  # Disable streaming for image generation
    return state

def input_modifier(string):
    """Process user input to detect image generation requests"""
    global params
    
    if not params['api_key']:
        return string  # No API key, skip processing
    
    if params['mode'] == 0:  # Manual mode only
        return string
    
    # Check for trigger words
    if contains_trigger_words(string):
        subject = extract_subject(string)
        
        if subject:
            # Generate enhanced prompt using textgen prefix
            enhanced_prompt = params['textgen_prefix'].replace('[subject]', subject)
        else:
            enhanced_prompt = string
        
        # Schedule image generation
        schedule_image_generation(enhanced_prompt)
        
        # Modify response based on mode
        if params['mode'] == 2:  # Always-on mode
            return f"I'll generate an image of: {subject if subject else 'your request'}"
        elif params['mode'] == 1:  # Interactive mode
            # Let the LLM respond naturally, image will be added
            pass
    
    return string

def output_modifier(string):
    """Process model output to inject generated images"""
    global picture_response, current_generation_task
    
    if not picture_response or not current_generation_task:
        return string
    
    try:
        # Generate the image
        image_html = generate_modelslab_image(current_generation_task)
        
        # Reset state
        picture_response = False
        current_generation_task = None
        
        # Inject image into response
        if params['mode'] == 2:  # Always-on mode
            return image_html
        else:  # Interactive mode
            return f"{string}\n\n{image_html}"
    
    except Exception as e:
        # Reset state on error
        picture_response = False
        current_generation_task = None
        
        error_msg = f"<p style='color: red;'>Image generation failed: {str(e)}</p>"
        return f"{string}\n\n{error_msg}"

def generate_modelslab_image(prompt):
    """Generate image using ModelsLab API and return HTML"""
    try:
        print(f"ModelsLab: Generating image for prompt: {prompt[:100]}...")
        
        # Initialize client
        client = ModelsLabClient(params['api_key'], params['base_url'])
        
        # Prepare full prompt
        if params['prompt_prefix']:
            full_prompt = f"{params['prompt_prefix']}, {prompt}"
        else:
            full_prompt = prompt
        
        print(f"ModelsLab: Full prompt: {full_prompt[:150]}...")
        
        # Generate image
        image_url = client.generate_image(
            prompt=full_prompt,
            negative_prompt=params['negative_prompt'],
            model=params['model'],
            width=params['width'],
            height=params['height'],
            steps=params['steps'],
            cfg_scale=params['cfg_scale'],
            seed=params['seed'] if params['seed'] != -1 else None,
            enhance_prompt=params['enhance_prompt'],
            safety_checker=params['safety_checker']
        )
        
        print(f"ModelsLab: Image generated successfully: {image_url}")
        
        # Save image if enabled
        if params['save_images']:
            try:
                save_generated_image(image_url, prompt)
            except Exception as e:
                print(f"ModelsLab: Could not save image: {e}")
        
        # Create HTML for display
        html = create_image_html(image_url, prompt)
        return html
        
    except Exception as e:
        print(f"ModelsLab: Generation error: {e}")
        raise e

def save_generated_image(image_url, prompt):
    """Save generated image to local storage"""
    try:
        # Create output directory
        output_dir = Path("outputs/modelslab")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Generate filename
        timestamp = date.today().strftime("%Y%m%d")
        safe_prompt = re.sub(r'[^\w\s-]', '', prompt[:50]).strip()
        safe_prompt = re.sub(r'[\s-]+', '_', safe_prompt)
        filename = f"{timestamp}_{safe_prompt}_{params['model']}.png"
        
        # Save file
        filepath = output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"ModelsLab: Image saved to {filepath}")
        
    except Exception as e:
        print(f"ModelsLab: Save error: {e}")

def create_image_html(image_url, prompt):
    """Create HTML for displaying generated image"""
    model_name = params['model_info'].get(params['model'], {}).get('name', params['model'])
    
    html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin: 10px 0; background: #f9f9f9;">
        <div style="margin-bottom: 10px; font-size: 14px; color: #666;">
            <strong>Generated with ModelsLab {model_name}</strong>
        </div>
        <img src="{image_url}" alt="Generated image" style="max-width: 100%; height: auto; border-radius: 4px;" />
        <div style="margin-top: 8px; font-size: 12px; color: #888; font-style: italic;">
            Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}
        </div>
    </div>
    """
    
    return html

def validate_api_key(api_key):
    """Validate API key by making a test request"""
    if not api_key:
        return False, "API key is required"
    
    try:
        client = ModelsLabClient(api_key, params['base_url'])
        # Make a minimal test request
        test_payload = {
            "key": api_key,
            "model_id": "flux",
            "prompt": "test",
            "width": 256,
            "height": 256,
            "samples": 1
        }
        
        response = requests.post(
            f"{params['base_url']}/images/text2img",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'error' in data:
                return False, f"API Error: {data.get('error', 'Unknown error')}"
            return True, "API key is valid"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:100]}"
            
    except Exception as e:
        return False, f"Validation failed: {str(e)}"

def update_param(key, value):
    """Update parameter and save settings"""
    params[key] = value
    save_settings()
    print(f"ModelsLab: Updated {key} = {value}")

def test_generation():
    """Test image generation with current settings"""
    if not params['api_key']:
        return "<p style='color: red;'>Please configure your API key first</p>"
    
    try:
        test_prompt = "A cute cat sitting in a garden, photorealistic"
        html = generate_modelslab_image(test_prompt)
        return html
    except Exception as e:
        return f"<p style='color: red;'>Test generation failed: {str(e)}</p>"

def ui():
    """Create Gradio interface for ModelsLab extension"""
    load_settings()  # Load settings when UI is created
    
    with gr.Accordion("🎨 ModelsLab API Settings", open=True):
        with gr.Row():
            api_key_input = gr.Textbox(
                label="API Key",
                type="password",
                value=params['api_key'],
                placeholder="Enter your ModelsLab API key",
                interactive=True
            )
            
            validate_button = gr.Button("Validate Key", size="sm")
            validation_output = gr.HTML(visible=False)
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=params['available_models'],
                value=params['model'],
                info="Choose generation model"
            )
            
            mode_radio = gr.Radio(
                label="Generation Mode",
                choices=[
                    ("Manual Only", 0),
                    ("Interactive (Recommended)", 1), 
                    ("Always Generate", 2)
                ],
                value=params['mode'],
                info="0=Manual commands only, 1=Trigger words, 2=Always generate"
            )
        
        with gr.Accordion("💡 Model Information", open=False):
            model_info_html = gr.HTML(
                value="<br>".join([
                    f"<strong>{info['name']}</strong>: {info['desc']} | Speed: {info['speed']}"
                    for info in params['model_info'].values()
                ])
            )
        
        with gr.Accordion("⚙️ Generation Settings", open=False):
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256, maximum=1536, step=64, 
                    value=params['width'], label="Width"
                )
                height_slider = gr.Slider(
                    minimum=256, maximum=1536, step=64,
                    value=params['height'], label="Height"
                )
            
            with gr.Row():
                steps_slider = gr.Slider(
                    minimum=10, maximum=50, step=1,
                    value=params['steps'], label="Steps"
                )
                cfg_slider = gr.Slider(
                    minimum=1.0, maximum=20.0, step=0.1,
                    value=params['cfg_scale'], label="CFG Scale"
                )
            
            with gr.Row():
                seed_input = gr.Number(
                    value=params['seed'], label="Seed (-1 for random)",
                    precision=0
                )
                
            with gr.Column():
                prompt_prefix_input = gr.Textbox(
                    label="Prompt Prefix",
                    value=params['prompt_prefix'],
                    placeholder="Added to beginning of every prompt"
                )
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    value=params['negative_prompt'],
                    placeholder="What to avoid in generation"
                )
            
            with gr.Row():
                enhance_prompt_check = gr.Checkbox(
                    label="Enhance Prompt", value=params['enhance_prompt']
                )
                safety_checker_check = gr.Checkbox(
                    label="Safety Checker", value=params['safety_checker']
                )
                save_images_check = gr.Checkbox(
                    label="Save Images", value=params['save_images']
                )
        
        with gr.Row():
            test_button = gr.Button("🧪 Test Generation", variant="secondary")
            test_output = gr.HTML()
    
    # Event handlers
    def validate_key(key):
        if not key:
            return gr.update(visible=True, value="<p style='color: red;'>Please enter an API key</p>")
        
        is_valid, message = validate_api_key(key)
        color = "green" if is_valid else "red"
        return gr.update(
            visible=True, 
            value=f"<p style='color: {color};'>{message}</p>"
        )
    
    # Connect event handlers
    api_key_input.change(lambda x: update_param('api_key', x), inputs=[api_key_input])
    model_dropdown.change(lambda x: update_param('model', x), inputs=[model_dropdown])
    mode_radio.change(lambda x: update_param('mode', x), inputs=[mode_radio])
    
    width_slider.change(lambda x: update_param('width', x), inputs=[width_slider])
    height_slider.change(lambda x: update_param('height', x), inputs=[height_slider])
    steps_slider.change(lambda x: update_param('steps', x), inputs=[steps_slider])
    cfg_slider.change(lambda x: update_param('cfg_scale', x), inputs=[cfg_slider])
    seed_input.change(lambda x: update_param('seed', x), inputs=[seed_input])
    
    prompt_prefix_input.change(lambda x: update_param('prompt_prefix', x), inputs=[prompt_prefix_input])
    negative_prompt_input.change(lambda x: update_param('negative_prompt', x), inputs=[negative_prompt_input])
    
    enhance_prompt_check.change(lambda x: update_param('enhance_prompt', x), inputs=[enhance_prompt_check])
    safety_checker_check.change(lambda x: update_param('safety_checker', x), inputs=[safety_checker_check])
    save_images_check.change(lambda x: update_param('save_images', x), inputs=[save_images_check])
    
    validate_button.click(validate_key, inputs=[api_key_input], outputs=[validation_output])
    test_button.click(test_generation, outputs=[test_output])

# Initialize settings on load
load_settings()
print("ModelsLab API extension loaded successfully!")
print(f"Current model: {params['model']}, Mode: {params['mode']}")
if not params['api_key']:
    print("⚠️  Please configure your ModelsLab API key in the extension settings")