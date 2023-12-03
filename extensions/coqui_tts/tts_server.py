import json
import random
import time
import os
from pathlib import Path
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

##########################
#### Webserver Imports####
##########################
from fastapi import FastAPI, Form, Request, HTTPException, BackgroundTasks, Response, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

########################
#### STARTUP CHECKS ####
########################
try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    logger.error(
        "[CoquiTTS Startup] \033[91mWarning\033[0m Could not find the TTS module. Make sure to install the requirements for the coqui_tts extension."
        "[CoquiTTS Startup] \033[91mWarning\033[0m Linux / Mac:\npip install -r extensions/coqui_tts/requirements.txt\n"
        "[CoquiTTS Startup] \033[91mWarning\033[0m Windows:\npip install -r extensions\\coqui_tts\\requirements.txt\n"
        "[CoquiTTS Startup] \033[91mWarning\033[0m If you used the one-click installer, paste the command above in the terminal window launched after running the \"cmd_\" script. On Windows, that's \"cmd_windows.bat\"."
    )
    raise

#DEEPSPEED Import - Check for DeepSpeed and import it if it exists
try:
    import deepspeed
    deepspeed_installed = True
    print("[CoquiTTS Startup] DeepSpeed \033[93mDetected\033[0m")
    print("[CoquiTTS Startup] Activate DeepSpeed in Coqui settings")
except ImportError:
    deepspeed_installed = False
    print("[CoquiTTS Startup] DeepSpeed \033[93mNot Detected\033[0m. See https://github.com/microsoft/DeepSpeed") 

@asynccontextmanager
async def startup_shutdown(no_actual_value_it_demanded_something_be_here):
    await setup()
    yield  
    # Shutdown logic

###########################
#### STARTUP VARIABLES ####
###########################
#STARTUP VARIABLE - Set "device" to cuda if exists, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
#STARTUP VARIABLE - Create "this_dir" variable as the current script directory
this_dir = Path(__file__).parent.resolve()
#STARTUP VARIABLE - Import languges file for Gradio to be able to display them in the interface
with open(this_dir / 'languages.json', encoding='utf8') as f:
    languages = json.load(f)
# Create FastAPI app with lifespan
app = FastAPI(lifespan=startup_shutdown)

######################################
#### LOAD PARAMS FROM CONFIG.JSON ####
######################################
def load_config(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

config_file_path = (this_dir / 'config.json')
params = load_config(config_file_path)

#############################################
#### LOAD PARAMS FROM MODELDOWNLOAD.JSON ####
############################################
#This is used only in the instance that someone has changed their model path
# Define the path to the JSON file
modeldownload_config_file_path = this_dir / 'modeldownload.json'

#Check if the JSON file exists
if modeldownload_config_file_path.exists():
    with open(modeldownload_config_file_path, 'r') as modeldownload_config_file:
        modeldownload_settings = json.load(modeldownload_config_file)

    #Extract settings from the loaded JSON
    modeldownload_base_path = Path(modeldownload_settings.get("base_path", ""))
    modeldownload_model_path = Path(modeldownload_settings.get("model_path", ""))
else:
    #Default settings if the JSON file doesn't exist or is empty
    print("[CoquiTTS Startup] \033[91mWarning\033[0m modeldownload.config is missing so please re-download it and save it in the coquii_tts main folder.")

#####################################
#### MODEL LOADING AND UNLOADING ####
#####################################
#MODEL LOADERS Picker For API TTS, API Local, XTTSv2 Local
async def setup():
    global device
    #Set a timer to calculate load times
    generate_start_time = time.time()  # Record the start time of loading the model
    #Start loading the correct model as set by "tts_method_api_tts", "tts_method_api_local" or "tts_method_xtts_local" being True/False
    if params["tts_method_api_tts"]:
        print(f"[CoquiTTS Model] \033[94mAPI TTS Loading\033[0m {params['model_name']} \033[94minto\033[93m", device, "\033[0m")
        model = await api_load_model()
    elif params["tts_method_api_local"]:
        print(f"[CoquiTTS Model] \033[94mAPI Local Loading\033[0m {modeldownload_model_path} \033[94minto\033[93m", device, "\033[0m")
        model = await api_manual_load_model()
    elif params["tts_method_xtts_local"]:
        print(f"[CoquiTTS Model] \033[94mXTTSv2 Local Loading\033[0m {modeldownload_model_path} \033[94minto\033[93m", device, "\033[0m")
        model = await xtts_manual_load_model()
    #Create an end timer for calculating load times
    generate_end_time = time.time()
    #Calculate start time minus end time
    generate_elapsed_time = generate_end_time - generate_start_time
    #Print out the result of the load time
    print(f"[CoquiTTS Model] \033[94mModel Loaded in \033[93m{generate_elapsed_time:.2f} seconds.\033[0m")
    #Set "model_loaded" to true
    params["model_loaded"] = True
    #Set the output path for wav files
    Path(f'{params["output_folder_wav"]}outputs/').mkdir(parents=True, exist_ok=True)

#MODEL LOADER For "API TTS"
async def api_load_model():
    global model
    model = TTS(params["model_name"]).to(device)
    return model

#MODEL LOADER For "API Local"
async def api_manual_load_model():
    global model
    #check to see if a custom path has been set in modeldownload.json and use that path to load the model if so
    if str(modeldownload_base_path) == "models":
        model = TTS(model_path=this_dir / 'models' / modeldownload_model_path,config_path=this_dir / 'models' / modeldownload_model_path / 'config.json').to(device)
    else:
        print("[CoquiTTS Model] \033[94mInfo\033[0m Loading your custom model set in \033[93mmodeldownload.json\033[0m:", modeldownload_base_path / modeldownload_model_path)
        model = TTS(model_path=modeldownload_base_path / modeldownload_model_path,config_path=modeldownload_base_path / modeldownload_model_path / 'config.json').to(device)
    return model

#MODEL LOADER For "XTTSv2 Local"
async def xtts_manual_load_model():
    global model
    config = XttsConfig()
    #check to see if a custom path has been set in modeldownload.json and use that path to load the model if so
    if str(modeldownload_base_path) == "models":
        config_path = this_dir / 'models' / modeldownload_model_path / 'config.json'
        checkpoint_dir = this_dir / 'models' / modeldownload_model_path
    else:
        print("[CoquiTTS Model] \033[94mInfo\033[0m Loading your custom model set in \033[93mmodeldownload.json\033[0m:", modeldownload_base_path / modeldownload_model_path)
        config_path = modeldownload_base_path / modeldownload_model_path / 'config.json'
        checkpoint_dir = modeldownload_base_path / modeldownload_model_path
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(checkpoint_dir), use_deepspeed=params['deepspeed_activate'])
    model.cuda()
    model.to(device)
    return model

#MODEL UNLOADER
async def unload_model(model):
    print("[CoquiTTS Model] \033[94mUnloading model \033[0m")
    del model
    torch.cuda.empty_cache()
    params["model_loaded"] = False
    return None

#MODEL - Swap model based on Gradio selection API TTS, API Local, XTTSv2 Local
async def handle_tts_method_change(tts_method):
    global model
    #Update the params dictionary based on the selected radio button
    print("[CoquiTTS Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
    #Set other parameters to False
    if tts_method == "API TTS":
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_tts"] = True
        params["deepspeed_activate"] = False
    elif tts_method == "API Local":
        params["tts_method_api_tts"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_local"] = True
        params["deepspeed_activate"] = False
    elif tts_method == "XTTSv2 Local":
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True

    #Unload the current model
    model = await unload_model(model)

    #Load the correct model based on the updated params
    await setup()

#MODEL WEBSERVER- API Swap Between Models
@app.route("/api/reload", methods=["POST"])
async def reload(request: Request):
    tts_method = request.query_params.get("tts_method")
    if tts_method not in ["API TTS", "API Local", "XTTSv2 Local"]:
        return {"status": "error", "message": "Invalid TTS method specified"}     
    await handle_tts_method_change(tts_method)
    return Response(
        content=json.dumps({"status": "model-success"}),
        media_type="application/json"
    )

##################
#### LOW VRAM ####
##################
#LOW VRAM - MODEL MOVER VRAM(cuda)<>System RAM(cpu) for Low VRAM setting
async def switch_device():
    global model, device
    if device == "cuda":
        device = "cpu"
        model.to(device) 
        torch.cuda.empty_cache()
    else:    
        device == "cpu"
        device = "cuda"
        model.to(device)

@app.post("/api/lowvramsetting")
async def set_low_vram(request: Request, new_low_vram_value: bool):
    global device
    try:
        if new_low_vram_value is None:
            raise ValueError("Missing 'low_vram' parameter")
        if params["low_vram"] == new_low_vram_value:
            return Response(
                content=json.dumps({"status": "success", "message": f"[CoquiTTS Model] LowVRAM is already {'enabled' if new_low_vram_value else 'disabled'}."}))
        params["low_vram"] = new_low_vram_value
        if params["low_vram"]:
            await unload_model(model)
            device = "cpu"
            print("[CoquiTTS Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
            print("[CoquiTTS Model] \033[94mLowVRAM Enabled.\033[0m Model will move between \033[93mVRAM(cuda) <> System RAM(cpu)\033[0m")
            await setup()
        else:
            await unload_model(model)
            device = "cuda"
            print("[CoquiTTS Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
            print("[CoquiTTS Model] \033[94mLowVRAM Disabled.\033[0m Model will stay in \033[93mVRAM(cuda)\033[0m")
            await setup()
        return Response(content=json.dumps({"status": "lowvram-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

###################
#### DeepSpeed ####
###################
#DEEPSPEED - Reload the model when DeepSpeed checkbox is enabled/disabled
async def handle_deepspeed_change(value):
    global model
    if value:
        # DeepSpeed enabled
        print("[CoquiTTS Model] \033[93mDeepSpeed Activating\033[0m")
        print("[CoquiTTS Model] \033[94mChanging model \033[92m(DeepSpeed can take 30 seconds to activate)\033[0m")
        print("[CoquiTTS Model] \033[91mInformation\033[0m If you have not set CUDA_HOME path, DeepSpeed may fail to load/activate")
        print("[CoquiTTS Model] \033[91mInformation\033[0m DeepSpeed needs to find nvcc from the CUDA Toolkit. Please check your CUDA_HOME path is")
        print("[CoquiTTS Model] \033[91mInformation\033[0m pointing to the correct location and use 'set CUDA_HOME=putyoutpathhere' (Windows) or")
        print("[CoquiTTS Model] \033[91mInformation\033[0m 'export CUDA_HOME=putyoutpathhere' (Linux) within your Python Environment")
        model = await unload_model(model)
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True
        params["deepspeed_activate"] = True
        await setup()
    else:
        # DeepSpeed disabled
        print("[CoquiTTS Model] \033[93mDeepSpeed De-Activating\033[0m")
        print("[CoquiTTS Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
        params["deepspeed_activate"] = False 
        model = await unload_model(model)
        await setup()

    return value # Return new checkbox value

#DEEPSPEED WEBSERVER- API Enable/Disable DeepSpeed
@app.post("/api/deepspeed")
async def deepspeed(request: Request, new_deepspeed_value: bool):
    try:
        if new_deepspeed_value is None:
            raise ValueError("Missing 'deepspeed' parameter")
        if params["deepspeed_activate"] == new_deepspeed_value:
            return Response(
                content=json.dumps({"status": "success", "message": f"DeepSpeed is already {'enabled' if new_deepspeed_value else 'disabled'}."}))
        params["deepspeed_activate"] = new_deepspeed_value
        await handle_deepspeed_change(params["deepspeed_activate"])
        return Response(content=json.dumps({"status": "deepspeed-success"}))
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}))

########################
#### TTS GENERATION ####
########################
#TTS VOICE GENERATION METHODS (called from voice_preview and output_modifer)
async def generate_audio(text, voice, language, output_file):
    global model
    if params["low_vram"] and device == "cpu":
        await switch_device()
    generate_start_time = time.time()  # Record the start time of generating TTS
    #XTTSv2 LOCAL Method
    if params["tts_method_xtts_local"]:     
        print("[CoquiTTS TTSGen] {}".format(text))
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[f"{this_dir}/voices/{voice}"])
        out = model.inference(
        text, 
        language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
        enable_text_splitting=True
        )
        torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    #API TTS and API LOCAL Methods 
    elif params["tts_method_api_tts"] or params["tts_method_api_local"]:
        #Set the correct output path (different from the if statement)
        print("[CoquiTTS TTSGen] Using API TTS/Local Method")
        model.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{voice}"],
            language=language,
        )
    #Print Generation time and settings
    generate_end_time = time.time()  # Record the end time to generate TTS
    generate_elapsed_time = generate_end_time - generate_start_time
    print(f"[CoquiTTS TTSGen] \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{params['low_vram']} \033[94mDeepSpeed: \033[33m{params['deepspeed_activate']}\033[0m")
    #Move model back to cpu system ram if needed.
    if params["low_vram"] and device == "cuda":
        await switch_device()
    return

#TTS VOICE GENERATION METHODS - generate TTS API
@app.route("/api/generate", methods=["POST"])
async def generate(request: Request):
    try:
        # Get parameters from JSON body
        data = await request.json()
        text = data["text"]
        voice = data["voice"]
        language = data["language"]
        output_file = data["output_file"] 
        # Generation logic
        await generate_audio(text, voice, language, output_file)       
        return JSONResponse(content={"status": "generate-success", "data": {"audio_path": output_file}})    
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})

#############################
#### JSON CONFIG UPDATER ####
#############################

#Create an instance of Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory=this_dir / "templates")

#Create a dependency to get the current JSON data
def get_json_data():
    with open(this_dir / 'config.json', "r") as json_file:
        data = json.load(json_file)
    return data

#Define an endpoint function
@app.get("/settings")
async def get_settings(request: Request):
    # Render the template with the current JSON data
    return templates.TemplateResponse("generate_form.html", {"request": request, "data": get_json_data(), "modeldownload_model_path": modeldownload_model_path})

@app.post("/update-settings")
async def update_settings(
    request: Request, 
    activate: bool = Form(...),
    autoplay: bool = Form(...),
    deepspeed_activate: bool = Form(...),
    delete_output_wavs: str = Form(...),
    ip_address: str = Form(...),
    language: str = Form(...),
    low_vram: bool = Form(...),
    model_loaded: bool = Form(...),
    model_name: str = Form(...),
    output_folder_wav: str = Form(...),
    port_number: str = Form(...),
    remove_trailing_dots: bool = Form(...),
    show_text: bool = Form(...),
    tts_method: str = Form(...),
    voice: str = Form(...),
    data: dict = Depends(get_json_data),
):
    # Update the settings based on the form values
    data["activate"] = activate
    data["autoplay"] = autoplay
    data["deepspeed_activate"] = deepspeed_activate
    data["delete_output_wavs"] = delete_output_wavs
    data["ip_address"] = ip_address
    data["language"] = language
    data["low_vram"] = low_vram
    data["model_loaded"] = model_loaded
    data["model_name"] = model_name
    data["output_folder_wav"] = output_folder_wav
    data["port_number"] = port_number
    data["remove_trailing_dots"] = remove_trailing_dots
    data["show_text"] = show_text
    data["tts_method_api_local"] = tts_method == "api_local"
    data["tts_method_api_tts"] = tts_method == "api_tts"
    data["tts_method_xtts_local"] = tts_method == "xtts_local"
    data["voice"] = voice

    #Save the updated settings back to the JSON file
    with open(this_dir / 'config.json', "w") as json_file:
        json.dump(data, json_file)

    #Redirect to the settings page to display the updated settings
    return RedirectResponse(url="/settings", status_code=303)

#############################################################
#### DOCUMENTATION - README ETC - PRESENTED AS A WEBPAGE ####
#############################################################

simple_webpage = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoquiTTS for Text-generation-WebUI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px; /* Adjusted max-width for better readability */
            margin: 40px auto;
            padding: 20px;
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        h1, h2, h3, h4, h5 {
            color: #333;
        }

        h1, h2, h3 {
        text-decoration: underline;
        }

        p, span {
            color: #555;
            font-size: 16px; /* Increased font size for better readability */
            line-height: 1.5; /* Adjusted line-height for better spacing */
        }

        code {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 2px 2px;
            font-size: 14px; /* Adjusted font size for better visibility */
            margin-bottom: 0px;
        }

        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
            font-size: 14px; /* Adjusted font size for better visibility */
            line-height: 1.5; /* Adjusted line-height for better spacing */
        }

        ul {
            color: #555;
            list-style-type: square; /* Set the bullet style */
            line-height: 1.5; /* Adjusted line-height for better spacing */
            margin-left: 2px; /* Adjust the left margin to create an indent */
        } 

        li {
            font-size: 16px; /* Set the font size for list items */
            margin-bottom: 8px; /* Add some space between list items */
        }

        .key {
        color: black; /* Color for keys */
        font-size: 14px; /* Increased font size for better readability */
        }

        .value {
        font-size: 14px; /* Increased font size for better readability */
        color: blue; /* Color for values */
        }


        a {
            color: #0077cc;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        strong {
            font-weight: bold;
        }
        
        .option-a {
            color: #33cc33;
            font-weight: bold;
        }

        .option-b {
            color: red;
            font-weight: bold;
        }

    </style>
</head>

<body>
    <h1 id="toc">CoquiTTS for Text-generation-WebUI</h1>
    <p>If you are looking for your <b>Text-generation-WebUI</b> webpage, try visiting <a href="http://127.0.0.1:7860" target="_blank">http://127.0.0.1:7860</a>.</p>
    <p><b>Text-generation-WebUI</b> documentation and wiki <a href="https://github.com/oobabooga/text-generation-webui/wiki" target="_blank">https://github.com/oobabooga/text-generation-webui/wiki</a>.</p>

    <iframe src="http://127.0.0.1:7851/settings" width="100%" height="440" frameborder="0" style="margin: 0; padding: 0;"></iframe>
    
    <h3>Table of Contents</h3>
    <ul>
        <li><a href="#getting-started">Getting Started with CoquiTTS</a></li>
        <li><a href="#server-information">Server Information</a></li>
        <li><a href="#using-voice-samples">Using Voice Samples</a></li>
        <li><a href="#where-are-the-outputs-stored">Automatic output wav file deletion</a></li>
        <li><a href="#low-vram-option-overview">Low VRAM Overview</a></li>
        <li><a href="#deepspeed-simplified">DeepSpeed Simplified</a></li>
        <li><a href="#setup_deepspeed">Setup DeepSpeed</a></li>
        <li><a href="#other-features">Other Features of Coqui_TTS Extension for Text-generation-webui</a></li>
        <li><a href="#TTSmodels">TTS Models/Methods</a></li>
        <li><a href="#customTTSmodels">Custom TTS Models and Model path</a></li>
        <li><a href="#curl-commands">CURL Commands</a></li>
        <li><a href="#configuration-details">Configuration Details</a></li>
        <li><a href="#debugging-and-tts-generation-information">Debugging and TTS Generation Information</a></li>
        <li><a href="#references">Thanks & References</a></li>
    </ul>

    <h2 id="getting-started">Getting Started</h2>
    <p>To generate TTS, you can use the provided interface or interact with the server using CURL commands. Below are some details and examples:</p>

    <h3 id="server-information">Server Information</h3>
    <ul>
        <li>Base URL: <code>http://127.0.0.1:7851</code></li>
        <li>Server Status: <code><a href="http://127.0.0.1:7851/ready">http://127.0.0.1:7851/ready</a></code></li>
    </ul>
    <p><a href="#toc">Back to top of page</a></p>

    <h3 id="using-voice-samples"><strong>Using Voice Samples</strong></h3>
    <h4 id="where-are-the-sample-voices-stored">Where are the sample voices stored?</h4>
    <p>Voice samples are stored in <b>/extensions/coqui_tts/voices/</b> and should be named using the following format <b>name.wav</b></p>
    <h4 id="where-are-the-outputs-stored">Where are the outputs stored & Automatic output wav file deletion.</h4>
    <p>Voice samples are stored in <b>/extensions/coqui_tts/outputs/</b></p>
    <p>You can configure automatic maintenence deletion of old wav files by setting <b>"Del WAV's older than"</b> in the settings above. When <b>"Disabled"</b> your output wav files will be left untouched. When set to a setting <b>"1 Day"</b> or greater, your output wav files older than that time period will be automatically deleted on startup.</p>
    <h4>Where are the models stored?</h4>
    <p>This extension will download the 2.0.2 model to <b>/extensions/coqui_tts/models/</b></p>
    <p>This TTS engine will also download the latest available model and store it wherever it normally stores it for your OS (Windows/Linux/Mac).</p>
    <h4>How do I create a new voice sample?</h4>
    <p>To create a new voice sample you need to make a <b>wav</b> file that is <b>22050Hz</b>, <b>Mono</b>, <b>16 bit</b> and between <b>6 to 12 seconds long</b>, though 7 to 9 seconds is usually good.</p>
    <p>You want to find a nice clear selection of audio, so lets say you wanted to clone your favourite celebrity. You may go looking for an interview where they are talking. Pay close attention to the audio you are listening to and trying to sample, are there noises in the backgroud, hiss on the soundtrack, a low humm, some quiet music playing or something? The better quality the audio the better the result. Dont forget, the AI that processes the sounds can hear everything, all those liitle noises, and it will use them in the voice its trying to recreate. </p>
    <p>Try make your clip one of nice flowing speech, like the included example files. No big pauses, gaps or other sounds. Preferably one that the person you are trying to copy will show a little vocal range and emotion in their voice. Also, try to avoid a clip starting or ending with breathy sounds (breathing in/out etc).</p>
    <h4>Generating the sample!</h4>
    <p>So youve downloaded your favoutie celebrity interview off Youtube or wherever. From here you need to chop it down to 6 to 12 seconds in length and resample it. If you need to clean it up, do audio processing, volume level changes etc, do it before the steps I am about to describe.</p>
    <p>Using the latest version of <b>Audacity</b>, select your clip and <b>Tracks > Resample to 22050Hz</b>, then <b>Tracks > Mix > Stereo to Mono</b>. and then <b>File > Export Audio</b>, saving it as a WAV of 22050Hz.</p>
    <p>Save your generated wav file in <b>/extensions/coqui_tts/voices/</b></p>
    <p><b>Note:</b> Using AI generated audio clips <b>may</b> introduce unwanted sounds as its already a copy/simulation of a voice.</p>
    <h4>Why doesnt it sound like XXX Person?</h4>
    <p>The reasons can be that you:</p>
    <ul>
    <li>Didn't downsample it as above.</li>
    <li>Have a bad quality voice sample.</li>
    <li>Try using the 3x different generation methods: <b>API TTS</b>, <b>API Local</b>, and <b>XTTSv2 Local</b> within the web interface, as they generate output in different ways and sound different.</li>
    </ul>
    <p><a href="#toc">Back to top of page</a></p>
    
    <h3 id="low-vram-option-overview"><strong>Low VRAM Overview:</strong></h3>
    <p>The Low VRAM option is a crucial feature designed to enhance performance under constrained Video Random Access Memory (VRAM) conditions, as the TTS models require 2GB-3GB of VRAM to run effectively. This feature strategically manages the relocation of the Text-to-Speech (TTS) model between your system's Random Access Memory (RAM) and VRAM, moving it between the two on the fly.</p>

    <h4>How It Works:</h4>
    <p>The Low VRAM option intelligently orchestrates the relocation of the entire TTS model. When the TTS engine requires VRAM for processing, the entire model seamlessly moves into VRAM, causing your LLM to unload/displace some layers, ensuring optimal performance of the TTS engine.</p>
    <p>The TTS model is fully loaded into VRAM, facilitating uninterrupted and efficient TTS generation, creating contiguous space for the TTS model and significantly accelerating TTS processing, especially for long paragraphs. Post-TTS processing, the model promptly moves back to RAM, freeing up VRAM space for your Language Model (LLM) to load back in the missing layers. This adds about 1-2 seconds to both text generation by the LLM and the TTS engine.</p>
    <p>By transferring the entire model between RAM and VRAM, the Low VRAM option avoids fragmentation, ensuring the TTS model remains cohesive and accessible.</p>
    <p>This creates a TTS generation performance Boost for Low VRAM Users and is particularly beneficial for users with less than 2GB of free VRAM after loading their LLM, delivering a substantial 5-10x improvement in TTS generation speed.</p>
    <p><a href="#toc">Back to top of page</a></p>

    <h3 id="deepspeed-simplified"><strong>DeepSpeed Simplified:</strong></h3>

    <h4>What's DeepSpeed?</h4>
    <p>DeepSpeed, developed by Microsoft, is like a speed boost for Text-to-Speech (TTS) tasks. It's all about making TTS happen faster and more efficiently.</p>

    <h4>How Does It Speed Things Up?</h5>
    <ul>
        <li><strong>Model Parallelism:</strong> Spreads the work across multiple GPUs, making TTS models handle tasks more efficiently.</li>
        <li><strong>Memory Magic:</strong> Optimizes how memory is used, reducing the memory needed for large TTS models.</li>
        <li><strong>Efficient Everything:</strong> DeepSpeed streamlines both training and generating speech from text, making the whole process quicker.</li>
    </ul>

    <h4>Why Use DeepSpeed for TTS?</h4>
    <ul>
        <li><strong>2x-3x Speed Boost:</strong> Generates speech much faster than usual.</li>
        <li><strong>Handles More Load:</strong> Scales up to handle larger workloads with improved performance.</li>
        <li><strong>Smart Resource Use:</strong> Uses your computer's resources smartly, getting the most out of your hardware.</li>
    </ul>
    
    <p><strong>Note:</strong> DeepSpeed only works with the XTTSv2 Local model.</p>
    <p><strong>Note:</strong> Requires Nvidia Cuda Toolkit installation and correct CUDA_HOME path configuration.</p>

    <h4>How to Use It:</h4>
    <p>In CoquiTTS, the DeepSpeed checkbox will only be available if DeepSpeed is detected on your system. Check the checkbox, wait 10 to 30 seconds and off you go!</p>
    <p><a href="#toc">Back to top of page</a></p>

    <h3 id="setup_deepspeed">Setup DeepSpeed</h3>
      <p><strong>Note:</strong> DeepSpeed WILL throw up a load of errors if you have not installed it fully, correctly, or installed the Nvidia Cuda Toolkit and correctly set the CUDA_HOME environment variable or if load text-generation-webui via its start-up scripts.</p>

    <h4>Linux - DeepSpeed - Installation Instructions</h4>
    <ol>
        <li>Download and install the <a href="https://developer.nvidia.com/cuda-toolkit-archive">Nvidia Cuda Toolkit for Linux</a>.</li>
        <li>Load up a terminal console.</li>
        <li>Install libaio-dev (however your Linux version installs things) e.g. <code>sudo apt install libaio-dev</code></li>
        <li>Move into your text-generation-webui folder e.g. <code>cd text-generation-webui</code></li>
        <li>Start the text-generation-webui Python environment e.g. <code>./start_linux.sh</code></li>
        <li>Text-generation-webui overwrites the CUDA_HOME variable on each startup, so you will need to either force this to be changed within the environment OR change it each time you <code>./start_linux.sh</code></li>
        <li>You can set the CUDA_HOME environment with <code>export CUDA_HOME=/usr/local/cuda</code> or <code>export CUDA_HOME=/etc/alternatives/cuda</code>. On some systems only one of those two commands may be the correct command, so you may need to try one, see if it works, if not try the other. IF you do not set it, expect a big messy output to the log when you try to activate DeepSpeed.</li>
        <li>Now install deepspeed with <code>pip install deepspeed</code></li>
        <p><strong>Note:</strong> You can run <code>ds_report</code> when you have installed DeepSpeed on your system to see if it is working correctly</p>
        <li>You can now start text-generation-webui <code>python server.py</code> ensuring to activate your extensions.</li>
    </ol>
    <p><a href="#toc">Back to top of page</a></p>

    <h4>Windows - DeepSpeed Version 8.3 & CUDA 11.8 or CUDA 12.1 - Installation Instructions</h4>
    <p>Currently DeepSpeed 8.3 is the latest build that works on Windows and it has to run inside a Python 3.9.18 environment. As such, either you need to have your own, already set up Python 3.9.18 environment and have installed the correct text-generation-webui requirements file into it. Or you can create a new Python environment. DeepSpeed version 9.x, 10.x, 11.x, and 12.x will all fail to compile <strong>(on Windows)</strong> for a multitude of reasons. As such, DeepSpeed v8.3 is the most recent build for Windows.</p>

    <h4>DeepSpeed on Windows - <span class="option-a">Option A</span></h4>
    <p><span class="option-a">Option A</span> is to fully go through the process of compiling DeepSpeed yourself. This takes a longer time to perform, but you can be sure it's set up for your system correctly.</p>

    <h4>DeepSpeed on Windows - <span class="option-b">Option B</span></h4>
    <p><span class="option-b">Option B</span> is to download someone else's precompiled wheels. If you can find someone's precompiled wheel file, you only need to follow <span class="option-b">Option B</span>.</p>

    <ol>
    <li>(<span class="option-a">OPTION A</span>) Download the 8.3 release of <a href="https://github.com/microsoft/DeepSpeed/releases/tag/v0.8.3">DeepSpeed</a> and extract it to a folder.</li>
    <li>(<span class="option-a">OPTION A</span>) Install Visual C++ build tools, such as <a href="https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#vs2019-download">VS2019 C++ x64/x86</a> build tools.</li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Download and install the <a href="https://developer.nvidia.com/cuda-toolkit-archive">Nvidia Cuda Toolkit 11.8 OR 12.1</a> <b>(11.8 or 12.1)</b></li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Edit your Windows environment variables to ensure that CUDA_HOME and CUDA_PATH are set to your Nvidia Cuda Toolkit path. (The folder <strong>above</strong> the bin folder that <strong>nvcc.exe</strong> is installed in). Examples are:<br>
        <br>CUDA 11.8<br><br>
        <code>CUDA_HOME=C:&#92;Program Files&#92;NVIDIA GPU Computing Toolkit&#92;CUDA&#92;v11.8</code><br>
        <code>CUDA_PATH=C:&#92;Program Files&#92;NVIDIA GPU Computing Toolkit&#92;CUDA&#92;v11.8</code><br>
        <br>
        CUDA 12.1<br><br>
        <code>CUDA_HOME=C:&#92;Program Files&#92;NVIDIA GPU Computing Toolkit&#92;CUDA&#92;v12.1</code><br>
        <code>CUDA_PATH=C:&#92;Program Files&#92;NVIDIA GPU Computing Toolkit&#92;CUDA&#92;v12.1</code><br><br>

        It is possible to permanently set these or use the <code>set</code> command at the command prompt when inside your python environment e.g. <b>set CUDA_HOME=C:&#92;Program Files&#92;NVIDIA GPU Computing Toolkit&#92;CUDA&#92;v12.1'</b>. You can then just type <code>set</code> again, and it will show you the list of all currently set environment variables so that you can confirm you have them set correctly.
    </li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) <strong>Currently Python versions 3.9.x are supported</strong>. If you do not have a Python environment already created, you can install <a href="https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html">Miniconda</a>, then at a command prompt, create and activate your environment with:
        <br><br><code>conda create -n pythonenv python=3.9.18</code>
        <br><br><code>activate pythonenv</code>
    </li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Launch the Command Prompt <strong>cmd</strong> with <strong>Administrator privilege</strong> as it requires admin to allow creating symlink folders.</li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Install <a href="https://pytorch.org/get-started/previous-versions/#v210">PyTorch</a>, 2.1.0 with CUDA 11.8 or CUDA 12.1 into your Python 3.9.x environment e.g.<br><br>
        CUDA 11.8<br>
        <br><code>activate pythonenv</code> (activate your python environment)<br>
        <br><code>conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia</code><br>
        <br>CUDA 12.1<br>
        <br><code>activate pythonenv</code> (activate your python environment)<br>
        <br><code>conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia</code>
    </li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) In your python environment double-check that your <b>CUDA_HOME</b> and <b>CUDA_PATH</b> are still pointing to the correct location.
        <br>Type <code>set</code> to list and check the Windows environment variables. Refer to step 4 if not.
    </li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Navigate to your deepspeed folder in the Command Prompt:<code>cd c:\deepspeed</code> (wherever you extracted it to)
    </li>
    <li>(<span class="option-a">OPTIONAL A</span>) You can try using the <strong>build_win.bat</strong> however you may need to set options to not install certain features e.g. <strong>set DS_BUILD_AIO=0</strong> <a href="https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops">https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops</a>. Type the following into the command prompt:
        <br><br><code>set DS_BUILD_AIO=0</code>
        <br><br><code>set DS_BUILD_SPARSE_ATTN=0</code>
        <br><br><code>set DS_BUILD_EVOFORMER_ATTN=0</code>
        <br><br><code>build_win.bat</code> (This will start building your wheel file and may take a while)
    </li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Now <strong>cd dist</strong> to go into your <strong>dist</strong> folder or wherever you have put your downloaded wheel file and you can now <strong>pip install deepspeed-_YOURFILENAME_.whl</strong> (or whatever your WHL file is called).
        <br><strong><br>Note:</strong> You can run <code>ds_report</code> when you have installed DeepSpeed on your system to see if it is working correctly
    </li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Move into your text-generation-webui folder e.g. <code>cd text-generation-webui</code></li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Don't forget to have <code>pip install -r requirements.txt</code> or whichever requirements file you want to use for your system, within the text-generation-webui folder.</li>
    <li>(<span class="option-a">OPTION A</span> <span class="option-b">and B</span>) Start the text-generation=web-ui with Python e.g. <code>python server.py</code> ensuring to activate your extensions.
        <br><strong><br>Note:</strong> you can't use <strong>start_windows.bat</strong> or <strong>cmd_windows.bat</strong> because it will use a different environment, incorrect version of Python and it will overwrite the CUDA_HOME path.
    </li>
    </ol>

    <p><a href="#toc">Back to top of page</a></p>

    <h2 id="other-features"><strong>Other Features of Coqui_TTS Extension for Text-generation-webui</strong></h2>

    <h3>Start-up Checks</h3>
    <p>Ensures a minimum TTS version (0.21.3) is installed and provides an error/instructions if not.</p>
    <p>Performs a basic environment check to ensure everything should work e.g. is the model already downloaded, are the configuration files set correctly etc.</p>
    <p>Downloads the Xtts model (version 2.0.2) to improve generation speech quality as the 2.0.3 model sounded terrible. The API TTS version uses the latest model (2.0.3 at the time of writing) so you have the best of both worlds.</p>
    <p><a href="#toc">Back to top of page</a></p>

    <h3 id="TTSmodels">TTS Models/Methods</h3>
    <p>It's worth noting that all models and methods can and do sound different from one another. Many people complained about the quality of audio produced by the 2.0.<b>3</b> model, so this extension will download the 2.0.<b>2</b> model to your models folder and give you the choice to use 2.0.<b>2</b> <b>(API Local and XTTSv2 Local)</b> or use the most current model 2.0.<b>3</b> <b>(API TTS)</b>. As/When a new model is released by Coqui it will be downloaded by the TTS service on startup and stored wherever the TTS service keeps new models on your operating system of choice.</p>
    <ul>
        <li><strong>API TTS:</strong> Uses the current TTS model available that's downloaded by the TTS API process (e.g. version 2.0.3 at the time of writing). This model is not stored in your "models" folder, but elsewhere on your system and managed by the TTS software.</li>
        <li><strong>API Local:</strong> Utilizes the <b>2.0.2</b> local model stored at <b>/coqui_tts/models/xttsv2_2.0.2</b>.</li>
        <li><strong>XTTSv2 Local:</strong> Employs the <b>2.0.2</b> local model <b>/coqui_tts/models/xttsv2_2.0.2</b> and utilizes a distinct TTS generation method. <b>Supports DeepSpeed acceleration</b>.</li>
    </ul>
    <p><a href="#toc">Back to top of page</a></p>

    <h3 id="customTTSmodels">Custom TTS Models and Model path</h3>

    <p>Its possible to set a custom model for the <strong>API Local</strong> and<strong> XTTSv2</strong> Local methods, or ideed point it at the same model that <strong>API TTS</strong> uses (wherever it is stored on your OS of choice). Many people did not like the sound quality of the Coqui <strong>2.0.<span style="color:#e74c3c">3</span></strong> model, and as such the Coqui_tts extension downloads the <strong>2.0.<span style="color:#2980b9">2</span></strong> model seperately to the <strong>2.0.<span style="color:#e74c3c">3</span></strong> model that TTS service downloads and manages. Typically the <strong>2.0.<span style="color:#2980b9">2</span></strong> model is stored in your <strong>/extensions/coqui_tts/models</strong> folder and it is always downloaded on first start-up of the&nbsp;Coqui_tts extension. However, you may either want to use a custom model version of your choosing, or point it to a different path on your system, or even point it so that&nbsp;API Local and XTTSv2 <strong>both</strong> use the same model that API TTS is using.</p>
    <p>If you do choose to change the location there are a couple of things to note.&nbsp;</p>
    <p>- The folder you place the model in, <strong><span style="color:#e74c3c">cannot</span></strong> be called &quot;<strong>models</strong>&quot;. This name is reserved solely for the system to identify you are or are not using a custom model.</p>
    <p>- On each startup, the&nbsp;Coqui_tts extension will check the custom location and if it does not exist, it will create it and download the files it needs. It will also re download any missing files in that location that are needed for the model to function.</p>
    <p>- There will be extra output at the console to inform you that you are using a custom model and each time you load up the Coqui_tts extension or switch between models.</p>
    <p>To change the model path, there are at minimum 2x settings you need to alter in the <strong>modeldownload.json</strong> file, <strong>base_path</strong> and <strong>model_path</strong>. These two settings are further detailed in the section below called <strong>Explanation of the modeldownload.json file</strong></p>
    <p>You would edit the settings in the <strong>modeldownload.json</strong> file as follows (make a backup of your current file in case):&nbsp;</p>
    <p><strong>Windows example:</strong>&nbsp;<span style="color:#2980b9">c:\mystuff\mydownloads\myTTSmodel\</span><em>{files in here}</em></p>

    <ul>
	    <li>&nbsp; &nbsp; &nbsp; &nbsp;<strong>base_path </strong>would be &quot;<span style="color:#2980b9">c:&#92;&#92;mystuff&#92;&#92;mydownloads&quot;</span></li>
	    <li>&nbsp; &nbsp; &nbsp; &nbsp;<strong>model_path</strong> would be:<span style="color:#2980b9">&nbsp;&quot;myTTSmodel&quot;</span></li>
    </ul>

    <p><strong>Note:</strong> On Windows systems, you have to specify a<strong> <span style="color:#e74c3c">double backslash &#92;&#92;</span></strong> for each folder level in the path (as above).</p>
    <p><strong>Linux example:</strong>&nbsp;<span style="color:#2980b9">/home/myaccount/myTTSmodel/</span><em>{files in here}</em></p>

    <ul>
	    <li>&nbsp; &nbsp; &nbsp; &nbsp;<strong>base_path </strong>would be &quot;<span style="color:#2980b9">/home/myaccount&quot;</span></li>
	    <li>&nbsp; &nbsp; &nbsp; &nbsp;<strong>model_path</strong> would be:<span style="color:#2980b9">&nbsp;&quot;TTSmodel&quot;</span></li>
    </ul>

    <p>Once you restart the&nbsp;Coqui_tts extension, it will check this path for the files and output any details at the console.</p>
    <p>When you are happy it's' working correctly, you are welcome to go delete the models folder stored at&nbsp;<strong>/extensions/coqui_tts/models.</strong></p>
    <p>If you wish to change the files that the modeldownloader is pulling at startup, you can futher edit the <strong>modeldownload.json</strong> and change the http addresses within this files&nbsp;<strong>files_to_download</strong> section &nbsp;e.g.</p>
    <p>&nbsp; &nbsp; &quot;files_to_download&quot;: {<br />
    &nbsp; &nbsp; &nbsp; &nbsp; &quot;LICENSE.txt&quot;: &quot;https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true&quot;,<br />
    &nbsp; &nbsp; &nbsp; &nbsp; &quot;README.md&quot;: &quot;https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true&quot;,<br />
    &nbsp; &nbsp; &nbsp; &nbsp; &quot;config.json&quot;: &quot;https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true&quot;,<br />
    &nbsp; &nbsp; &nbsp; &nbsp; &quot;model.pth&quot;: &quot;https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true&quot;,<br />
    &nbsp; &nbsp; &nbsp; &nbsp; &quot;vocab.json&quot;: &quot;https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true&quot;<br />
    &nbsp; &nbsp; }</p>
    <p><a href="#toc">Back to top of page</a></p>

    <h3 id="configuration-details"><strong>Configuration Details</strong></h3>
    <p>Explanation of the <b>config.json</b> file:</p>

    <code><span class="key">"activate:"</span> <span class="value">true</span>,</code><span class="key"> Used within the code, do not change.</span><br>
    <code><span class="key">"autoplay:"</span> <span class="value">true</span>,</code><span class="key"> Controls whether the TTS audio plays automatically within Text-generation-webUI.</span><br>
    <code><span class="key">"deepspeed_activate:"</span> <span class="value">false</span>,</code><span class="key"> Controls whether the DeepSpeed option is activated or disabled in the Gradio interface.</span><br>
    <code><span class="key">"delete_output_wavs:"</span> <span class="value">""Disabled""</span>,</code><span class="key"> If set this will delete your old output wav files, older than the date set, when the system starts up.</span><br>
    <code><span class="key">"ip_address:"</span> <span class="value">"127.0.0.1"</span>,</code><span class="key"> Specifies the default IP address for the web server.</span><br>
    <code><span class="key">"language:"</span> <span class="value">"English"</span>,</code><span class="key"> Specifies the default language to use for TTS.</span><br>
    <code><span class="key">"low_vram:"</span> <span class="value">false</span>,</code><span class="key"> Controls whether the Low VRAM option is enabled or disabled.</span><br>
    <code><span class="key">"model_loaded:"</span> <span class="value">true</span>,</code><span class="key"> Used within the code, do not change.</span><br>
    <code><span class="key">"model_name:"</span> <span class="value">"tts_models/multilingual/multi-dataset/xtts_v2"</span>,</code><span class="key"> Specifies the model that the "API TTS" method will use for TTS generation.</span><br>
    <code><span class="key">"port_number:"</span> <span class="value">"7851"</span>,</code><span class="key"> Specifies the default port number for the web server.</span><br>
    <code><span class="key">"output_folder_wav:"</span> <span class="value">"extensions/coqui_tts/outputs/"</span>,</code><span class="key"> Sets the output folder to send files to on generating TTS.</span><br>
    <code><span class="key">"remove_trailing_dots:"</span> <span class="value">false</span>,</code><span class="key"> Controls whether trailing dots are removed from text segments before generation.</span><br>
    <code><span class="key">"show_text:"</span> <span class="value">true</span>,</code><span class="key"> Controls whether message text is shown under the audio player.</span><br>
    <code><span class="key">"tts_method_api_local:"</span> <span class="value">false</span>,</code><span class="key"> Controls whether the "API Local" model/method is turned on or off.</span><br>
    <code><span class="key">"tts_method_api_tts:"</span> <span class="value">false</span>,</code><span class="key"> Controls whether the "API TTS" model/method is turned on or off.</span><br>
    <code><span class="key">"tts_method_xtts_local:"</span> <span class="value">true</span>,</code><span class="key"> Controls whether the "XTTSv2 Local" model/method is turned on or off.</span><br>
    <code><span class="key">"voice:"</span> <span class="value">"female_01.wav"</span></code><span class="key"> Specifies the default voice to use for TTS.</span><br>
    <p><a href="#toc">Back to top of page</a></p>

    <p>Explanation of the <b>modeldownload.json</b> file:</p>

    <code><span class="key">"base_path:"</span> <span class="value">"models"</span>,</code><span class="key"> Specifies the base directory for the TTS models. If this is changed from the word "models" then the new custom path will be used e.g. <b>C:&bsol;&bsol;mymodels</b></span><br>
    <p><b>(Please note the use of a double backslash &bsol;&bsol; between folders on Windows systems. Linux systems will use "/mymodels" style)</b></p>
    <code><span class="key">"model_path:"</span> <span class="value">"xttsv2_2.0.2"</span>,</code><span class="key"> Specifies the directory path for the XTTSv2 model underneath the base_path.</span><br>
    <p><b>So if base_path is C:&bsol;&bsol;mymodels and model_path is xttsv2_2.0.2, then your path would become c:&bsol;&bsol;mymodels&bsol;&bsol;xttsv2_2.0.2</b></p>
    <code><span class="key">"files_to_download:"</span></code><span class="key"> Dictionary containing files to download with their respective URLs.</span><br>
    <code><span class="key">    "LICENSE.txt":</span> <span class="value">"https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/LICENSE.txt?download=true"</span>,</code><span class="key"> License file URL.</span><br>
    <code><span class="key">    "README.md":</span> <span class="value">"https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/README.md?download=true"</span>,</code><span class="key"> README file URL.</span><br>
    <code><span class="key">    "config.json":</span> <span class="value">"https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json?download=true"</span>,</code><span class="key"> Config file URL.</span><br>
    <code><span class="key">    "model.pth":</span> <span class="value">"https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth?download=true"</span>,</code><span class="key"> Model file URL.</span><br>
    <code><span class="key">    "vocab.json":</span> <span class="value">"https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json?download=true"</span>,</code><span class="key"> Vocabulary file URL.</span><br>
    <p>If you wish to set your own path</p>

    <h3 id="curl-commands"><strong>CURL Commands</strong></h3>
    <p><b>Generating TTS - Linux</b></p>
    <code><span class="key">curl -X POST -H "Content-Type: application/json" -d '{"text": "This is text to generate as TTS","voice": "female_01.wav", "language": "en", "output_file": "outputfile.wav"}' "http://127.0.0.1:7851/api/generate"</code></span>
    <p><b>Generating TTS - Windows</b></p>
    <code><span class="key">curl -X POST -H "Content-Type: application/json" -d="{&#92;&quot;text&#92;&quot;: &#92;&quot;This is text to generate as TTS&#92;&quot;,&#92;&quot;voice&#92;&quot;: &#92;&quot;female_01.wav&#92;&quot;, &#92;&quot;language&#92;&quot;: &#92;&quot;en&#92;&quot;, &#92;&quot;output_file&#92;&quot;: &#92;&quot;outputfile.wav&#92;&quot;}" 'http://127.0.0.1:7851/api/generate'</span></code>
    <p><b>Generating TTS - Notes</b></p>
    <p>Replace <code>This is text to generate as TTS</code> with whatever you want it to say. <code>female_01.wav</code> with the voice sample you want to use. <code>output_file.wav</code> with the file name you want it to create.</p>
    <p>JSON return <code>{"status":"generate-success","data":{"audio_path":"outputfile.wav"}}</code></p>
    
    <p><b>Switch Model</b></p>
    <code><span class="key">curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=API%20Local"</code></span><br>
    <code><span class="key">curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=API%20TTS"</code></span><br>
    <code><span class="key">curl -X POST "http://127.0.0.1:7851/api/reload?tts_method=XTTSv2%20Local"</code></span>
    <p>Switch between the 3 models respectively.</p>
    <p>JSON return <code>{"status": "model-success"}</code></p>

    <p><b>Switch DeepSpeed on/off</b></p>
    <code><span class="key">curl -X POST -H "Content-Type: application/json" "http://127.0.0.1:7851/api/deepspeed?new_deepspeed_value=True"</code></span>
    <p>Replace <code>True</code> with <code>False</code> to disable DeepSpeed mode.</p>
    <p>JSON return <code>{"status": "deepspeed-success"}</code></p>
    <p><b>Note:</b> Enabling DeepSpeed on systems that dont have DeepSpeed installed, may cause errors.</p>

    <p><b>Switch Low VRAM on/off</b></p>
    <code><span class="key">curl -X POST -H "Content-Type: application/json" "http://127.0.0.1:7851/api/lowvramsetting?new_low_vram_value=True"</code></span>
    <p>Replace <code>True</code> with <code>False</code> to disable Low VRAM mode.</p>
    <p>JSON return <code>{"status": "lowvram-success"}</code></p>
    <p><a href="#toc">Back to top of page</a></p>

    <h3 id="debugging-and-tts-generation-information"><strong>Debugging and TTS Generation Information:</strong></h3>
    <p>Command line outputs are more verbose to assist in understanding backend processes and debugging.</p>
    <p>Its possible during startup you can get a warning message such as <b>[CoquiTTS Startup] Warning TTS Subprocess has NOT started up yet, Will keep trying for 60 seconds maximum</b> This is normal behavior if the subprocess is taking a while to start, however, if there is an issue starting the subprocess, you may see multiples of this message and an it will time out after 60 seconds, resulting in the TTS engine not starting. Its likely that you are not in the correct python environment or one that has a TTS engine inside, if this happens, though the system will output a warning about that ahead of this message</p>
    <p>Typically the command line console will output any warning or error messages. If you need to reset your default configuation, the settings are all listed above in the configuration details.</p>
    <p><a href="#toc">Back to top of page</a></p>

    <h2 id="references"><strong>Thanks & References</strong></h2>
    <h3>Coqui TTS Engine</h3>
    <ul>
        <li><a href="https://coqui.ai/cpml.txt" target="_blank">Coqui License</a></li>
        <li><a href="https://github.com/coqui-ai/TTS" target="_blank">Coqui TTS GitHub Repository</a></li>
    </ul>
        <h3>Extension coded by</h3>
    <ul>
        <li><a href="https://github.com/erew123" target="_blank">Erew123 GitHub Profile</a></li>
    </ul>    
    <h3>Thanks to & Text-generation-WebUI</h3>
    <ul>
        <li><a href="https://github.com/oobabooga/text-generation-webui" target="_blank">Ooobabooga GitHub Repository</a> (Portions of orginal Coquii_TTS extension)</li>
    </ul>    
    <h3>Thanks to</h3>
    <ul>
        <li><a href="https://github.com/daswer123" target="_blank">daswer123 GitHub Profile</a> (Assistance with cuda to cpu moving)</li>
        <li><a href="https://github.com/kanttouchthis" target="_blank">kanttouchthis GitHub Profile</a> (Portions of orginal Coquii_TTS extension)</li>
        <li><a href="https://github.com/kanttouchthis" target="_blank">Wuzzooy GitHub Profile</a> (Trying out the code while in development)</li>
    </ul>
    <p><a href="#toc">Back to top of page</a></p>
</body>

</html>
"""

###################################################
#### Webserver Startup & Initial model Loading ####
###################################################
@app.get("/ready") 
async def ready():
    return Response("Ready endpoint")

@app.get("/")
async def read_root():
    return HTMLResponse(content=simple_webpage, status_code=200)

#Start Uvicorn Webserver
host_parameter = {params["ip_address"]}
port_parameter = str(params["port_number"])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=host_parameter, port=port_parameter, log_level="warning")
