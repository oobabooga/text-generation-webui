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
from fastapi.responses import JSONResponse, HTMLResponse
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

#######################################
#### LOAD PARAMS FROM CONFFIG.JSON ####
#######################################
def load_config(file_path):
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
    return config

config_file_path = (this_dir / 'config.json')
params = load_config(config_file_path)

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
        print(f"[CoquiTTS Model] \033[94mAPI Local Loading\033[0m {params['model_version']} \033[94minto\033[93m", device, "\033[0m")
        model = await api_manual_load_model()
    elif params["tts_method_xtts_local"]:
        print(f"[CoquiTTS Model] \033[94mXTTSv2 Local Loading\033[0m {params['model_version']} \033[94minto\033[93m", device, "\033[0m")
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
    Path(f"{this_dir}/outputs").mkdir(parents=True, exist_ok=True)

#MODEL LOADER For "API TTS"
async def api_load_model():
    global model
    model = TTS(params["model_name"]).to(device)
    return model

#MODEL LOADER For "API Local"
async def api_manual_load_model():
    global model
    model = TTS(model_path=this_dir / 'models' / params['model_version'],config_path=this_dir / 'models' / params['model_version'] / 'config.json').to(device)
    return model

#MODEL LOADER For "XTTSv2 Local"
async def xtts_manual_load_model():
    global model
    config = XttsConfig()
    config_path = this_dir / 'models' / params['model_version'] / 'config.json'
    checkpoint_dir = this_dir / 'models' / params['model_version']
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
    # Update the params dictionary based on the selected radio button
    print("[CoquiTTS Model] \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
    # Set other parameters to False
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

    # Unload the current model
    model = await unload_model(model)

    # Load the correct model based on the updated params
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


#############################################################
#### DOCUMENTATION - README ETC - PRESENTED AS A WEBPAGE ####
#############################################################

simple_webpage = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to CoquiTTS for Text-generation-WebUI</title>
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

        p, span {
            color: #555;
            font-size: 16px; /* Increased font size for better readability */
            line-height: 1.5; /* Adjusted line-height for better spacing */
        }

        code {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 2px 4px;
            font-size: 14px; /* Adjusted font size for better visibility */
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
            list-style: none;
            padding: 0;
        }

        li {
            margin-bottom: 10px;
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
    </style>
</head>

<body>
    <h1>CoquiTTS for Text-generation-WebUI</h1>
    <p>This is a Text-to-Speech (TTS) system powered by CoquiTTS (XTTSv2). You can use this system to convert text into spoken audio.</p>
    <p>If you've come here by mistake looking for the <b>Text-generation-WebUI</b>, try visiting <a href="http://127.0.0.1:7860" target="_blank">http://127.0.0.1:7860</a>.</p>

    <h2>Getting Started</h2>
    <p>To generate TTS, you can use the provided interface or interact with the server using CURL commands. Below are some details and examples:</p>

    <h3>Server Information</h3>
    <ul>
        <li>Base URL: <code>http://127.0.0.1:7851</code></li>
        <li>Server Status: <code><a href="http://127.0.0.1:7851/ready">http://127.0.0.1:7851/ready</a></code></li>
    </ul>

    <h3><strong>Using Voice Samples?</strong></h3>
    <h4>Where are the sample voices stored?</h4>
    <p>Voice samples are stored in <b>/extensions/coqui_tts/voices/</b> and should be named using the following format <b>name.wav</b></p>
    <h4>Where are the outputs stored?</h4>
    <p>Voice samples are stored in <b>/extensions/coqui_tts/outputs/</b></p>
    <p>FILES ARE NOT CURRENTLY AUTOMATICALLY DELETED</p>
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
    <p>NOTE: Using AI generated audio clips <b>may</b> introduce unwanted sounds as its already a copy/simulation of a voice.</p>
    <h4>Why doesnt it sound like XXX Person?</h4>
    <p>The reasons can be that you:</p>
    <p>    Didnt downsample it as above.</p>
    <p>    Have a bad quality voice sample.</p>
    <p>    Try using the 3x different generation methods, <b>API TTS</b>, <b>API Local</b> and <b>XTTSv2 Local</b> within the web interface, as they generate output in different ways and sound different.</p>
        
    <h3><strong>Low VRAM Option Overview:</strong></h3>
    <p>The Low VRAM option is a crucial feature designed to enhance performance under constrained Video Random Access Memory (VRAM) conditions, as the TTS models require 2GB-3GB of VRAM to run effectively. This feature strategically manages the relocation of the Text-to-Speech (TTS) model between your system's Random Access Memory (RAM) and VRAM, moving it between the two on the fly.</p>

    <h4>How It Works:</h4>
    <p>Dynamic Model Movement:</p>
    <p>The Low VRAM option intelligently orchestrates the relocation of the entire TTS model. When the TTS engine requires VRAM for processing, the entire model seamlessly moves into VRAM, causing your LLM to unload/displace some layers, ensuring optimal performance of the TTS engine.</p>
    <p>The TTS model is fully loaded into VRAM, facilitating uninterrupted and efficient TTS generation, creating contiguous space for the TTS model and significantly accelerating TTS processing, especially for long paragraphs. Post-TTS processing, the model promptly moves back to RAM, freeing up VRAM space for your Language Model (LLM) to load back in the missing layers. This adds about 1-2 seconds to both text generation by the LLM and the TTS engine.</p>
     <p>By transferring the entire model between RAM and VRAM, the Low VRAM option avoids fragmentation, ensuring the TTS model remains cohesive and accessible.</p>
    <p>This creates a TTS generation performance Boost for Low VRAM Users and is particularly beneficial for users with less than 2GB of free VRAM after loading their LLM, delivering a substantial 5-10x improvement in TTS generation speed.</p>

    <h3><strong>DeepSpeed Simplified:</strong></h3>

    <h4>What's DeepSpeed?</h4>
    <p>DeepSpeed, developed by Microsoft, is like a speed boost for Text-to-Speech (TTS) tasks. It's all about making TTS happen faster and more efficiently.</p>

    <h4>How Does It Speed Things Up?</h5>
    <p>   Model Parallelism: Spreads the work across multiple GPUs, making TTS models handle tasks more efficiently.</p>
    <p>   Memory Magic: Optimizes how memory is used, reducing the memory needed for large TTS models.</p>
    <p>   Efficient Everything: DeepSpeed streamlines both training and generating speech from text, making the whole process quicker.</p>

    <h4>Why Use DeepSpeed for TTS?</h4>
    <p>   2x-3x Speed Boost:</strong> Generates speech much faster than usual.</p>
    <p>   Handles More Load:</strong> Scales up to handle larger workloads with improved performance.</p>
    <p>   Smart Resource Use:</strong> Uses your computer's resources smartly, getting the most out of your hardware.</p>
    
    <p><strong>NOTE:</strong> DeepSpeed only works with the XTTSv2 Local model.</p>
    <p><strong>NOTE:</strong> Requires Nvidia Cuda Toolkit installation and correct CUDA_HOME path configuration.</p>

    <h4>How to Use It:</h4>
    <p>In CoquiTTS, the DeepSpeed checkbox will only be available if DeepSpeed is detected on your systsme. Check the checkbox and off you go!</p>

    <h2><strong>Other Features of Coqui_TTS Extension for Text-generation-webui</strong></h2>

    <h4>Start-up Checks</h4>
    <p>Ensures a minimum TTS version (0.21.1) is installed and provides an error/instructions if not.</p>
    <p>Downloads the Xtts model (version 2.0.2) to improve generation speed on 'local' methods. The API TTS version dynamically uses the latest model (2.0.3 at the time of writing).</p>
    <p>Allows dynamic switching between TTS models/methods with a 10-15 second transition.</p>

    <h4>TTS Models/Methods</h4>
    <p>It's worth noting that all models and methods can and do sound different from one another.</p>
    <p>   API TTS: Uses the current TTS model available that's downloaded by the TTS API process (e.g., version 2.0.3 at the time of writing).</p>
    <p>   API Local: Utilizes the 2.0.2 local model stored at /coqui_tts/models/xttsv2_2.0.2.</p>
    <p>   XTTSv2 Local: Employs the 2.0.2 local model and utilizes a distinct TTS generation method. Supports DeepSpeed acceleration.</p>

    <h4><strong>CURL Commands</strong></h4>
    <p>Example CURL commands:</p>

    <pre>
        <code>curl -X POST -H "Content-Type: application/json" "http://127.0.0.1:7851/api/lowvramsetting?new_low_vram_value=True"</code>
    </pre>
       <p>Replace <code>True</code> with <code>False</code> to disable Low VRAM mode.</p>
    <pre>
        <code>curl -X POST -H "Content-Type: application/json" -d '{"text": "This is text to generate as TTS", "voice": "female_01.wav", "language": "en", "output_file": "outputfile.wav"}' "http://127.0.0.1:7851/api/generate"</code>
    </pre>

    <h4><strong>Configuration Details</strong></h4>
    <p>Explanation of the <code>config.json</code> file:</p>

    <pre>
        <code>
  "activate": true,
  "autoplay": true,
  "deepspeed_activate": false,
  "ip_address": "127.0.0.1",
  "language": "English",
  "low_vram": false,
  "model_loaded": true,
  "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
  "model_version": "xttsv2_2.0.2",
  "port_number": "7851",
  "remove_trailing_dots": false,
  "show_text": true,
  "tts_method_api_local": false,
  "tts_method_api_tts": false,
  "tts_method_xtts_local": true,
  "voice": "female_01.wav"
</code>
    </pre>

    <ul>
        <li><strong>Activate:</strong> Used within the code, do not change. Default: true</li>
        <li><strong>Autoplay:</strong> Controls whether the TTS audio plays automatically within Text-generation-webUI. Default: true</li>
        <li><strong>DeepSpeed Activate:</strong> Controls whether the DeepSpeed option is activated or disabled in the Gradio interface. Default: false</li>
        <li><strong>IP Address:</strong> Specifies the default IP address for the web server. Default: "127.0.0.1"</li>
        <li><strong>Language:</strong> Specifies the default language to use for TTS. Default: "English"</li>
        <li><strong>Low VRAM:</strong> Controls whether the Low VRAM option is enabled or disabled. Default: false</li>
        <li><strong>Model Loaded:</strong> Used within the code, do not change. Default: true</li>
        <li><strong>Model Name:</strong> Specifies the model that the "API TTS" method will use for TTS generation.</li>
        <li><strong>Model Version:</strong> Specifies the version of the model that the "API Local" and "XTTSv2 Local" methods will use. The models are expected to be located in /coqui_tts/models/ directory. Default: "xttsv2_2.0.2"</li>
        <li><strong>Port Number:</strong> Specifies the default port number for the web server. Default: "7851"</li>
        <li><strong>Remove Trailing Dots:</strong> Controls whether trailing dots are removed from text segments before generation. Default: false</li>
        <li><strong>Show Text:</strong> Controls whether message text is shown under the audio player. Default: true</li>
        <li><strong>Below TTS methods:</strong> Only one can be True and the other two have to be False.</li>
        <li><strong>TTS Method API Local:</strong> Controls whether the "API Local" model/method is turned on or off. Default: false</li>
        <li><strong>TTS Method API TTS:</strong> Controls whether the "API TTS" model/method is turned on or off. Default: false</li>
        <li><strong>TTS Method XTTS Local:</strong> Controls whether the "XTTSv2 Local" model/method is turned on or off. Default: true</li>
        <li><strong>Voice:</strong> Specifies the default voice to use for TTS. Default: "female_01.wav"</li>
    </ul>

    <h4><strong>Debugging and TTS Generation Information:</strong></h4>
    <p>Command line outputs are more verbose to assist in understanding backend processes and debugging.</p>

    <h2><strong>References</strong></h2>
    <ul>
        <li>Coqui TTS Engine</li>
        <li><a href="https://coqui.ai/cpml.txt" target="_blank">Coqui License</a></li>
        <li><a href="https://github.com/coqui-ai/TTS" target="_blank">Coqui TTS GitHub Repository</a></li>
        <li>Extension coded by</li>
        <li><a href="https://github.com/erew123" target="_blank">Erew123 GitHub Profile</a></li>
        <li>Thanks to & Text-generation-WebUI</li>
        <li><a href="https://github.com/oobabooga/text-generation-webui" target="_blank">Ooobabooga GitHub Repository</a></li>
        <li>Thanks to</li>
        <li><a href="https://github.com/daswer123" target="_blank">daswer123 GitHub Profile</a></li>
        <li><a href="https://github.com/kanttouchthis" target="_blank">kanttouchthis GitHub Profile</a></li>

    </ul>
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
