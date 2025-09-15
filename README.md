# Text Generation WebUI

## Table of Contents
- [Features](#features)
- [How to install](#how-to-install)
  - [Option 1: Portable builds](#option-1-portable-builds-get-started-in-1-minute)
  - [Option 2: Manual portable install with venv](#option-2-manual-portable-install-with-venv)
  - [Option 3: One-click installer](#option-3-one-click-installer)
- [Downloading models](#downloading-models)
- [Documentation](#documentation)
- [Community](#community)
- [Acknowledgment](#acknowledgment)

## Features
- Supports multiple local text generation backends, including llama.cpp, Transformers, ExLlamaV3, ExLlamaV2, and TensorRT-LLM (the latter via its own Dockerfile).  
- Easy setup: Choose between portable builds (zero setup, just unzip and run) for GGUF models on Windows/Linux/macOS, or the one-click installer that creates a self-contained installer_files directory.  
- 100% offline and private, with zero telemetry, external resources, or remote update requests.  
- File attachments: Upload text files, PDF documents, and .docx documents to talk about their contents.  
- Vision (multimodal models): Attach images to messages for visual understanding (tutorial).  
- Web search: Optionally search the internet with LLM-generated queries to add context to the conversation.  
- Aesthetic UI with dark and light themes.  
- Syntax highlighting for code blocks and LaTeX rendering for mathematical expressions.  
- instruct mode for instruction-following (like ChatGPT), and chat-instruct/chat modes for talking to custom characters.  
- Automatic prompt formatting using Jinja2 templates. You don't need to ever worry about prompt formats.  
- Edit messages, navigate between message versions, and branch conversations at any point.  
- Multiple sampling parameters and generation options for sophisticated text generation control.  
- Switch between different models in the UI without restarting.  
- Automatic GPU layers for GGUF models (on NVIDIA GPUs).  
- Free-form text generation in the Notebook tab without being limited to chat turns.  
- OpenAI-compatible API with Chat and Completions endpoints, including tool-calling support – see examples.  
- Extension support, with numerous built-in and user-contributed extensions available. See the wiki and [extensions directory](./extensions) for details.  

## How to install

### ✅ Option 1: Portable builds (get started in 1 minute)
No installation needed – just download, unzip and run. All dependencies included.

Compatible with GGUF (llama.cpp) models on Windows, Linux, and macOS.

Download from here: https://github.com/oobabooga/text-generation-webui/releases

### Option 2: Manual portable install with venv
Very fast setup that should work on any Python 3.9+:

```bash
# Clone repository
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies (choose appropriate file under requirements/portable for your hardware)
pip install -r requirements/portable/requirements.txt --upgrade

# Launch server (basic command)
python server.py --portable --api --auto-launch

# When done working, deactivate
deactivate

