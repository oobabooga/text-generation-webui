#!/bin/bash
source ./venv/bin/activate
python server.py --load-in-8bit --chat \
  --extensions character_bias silero_tts \
  --model llama-13b-hf
