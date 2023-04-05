#!/bin/bash

usage='
This is a bash script using the API for oobabooga/text-generation-webui.

Make sure to start the web UI with the following flags:

python server.py --model MODEL --listen --no-stream

Optionally, you can also add the --share flag to generate a public gradio URL,
allowing you to use the API remotely.

Usage: bash ./api-bash-script.sh [PRESET] PROMPT

Examples:

preset="NovelAI-Sphinx Moth"
prompt="This is a conversation with your Assistant. The Assistant is very helpful and is eager to chat with you and answer your questions.\nYou: Explain quantum computing in simple terms\nAssistant:"
server=127.0.0.1 port=7860 seed=-1 bash ./api-bash-script.sh "$preset" "$prompt"

The word bash can be left out if you made the script executable (chmod +x api-bash-script.sh).
'

# If no arguments were passed, display a usage message and exit with an error code
if [[ "$#" -eq 0 ]]; then
    echo "$usage"
    exit 1
fi

# Set LANG variable to ensure consistent language environment across different systems
LANG=

# Load preset
if [[ -e "presets/$1.txt" ]]; then
    source "presets/$1.txt"
    shift
fi

# Set generation parameters
prompt=$(jq -n --arg prompt "$*" '$prompt')
max_new_tokens=${max_new_tokens:-200}
do_sample=${do_sample:-true}
temperature=$(printf "%.2f" "${temperature:-0.5}")
top_p=$(printf "%.2f" "${top_p:-0.9}")
typical_p=$(printf "%.2f" "${typical_p:-1}")
repetition_penalty=$(printf "%.2f" "${repetition_penalty:-1.05}")
encoder_repetition_penalty=$(printf "%.2f" "${encoder_repetition_penalty:-1.0}")
top_k=${top_k:-0}
min_length=${min_length:-0}
no_repeat_ngram_size=${no_repeat_ngram_size:-0}
num_beams=${num_beams:-1}
penalty_alpha=$(printf "%.2f" "${penalty_alpha:-0}")
length_penalty=$(printf "%.2f" "${length_penalty:-1}")
early_stopping=${early_stopping:-false}
seed=${seed:--1}

# Encode data as JSON string
data="{
  \"data\": [
    $(jq -n --arg prompt "$*" '$prompt'),
    $max_new_tokens,
    ${do_sample,,},
    $temperature,
    $top_p,
    $typical_p,
    $repetition_penalty,
    $encoder_repetition_penalty,
    $top_k,
    $min_length,
    $no_repeat_ngram_size,
    $num_beams,
    $penalty_alpha,
    $length_penalty,
    ${early_stopping,,},
    $seed
  ]
}"

# Make HTTP request and extract generated text
response=$(curl -s -X POST "http://${server:-127.0.0.1}:${port:-7860}/run/textgen" \
    -H "Content-Type: application/json" \
    -d "$data")

reply=$(jq -r '.data[0]' <<< "$response")
echo "$reply"
