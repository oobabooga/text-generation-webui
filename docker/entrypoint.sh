#!/bin/sh

mkdir /app/repositories/alpaca_lora_4bit
git config --global http.postBuffer 1048576000
git clone ${LORA_4BIT_URL} /app/repositories/alpaca_lora_4bit
cd /app/repositories/alpaca_lora_4bit || exit
git checkout ${LORA_4BIT_BRANCH}
git reset --hard ${LORA_4BIT_SHA}
echo "git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit" >> requirements.txt
pip install -r requirements.txt
pip install scipy
cd /app || exit

exec "$@"

# pip install git+https://github.com/johnsmith0031/alpaca_lora_4bit@winglian-setup_pip

python3 server.py --model TheBloke_wizardLM-7B-GPTQ --wbits 4 --listen --auto-devices --groupsize 128 --pre_layer 30 --model_type LLaMA --monkey-patch --notebook --verbose --extensions api

# the following examples have been tested with the files linked in docs/README_docker.md:
# example running 13b with 4bit/128 groupsize        : CLI_ARGS=--model llama-13b-4bit-128g --wbits 4 --listen --groupsize 128 --pre_layer 25
# example with loading api extension and public share: CLI_ARGS=--model llama-7b-4bit --wbits 4 --listen --auto-devices --no-stream --extensions api --share
# example running 7b with 8bit groupsize             : CLI_ARGS=--model llama-7b --load-in-8bit --listen --auto-devices