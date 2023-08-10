#!/usr/bin/env bash

#
#  This script is used to run the both the text-generation-webui and
#  sillytavern in a virtual environment.
#
#  It assumes that the virtual environment has already been created,
#  and that the .env file exists in the text-generation-webui directory.
#
#  If the .env file does not exist, then it will prompt the user for
#  the paths to the text-generation-webui directory and the virtual
#  environment. It will then create the .env file in the
#  text-generation-webui directory.
#
#  The .env file will contain the paths to the text-generation-webui
#  directory and the virtual environment, then the .env file will be
#  sourced to set the environment variables.
#
#  The script will then activate the virtual environment and run a python script that runs both
#  the text-generation-webui and sillytavern concurrently.
#

# If the .env file exists, then source it
if [ -f .env ]; then
    echo "Sourcing .env file for environment variables"
    source .env
elif [ -f ~/.config/text-generation-webui/.env ]; then
    echo "Sourcing .env file in ~/.config/text-generation-webui/ for environment variables"
    source ~/.config/text-generation-webui/.env
else
    echo "No .env file found. Please enter the paths to the text-generation-webui directory and the virtual environment"
fi

# If the .env file does not exist, then prompt the user for the paths to the text-generation-webui directory and the virtual environment
if [[ -z "$text_generation_webui_path" ]]; then
    echo "Enter the path to text-generation-webui"
    read text_generation_webui_path
    echo "text_generation_webui_path=$text_generation_webui_path" >$text_generation_webui_path/.env
fi

if [[ -z "$silly_tavern_path" ]]; then
    echo "Enter the path to Silly Tavern"
    read silly_tavern_path
    echo "silly_tavern_path=$silly_tavern_path" >>$text_generation_webui_path/.env
fi

if [[ -z "$venv_path" ]]; then
    echo "Enter the path to the text-generation-webui virtual environment"
    read venv_path
    echo "venv_path=$venv_path" >>$text_generation_webui_path/.env
fi

if [[ -z "$model_path" ]]; then
    echo "Enter the path to where you store your models"
    read model_path
    echo "model_path=$model_path" >>$text_generation_webui_path/.env
fi

if [[ -z "$model_name" ]]; then
    echo "Enter the name of the model you want to use"
    read model_name
    echo "model_name=$model_name" >>$text_generation_webui_path/.env
fi

if [[ -z "$loader" ]]; then
    echo "Enter the name of the loader you want to use"
    read loader
    echo "loader=$loader" >>$text_generation_webui_path/.env
fi

if [[ -z "$max_seq_len" ]]; then
    echo "Enter the max_seq_len you want to use. If you don't know, enter 2048"
    read max_seq_len
    echo "max_seq_len=$max_seq_len" >>$text_generation_webui_path/.env
fi

if [[ -z "$host" ]]; then
    echo "Enter the hostname you want to use. If you don't know, enter localhost"
    read host
    echo "host=$host" >>$text_generation_webui_path/.env
fi

if [[ -z "$port" ]]; then
    echo "Enter the port you want to use. If you don't know, enter 7860"
    read port
    echo "port=$port" >>$text_generation_webui_path/.env
fi

# If ~/.config/text-generation-webui/.env does not exist, then create it
if [ ! -f ~/.config/text-generation-webui/.env ]; then
    echo "Saving config to ~/.config/text-generation-webui/.env"
    mkdir -p ~/.config/text-generation-webui
    cp $text_generation_webui_path/.env ~/.config/text-generation-webui/.env
fi


source $venv_path/bin/activate
$venv_path/bin/python3 $text_generation_webui_path/tavern.py
