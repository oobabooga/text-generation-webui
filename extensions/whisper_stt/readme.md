![Screenshot from 2024-07-01 15-44-30](https://github.com/RandomInternetPreson/text-generation-webui_Whisperfix/assets/6488699/2f683f92-f93d-4003-aeaf-5d6ce405d6f1)

If "Submit the transcribed audio automatically" is not checked, the transcrition will only be seen in the console window.

This is a js replacement of the micriphone gradio element, as it would cause browsers to crash.

# whisper_stt

Allows you to enter your inputs in chat mode using your microphone.

## Settings

To adjust your default settings, you can add the following to your settings.yaml file.

```
whisper_stt-whipser_language: chinese
whisper_stt-whipser_model: tiny
whisper_stt-auto_submit: False
```

See source documentation for [model names](https://github.com/openai/whisper#available-models-and-languages) and [languages](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) you can use.
