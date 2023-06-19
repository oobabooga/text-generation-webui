# ExLlama

## About

ExLlama is an extremely optimized GPTQ backend ("loader") for LLaMA models. It features much lower VRAM usage and much higher speeds due to not relying on unoptimized transformers code.

## Installation:

1) Clone the ExLlama repository into your `text-generation-webui/repositories` folder:

```
mkdir repositories
cd repositories
git clone https://github.com/turboderp/exllama
```

2) Follow the remaining set up instructions in the official README: https://github.com/turboderp/exllama#exllama

3) Configure text-generation-webui to use exllama via the UI or command line:
   - In the "Model" tab, set "Loader" to "exllama"
   - Specify `--loader exllama` on the command line
