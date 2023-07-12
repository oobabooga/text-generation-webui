# ExLlama

### About

ExLlama is an extremely optimized GPTQ backend for LLaMA models. It features much lower VRAM usage and much higher speeds due to not relying on unoptimized transformers code.

### Usage

Configure text-generation-webui to use exllama via the UI or command line:
   - In the "Model" tab, set "Loader" to "exllama"
   - Specify `--loader exllama` on the command line

### Manual setup

No additional installation steps are necessary since an exllama package is already included in the requirements.txt. If this package fails to install for some reason, you can install it manually by cloning the original repository into your `repositories/` folder:

```
mkdir repositories
cd repositories
git clone https://github.com/turboderp/exllama
```

