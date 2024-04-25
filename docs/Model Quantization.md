If you can't find a quantized version of the model you need on HuggingFace, you can quantize the model yourself using this guide.
# GGUF
### Requirements:
- Python 3.10/3.11
- Git
- Downloaded model in Transformers format
  - Models in this format are a folder containing several large `pytorch_model-XXXXX-of-XXXXX` files with `.bin` or `.safetensors` extension,
  as well as several `.json' files.
- A lot of free disk space

## Windows
### Preparation
- Open a terminal in any folder with enough free space and run the following commands:
  - `git clone --depth 1 https://github.com/ggerganov/llama.cpp.git`
  - `cd llama.cpp`
  - `python -m pip install -r requirements.txt`.
- Move the folder with the model weights to llama.cpp\models (for convenience, this is optional)
- Download the `w64devkit-fortran-<version>.zip` file from https://github.com/skeeto/w64devkit/releases/latest and unzip it anywhere convenient.
Run `w64devkit.exe` and use the `cd` command reach the to the `llama.cpp` folder, for example: `cd "A:\LLM models\llama.cpp"`.
Run the `make` command and wait for the compilation to complete.
  

### Convert
- Run the `python convert.py models\<your model>\` command.
  - If you get a `TypeError: <model> must be converted with BpeVocab` error, add the `--vocab-type bpe` flag to the end of the command.
For example `python convert.py models/Meta-Llama-3-8B/ --vocab-type bpe`.

After that, the file `ggml-model-f16.gguf` or `ggml-model-f32.gguf` will appear in your model folder.


### Quantize
- At the command line with the llama.cpp folder open, run the `.\quantize.exe .\<the resulting converted .gguf file> <quantization method>` command.
  - The quantization methods are `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`.
  They are listed in ascending order of accuracy. The higher the accuracy, the larger the file, the greater the RAM/VRAM requirement, the lower the generation speed, but the better the quality of the model output.
  Accuracy below `Q4_0` is not recommended, `Q4_K_M`, `Q5_K_M` and `Q8_0` are recommended.
  - Example: `.\quantize.exe .\models\Meta-Llama-3-8B\ggml-model-f32.gguf Q8_0`.


## Linux
### Preparation
- Run the following commands in any convenient directory with plenty of free space.
  - `git clone --depth 1 https://github.com/ggerganov/llama.cpp.git`
  - `python3 -m pip install -r requirements.txt`
  - `make`.
- Move the folder with the model weights to llama.cpp/models (for convenience, this is optional)
  

### Convert
- Run the `python convert.py models/<your model>/` command.
  - If you get a `TypeError: <model> must be converted with BpeVocab` error, add the `--vocab-type bpe` flag to the end of the command.
For example `python convert.py models/Meta-Llama-3-8B/ --vocab-type bpe`.

After that, the file `ggml-model-f16.gguf` or `ggml-model-f32.gguf` will appear in your model folder.


### Quantize
- In the llama.cpp directory, run the `./quantize ./<converted .gguf file> <quantization method>` command.
  - The quantization methods are `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`.
  They are listed in ascending order of accuracy. The higher the accuracy, the larger the file, the greater the RAM/VRAM requirement, the lower the generation speed, but the better the quality of the model output.
  Accuracy below `Q4_0` is not recommended, `Q4_K_M`, `Q5_K_M` and `Q8_0` are recommended.
  - Example: `./quantize ./models/Meta-Llama-3-8B/ggml-model-f32.gguf Q8_0`.
