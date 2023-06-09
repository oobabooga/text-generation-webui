# Performance optimizations

In order to get the highest possible performance for your hardware, you can try compiling the following 3 backends manually instead of relying on the pre-compiled binaries that are part of `requirements.txt`:

* AutoGPTQ (the default GPTQ loader)
* GPTQ-for-LLaMa (secondary GPTQ loader)
* llama-cpp-python

If you go this route, you should update the Python requirements for the webui in the future with

```
pip install -r requirements-minimal.txt --upgrade
```

and then install the up-to-date backends using the commands below. The file `requirements-minimal.txt` contains the all requirements except for the pre-compiled wheels for GPTQ and llama-cpp-python.

## AutoGPTQ

```
conda activate textgen
pip uninstall auto-gptq -i
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .
```

## GPTQ-for-LLaMa

```
conda activate textgen
pip uninstall quant-cuda -y
cd text-generation-webui/repositories
rm -r GPTQ-for-LLaMa
git clone https://github.com/oobabooga/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
python setup_cuda.py install
```

## llama-cpp-python

If you do not have a GPU:

```
conda activate textgen
pip uninstall -y llama-cpp-python
pip install llama-cpp-python
```

If you have a GPU, use the commands here instead: [llama.cpp-models.md#gpu-acceleration](https://github.com/oobabooga/text-generation-webui/blob/main/docs/llama.cpp-models.md#gpu-acceleration)
