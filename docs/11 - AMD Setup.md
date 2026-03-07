## Using an AMD GPU in Linux

Requires ROCm 6.4 to be installed.

### Option 1: One-click installer

The one-click installer (`start_linux.sh`) automatically detects AMD GPUs. When prompted, select the AMD option, or set the `GPU_CHOICE` environment variable before running:

```
GPU_CHOICE=B ./start_linux.sh
```

### Option 2: Manual conda install

Follow the manual conda installation instructions in the README, using the AMD PyTorch command:

```
pip3 install torch==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
```

Then install the project requirements with the AMD requirements file:

```
pip install -r requirements/full/requirements_amd.txt
```
