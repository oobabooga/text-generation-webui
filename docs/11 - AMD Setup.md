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

### Systems with both integrated and discrete AMD GPUs

ROCm may detect the CPU's integrated GPU as well as a discrete GPU. If loading a model crashes on such a system, restrict ROCm to the supported GPU you intend to use before starting textgen.

Run `rocminfo` to identify the GPU agents by their model names and UUIDs. On Linux, [`ROCR_VISIBLE_DEVICES`](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html) accepts GPU indices or UUIDs. For example, if your discrete GPU is device `0`:

```bash
ROCR_VISIBLE_DEVICES=0 ./start_linux.sh
```

For a manual installation, activate its environment first, then use:

```bash
ROCR_VISIBLE_DEVICES=0 python server.py
```

Replace `0` with the GPU you identified; the discrete GPU is not always device `0`. You can also use its `GPU-...` UUID. Keep multiple supported GPUs available with a comma-separated list, such as `ROCR_VISIBLE_DEVICES=0,2`.

This is an environment variable, so set it in the launch command or launcher script, not in `user_data/CMD_FLAGS.txt`. It must be set before ROCm initializes. Use one visibility filter consistently: an existing `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` setting can further restrict the devices exposed by `ROCR_VISIBLE_DEVICES`.
