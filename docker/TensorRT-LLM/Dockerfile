FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# Install Python 3.12, Git, and OpenMPI
RUN apt update && apt install -y python3.12 python3-pip git build-essential openmpi-bin libopenmpi-dev

# Set the working directory
WORKDIR /app

# This is needed to avoid an error about "Failed to build mpi4py" in the next command
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install text-generation-webui
RUN git clone https://github.com/oobabooga/text-generation-webui
WORKDIR /app/text-generation-webui
RUN pip install --break-system-packages -r requirements/full/requirements.txt

# Install TensorRT-LLM
RUN pip3 install --break-system-packages tensorrt_llm==1.1.0 --extra-index-url https://pypi.nvidia.com

# Expose the necessary port for the Python server
EXPOSE 7860 5000

# Run the Python server.py script with the specified command
CMD ["python3", "server.py", "--api", "--listen"]
