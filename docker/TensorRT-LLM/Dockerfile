FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install Git
RUN apt update && apt install -y git

# System-wide TensorRT-LLM requirements
RUN apt install -y openmpi-bin libopenmpi-dev

# Set the working directory
WORKDIR /app

# Install text-generation-webui
RUN git clone https://github.com/oobabooga/text-generation-webui
WORKDIR /app/text-generation-webui
RUN pip install -r requirements.txt

# This is needed to avoid an error about "Failed to build mpi4py" in the next command
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install TensorRT-LLM
RUN pip3 install tensorrt_llm==0.10.0 -U --pre --extra-index-url https://pypi.nvidia.com

# Expose the necessary port for the Python server
EXPOSE 7860 5000

# Run the Python server.py script with the specified command
CMD ["python", "server.py", "--api", "--listen"]
