import json
import os
import platform
import socket
import subprocess
import time

import requests

from modules import shared


class LlamaServer:
    def __init__(
        self,
        model_path,
        ctx_size=8192,
        n_gpu_layers=100,
        server_path=None,
        **kwargs
    ):
        """
        Initialize and start a server for llama.cpp models.
        """
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.server_path = server_path
        self.additional_args = kwargs
        self.port = self._find_available_port()
        self.process = None
        self.max_context_length = None

        # Start the server
        self._start_server()

    def _find_available_port(self):
        """Find an available port by letting the OS assign one."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Bind to port 0 to get an available port
            return s.getsockname()[1]

    def _start_server(self):
        """Start the llama.cpp server and wait until it's ready."""
        # Determine the server path
        if self.server_path is None:
            if shared.args.cpu:
                import llama_cpp_binaries
                self.server_path = llama_cpp_binaries.get_binary_path()
            elif shared.args.tensorcores:
                import llama_cpp_binaries_cuda_tensorcores
                self.server_path = llama_cpp_binaries_cuda_tensorcores.get_binary_path()
            else:
                import llama_cpp_binaries_cuda
                self.server_path = llama_cpp_binaries_cuda.get_binary_path()

        # Build the command
        cmd = [
            self.server_path,
            "--model", self.model_path,
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--port", str(self.port),
        ]

        # Add any additional arguments
        for key, value in self.additional_args.items():
            key = key.replace('_', '-')  # Convert snake_case to kebab-case for CLI args
            cmd.append(f"--{key}")
            if value is not None:
                cmd.append(str(value))

        # Set up environment
        env = os.environ.copy()
        if platform.system() == "Linux":
            # Add the LD_LIBRARY_PATH for Linux
            lib_path = "/usr/lib/x86_64-linux-gnu"
            if "LD_LIBRARY_PATH" in env:
                env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
            else:
                env["LD_LIBRARY_PATH"] = lib_path

        # Start the server
        # self.process = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.process = subprocess.Popen(cmd, env=env)

        # Wait for server to be healthy
        health_url = f"http://localhost:{self.port}/health"
        print(health_url)
        start_time = time.time()
        current_delay = 1.0
        max_delay = 10.0
        max_wait_time = 7200  # 2 hours

        while time.time() - start_time < max_wait_time:
            print("Trying health...", self.port)
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    break
            except Exception as e:
                print(e)
                pass

            time.sleep(current_delay)
            current_delay = min(current_delay * 1.5, max_delay)
        else:
            raise TimeoutError(f"Server health check timed out after {max_wait_time} seconds")

        # Server is now healthy, get model info
        self._get_max_context_length()
        return self.port

    def _get_max_context_length(self):
        """Get and store the model's maximum context length."""
        models_url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(models_url).json()

        if "data" in response and len(response["data"]) > 0:
            model_info = response["data"][0]
            if "meta" in model_info and "n_vocab" in model_info["meta"]:
                self.max_context_length = model_info["meta"]["n_vocab"]

    def get_logits(self, input_ids, n_probs=4096):
        """Get the logits/probabilities for the next token after a prompt"""
        url = f"http://localhost:{self.port}/completion"

        payload = {
            "prompt": input_ids,
            "n_predict": 0,
            "logprobs": True,
            "n_probs": n_probs,
            "post_sampling_probs": False
        }

        response = requests.post(url, json=payload)
        result = response.json()

        if "completion_probabilities" in result:
            return result["completion_probabilities"][0]["top_logprobs"]
        else:
            raise Exception(f"Unexpected response format: 'completion_probabilities' not found in {result}")

    def encode(self, text, **kwargs):
        url = f"http://localhost:{self.port}/tokenize"
        payload = {"content": text}
        response = requests.post(url, json=payload)
        result = response.json()
        return result.get("tokens", [])

    def decode(self, token_ids, **kwargs):
        url = f"http://localhost:{self.port}/detokenize"
        payload = {"tokens": token_ids}
        response = requests.post(url, json=payload)
        result = response.json()
        return result.get("content", "")

    def generate_with_streaming(
        self,
        prompt,
        state,
    ):
        url = f"http://localhost:{self.port}/completion"

        payload = {
            "prompt": prompt,
            "n_predict": state["max_new_tokens"],
            "temperature": state["temperature"],
            "top_k": state["top_k"],
            "top_p": state["top_p"],
            "min_p": state["min_p"],
            "tfs_z": state["tfs"],
            "typical_p": state["typical_p"],
            "repeat_penalty": state["repetition_penalty"],
            "repeat_last_n": state["repetition_penalty_range"],
            "presence_penalty": state["presence_penalty"],
            "frequency_penalty": state["frequency_penalty"],
            "mirostat": state["mirostat_mode"],
            "mirostat_tau": state["mirostat_tau"],
            "mirostat_eta": state["mirostat_eta"],
            "seed": state["seed"],
            "ignore_eos": state["ban_eos_token"],
            "stream": True
        }

        # Make a direct request with streaming enabled
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        full_text = ""

        # Process the streaming response
        for line in response.iter_lines():
            if shared.stop_everything:
                break

            if line:
                try:
                    # Check if the line starts with "data: " and remove it
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]  # Remove the "data: " prefix

                    # Parse the JSON data
                    data = json.loads(line_str)

                    # Extract the token content
                    if 'content' in data:
                        token_text = data['content']
                        full_text += token_text
                        yield full_text

                    # Check if generation is complete
                    if data.get('stop', False):
                        break

                except json.JSONDecodeError as e:
                    # Log the error and the problematic line
                    print(f"JSON decode error: {e}")
                    print(f"Problematic line: {line}")
                    continue

    def __enter__(self):
        """Support for context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager."""
        self.stop()

    def __del__(self):
        """Cleanup when the object is deleted."""
        self.stop()

    def stop(self):
        """Stop the server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

            self.process = None
