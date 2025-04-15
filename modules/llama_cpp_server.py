import os
import platform
import socket
import subprocess
import time

import requests


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

    def _request_with_retry(
        self,
        url,
        payload=None,
        method="POST",
        max_retries=5,
        initial_delay=0.01,
        max_delay=1.0,
        health_check=False,
        max_wait_time=None
    ):
        """Make a request with exponential backoff retry"""
        delay = initial_delay
        start_time = time.time()
        attempt = 0

        while max_wait_time is None or time.time() - start_time < max_wait_time:
            try:
                if method.upper() == "GET":
                    response = requests.get(url)
                else:
                    response = requests.post(url, json=payload)

                # Special handling for health check
                if health_check:
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 503:
                        # Still loading, continue retrying
                        pass
                    else:
                        response.raise_for_status()
                else:
                    # Normal handling for other requests
                    response.raise_for_status()
                    return response.json()

            except requests.exceptions.RequestException:
                pass

            # Increment attempt count for regular retry logic
            if not health_check and attempt >= max_retries - 1:
                raise Exception(f"Failed to connect after {max_retries} attempts")

            # Sleep with backoff
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            attempt += 1

        if max_wait_time is not None:
            raise TimeoutError(f"Request timed out after {max_wait_time} seconds")
        else:
            raise Exception(f"Failed to connect after {max_retries} attempts")

    def _start_server(self):
        """Start the llama.cpp server and wait until it's ready."""
        # Determine the server path
        if self.server_path is None:
            # Default path from the original command
            system = platform.system()
            if system == "Windows":
                executable = "llama-server.exe"
            else:
                executable = "llama-server"
            self.server_path = os.path.join("..", "bin", "llama.cpp", executable)

        # Build the command
        cmd = [
            self.server_path,
            "--model", self.model_path,
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--port", str(self.port)
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
        self.process = subprocess.Popen(cmd, env=env)

        # Wait for server to be healthy using _request_with_retry with appropriate parameters
        health_url = f"http://localhost:{self.port}/health"
        self._request_with_retry(
            health_url,
            method="GET",
            initial_delay=1.0,
            max_delay=10.0,
            health_check=True,
            max_wait_time=7200  # 2 hours max wait time
        )

        # Server is now healthy, get model info
        self._get_max_context_length()
        return self.port

    def start(self):
        """
        For backward compatibility - returns port if server is already running.
        Otherwise starts the server.
        """

        if self.process is None:
            return self._start_server()

        return self.port

    def _get_max_context_length(self):
        """Get and store the model's maximum context length."""
        try:
            models_url = f"http://localhost:{self.port}/v1/models"
            response = self._request_with_retry(
                models_url,
                method="GET",
                max_retries=20,
                initial_delay=1.0,
                max_delay=10.0
            )

            if response and "data" in response and len(response["data"]) > 0:
                model_info = response["data"][0]
                if "meta" in model_info and "n_ctx_train" in model_info["meta"]:
                    self.max_context_length = model_info["meta"]["n_ctx_train"]
        except Exception as e:
            print(f"Failed to get model info: {e}")

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
