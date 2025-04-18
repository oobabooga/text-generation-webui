import json
import pprint
import socket
import subprocess
import sys
import threading
import time

import llama_cpp_binaries
import requests

from modules import shared
from modules.logging_colors import logger

llamacpp_valid_cache_types = {"fp16", "q8_0", "q4_0"}


class LlamaServer:
    def __init__(self, model_path, server_path=None):
        """
        Initialize and start a server for llama.cpp models.
        """
        self.model_path = model_path
        self.server_path = server_path
        self.port = self._find_available_port()
        self.process = None
        self.max_context_length = None
        self.bos_token = "<s>"

        # Start the server
        self._start_server()

    def encode(self, text, add_bos_token=False, **kwargs):
        if self.bos_token and text.startswith(self.bos_token):
            add_bos_token = False

        url = f"http://localhost:{self.port}/tokenize"
        payload = {
            "content": text,
            "add_special": add_bos_token,
        }

        response = requests.post(url, json=payload)
        result = response.json()
        return result.get("tokens", [])

    def decode(self, token_ids, **kwargs):
        url = f"http://localhost:{self.port}/detokenize"
        payload = {
            "tokens": token_ids,
        }

        response = requests.post(url, json=payload)
        result = response.json()
        return result.get("content", "")

    def prepare_payload(self, state):
        # Prepare DRY
        dry_sequence_breakers = state['dry_sequence_breakers']
        if not dry_sequence_breakers.startswith("["):
            dry_sequence_breakers = "[" + dry_sequence_breakers + "]"
        dry_sequence_breakers = json.loads(dry_sequence_breakers)

        # Prepare the sampler order
        samplers = state["sampler_priority"]
        samplers = samplers.split("\n") if isinstance(samplers, str) else samplers
        penalty_found = False
        filtered_samplers = []
        for s in samplers:
            if s.strip() in ["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]:
                filtered_samplers.append(s.strip())
            elif not penalty_found and s.strip() == "repetition_penalty":
                filtered_samplers.append("penalties")
                penalty_found = True

        samplers = filtered_samplers

        # Move temperature to the end if temperature_last is true and temperature exists in the list
        if state["temperature_last"] and "temperature" in samplers:
            samplers.remove("temperature")
            samplers.append("temperature")

        payload = {
            "temperature": state["temperature"] if not state["dynamic_temperature"] else (state["dynatemp_low"] + state["dynatemp_high"]) / 2,
            "dynatemp_range": 0 if not state["dynamic_temperature"] else (state["dynatemp_high"] - state["dynatemp_low"]) / 2,
            "dynatemp_exponent": state["dynatemp_exponent"],
            "top_k": state["top_k"],
            "top_p": state["top_p"],
            "min_p": state["min_p"],
            "tfs_z": state["tfs"],
            "typical_p": state["typical_p"],
            "repeat_penalty": state["repetition_penalty"],
            "repeat_last_n": state["repetition_penalty_range"],
            "presence_penalty": state["presence_penalty"],
            "frequency_penalty": state["frequency_penalty"],
            "dry_multiplier": state["dry_multiplier"],
            "dry_base": state["dry_base"],
            "dry_allowed_length": state["dry_allowed_length"],
            "dry_penalty_last_n": state["repetition_penalty_range"],
            "dry_sequence_breakers": dry_sequence_breakers,
            "xtc_probability": state["xtc_probability"],
            "xtc_threshold": state["xtc_threshold"],
            "mirostat": state["mirostat_mode"],
            "mirostat_tau": state["mirostat_tau"],
            "mirostat_eta": state["mirostat_eta"],
            "grammar": state["grammar_string"],
            "seed": state["seed"],
            "ignore_eos": state["ban_eos_token"],
            "samplers": samplers,
        }

        if state['custom_token_bans']:
            to_ban = [[int(token_id), False] for token_id in state['custom_token_bans'].split(',')]
            payload["logit_bias"] = to_ban

        return payload

    def generate_with_streaming(
        self,
        prompt,
        state,
    ):
        url = f"http://localhost:{self.port}/completion"
        payload = self.prepare_payload(state)

        token_ids = self.encode(prompt, add_bos_token=state["add_bos_token"])
        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - len(token_ids)
        else:
            max_new_tokens = state['max_new_tokens']

        payload.update({
            "prompt": token_ids,
            "n_predict": max_new_tokens,
            "stream": True,
        })

        if shared.args.verbose:
            logger.info("GENERATE_PARAMS=")
            printable_payload = {k: v for k, v in payload.items() if k != "prompt"}
            pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(printable_payload)
            print()

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

    def get_logits(self, prompt, state, n_probs=128, use_samplers=False):
        """Get the logits/probabilities for the next token after a prompt"""
        url = f"http://localhost:{self.port}/completion"

        payload = self.prepare_payload(state)
        payload.update({
            "prompt": self.encode(prompt, add_bos_token=state["add_bos_token"]),
            "n_predict": 0,
            "logprobs": True,
            "n_probs": n_probs,
            "stream": False,
            "post_sampling_probs": use_samplers,
        })

        if shared.args.verbose:
            logger.info("GENERATE_PARAMS=")
            printable_payload = {k: v for k, v in payload.items() if k != "prompt"}
            pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(printable_payload)
            print()

        response = requests.post(url, json=payload)
        result = response.json()

        if "completion_probabilities" in result:
            if use_samplers:
                return result["completion_probabilities"][0]["top_probs"]
            else:
                return result["completion_probabilities"][0]["top_logprobs"]
        else:
            raise Exception(f"Unexpected response format: 'completion_probabilities' not found in {result}")

    def _get_max_context_length(self):
        """Get and store the model's maximum context length."""
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url).json()

        if "data" in response and len(response["data"]) > 0:
            model_info = response["data"][0]
            if "meta" in model_info and "n_vocab" in model_info["meta"]:
                self.max_context_length = model_info["meta"]["n_vocab"]

    def _get_bos_token(self):
        """Get and store the model's BOS token."""
        url = f"http://localhost:{self.port}/props"
        response = requests.get(url).json()
        if "bos_token" in response:
            self.bos_token = response["bos_token"]

    def _find_available_port(self):
        """Find an available port by letting the OS assign one."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Bind to port 0 to get an available port
            return s.getsockname()[1]

    def _start_server(self):
        """Start the llama.cpp server and wait until it's ready."""
        # Determine the server path
        if self.server_path is None:
            self.server_path = llama_cpp_binaries.get_binary_path()

        # Build the command
        cmd = [
            self.server_path,
            "--model", self.model_path,
            "--ctx-size", str(shared.args.n_ctx),
            "--n-gpu-layers", str(shared.args.n_gpu_layers),
            "--batch-size", str(shared.args.batch_size),
            "--port", str(self.port),
        ]

        if shared.args.flash_attn:
            cmd.append("--flash-attn")
        if shared.args.threads > 0:
            cmd += ["--threads", str(shared.args.threads)]
        if shared.args.threads_batch > 0:
            cmd += ["--threads-batch", str(shared.args.threads_batch)]
        if shared.args.no_mmap:
            cmd.append("--no-mmap")
        if shared.args.mlock:
            cmd.append("--mlock")
        if shared.args.tensor_split:
            cmd += ["--tensor-split", shared.args.tensor_split]
        if shared.args.numa:
            cmd += ["--numa", "distribute"]
        if shared.args.no_kv_offload:
            cmd.append("--no-kv-offload")
        if shared.args.row_split:
            cmd += ["--split-mode", "row"]
        if shared.args.cache_type != "fp16" and shared.args.cache_type in llamacpp_valid_cache_types:
            cmd += ["--cache-type-k", shared.args.cache_type, "--cache-type-v", shared.args.cache_type]
        if shared.args.compress_pos_emb != 1:
            cmd += ["--rope-freq-scale", str(1.0 / shared.args.compress_pos_emb)]

        # Start the server with pipes for output
        self.process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def filter_stderr():
            for line in iter(self.process.stderr.readline, ''):
                if not line.startswith(('srv ', 'slot ')):
                    sys.stderr.write(line)
                    sys.stderr.flush()

        threading.Thread(target=filter_stderr, daemon=True).start()

        # Wait for server to be healthy
        health_url = f"http://localhost:{self.port}/health"
        start_time = time.time()
        timeout = 3600 * 8  # 8 hours
        while time.time() - start_time < timeout:
            # Check if process is still alive
            if self.process.poll() is not None:
                # Process has terminated
                exit_code = self.process.poll()
                raise RuntimeError(f"Server process terminated unexpectedly with exit code: {exit_code}")

            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    break
            except:
                pass

            time.sleep(1)
        else:
            raise TimeoutError(f"Server health check timed out after {timeout} seconds")

        # Server is now healthy, get model info
        self._get_max_context_length()
        self._get_bos_token()
        return self.port

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
