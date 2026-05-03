import json
import os
import pprint
import shlex
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, List

import requests

from modules import shared
from modules.image_utils import (
    convert_image_attachments_to_pil,
    convert_openai_messages_to_images,
    convert_pil_to_base64
)
from modules.logging_colors import logger
from modules.utils import resolve_model_path

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
        self.session = requests.Session()
        self.vocabulary_size = None
        self.n_ctx = None
        self.bos_token = "<s>"
        self.media_marker = "<__media__>"
        self.last_prompt_token_count = 0

        # Start the server
        self._start_server()

    def encode(self, text, add_bos_token=False, **kwargs):
        if self.bos_token and text.startswith(self.bos_token):
            add_bos_token = False

        url = f"http://127.0.0.1:{self.port}/tokenize"
        payload = {
            "content": text,
            "add_special": add_bos_token,
        }

        response = self.session.post(url, json=payload)
        result = response.json()
        return result.get("tokens", [])

    def decode(self, token_ids, **kwargs):
        url = f"http://127.0.0.1:{self.port}/detokenize"
        payload = {
            "tokens": token_ids,
        }

        response = self.session.post(url, json=payload)
        result = response.json()
        return result.get("content", "")

    def prepare_payload(self, state):
        payload = {
            "temperature": state["temperature"] if not state["dynamic_temperature"] else (state["dynatemp_low"] + state["dynatemp_high"]) / 2,
            "dynatemp_range": 0 if not state["dynamic_temperature"] else (state["dynatemp_high"] - state["dynatemp_low"]) / 2,
            "dynatemp_exponent": state["dynatemp_exponent"],
            "top_k": state["top_k"],
            "top_p": state["top_p"],
            "min_p": state["min_p"],
            "top_n_sigma": state["top_n_sigma"] if state["top_n_sigma"] > 0 else -1,
            "adaptive_target": state["adaptive_target"] if state["adaptive_target"] > 0 else -1,
            "adaptive_decay": state["adaptive_decay"],
            "typical_p": state["typical_p"],
            "repeat_penalty": state["repetition_penalty"],
            "repeat_last_n": state["repetition_penalty_range"],
            "presence_penalty": state["presence_penalty"],
            "frequency_penalty": state["frequency_penalty"],
            "dry_multiplier": state["dry_multiplier"],
            "dry_base": state["dry_base"],
            "dry_allowed_length": state["dry_allowed_length"],
            "dry_penalty_last_n": state["repetition_penalty_range"],
            "xtc_probability": state["xtc_probability"],
            "xtc_threshold": state["xtc_threshold"],
            "mirostat": state["mirostat_mode"],
            "mirostat_tau": state["mirostat_tau"],
            "mirostat_eta": state["mirostat_eta"],
            "grammar": state["grammar_string"],
            "seed": state["seed"],
            "ignore_eos": state["ban_eos_token"],
        }

        # DRY
        dry_sequence_breakers = state['dry_sequence_breakers']
        if not dry_sequence_breakers.startswith("["):
            dry_sequence_breakers = "[" + dry_sequence_breakers + "]"

        dry_sequence_breakers = json.loads(dry_sequence_breakers)
        payload["dry_sequence_breakers"] = dry_sequence_breakers

        # Sampler order
        if state["sampler_priority"]:
            samplers = state["sampler_priority"]
            samplers = samplers.split("\n") if isinstance(samplers, str) else samplers
            filtered_samplers = []

            penalty_found = False
            for s in samplers:
                if s.strip() in ["dry", "top_k", "top_p", "top_n_sigma", "min_p", "temperature", "xtc"]:
                    filtered_samplers.append(s.strip())
                elif s.strip() == "typical_p":
                    filtered_samplers.append("typ_p")
                elif not penalty_found and s.strip() == "repetition_penalty":
                    filtered_samplers.append("penalties")
                    penalty_found = True

            # Move temperature to the end if temperature_last is true and temperature exists in the list
            if state["temperature_last"] and "temperature" in filtered_samplers:
                filtered_samplers.remove("temperature")
                filtered_samplers.append("temperature")

            # adaptive-p replaces the default dist sampler; llama.cpp always
            # places it at the end of the chain regardless of position, so we
            # activate it based on the parameter value rather than sampler order.
            if state.get("adaptive_target", 0) > 0:
                filtered_samplers.append("adaptive_p")

            payload["samplers"] = filtered_samplers

        logit_bias = []
        if state['custom_token_bans']:
            logit_bias.extend([[int(token_id.strip()), False] for token_id in state['custom_token_bans'].split(',') if token_id.strip()])

        if state.get('logit_bias'):
            for token_id_str, bias in state['logit_bias'].items():
                logit_bias.append([int(token_id_str), bias])

        if logit_bias:
            payload["logit_bias"] = logit_bias

        n_probs = state.get('logprobs', 0)
        if n_probs and n_probs > 0:
            payload["n_probs"] = n_probs

        return payload

    def _process_images_for_generation(self, state: dict) -> List[Any]:
        """
        Process all possible image inputs and return PIL images
        """
        pil_images = []
        # Source 1: Web UI (from chatbot_wrapper)
        if 'image_attachments' in state and state['image_attachments']:
            pil_images.extend(convert_image_attachments_to_pil(state['image_attachments']))
        # Source 2: Chat Completions API (/v1/chat/completions)
        elif 'history' in state and state.get('history', {}).get('messages'):
            pil_images.extend(convert_openai_messages_to_images(state['history']['messages']))
        # Source 3: Legacy Completions API (/v1/completions)
        elif 'raw_images' in state and state['raw_images']:
            pil_images.extend(state.get('raw_images', []))

        return pil_images

    def is_multimodal(self) -> bool:
        """Check if this model supports multimodal input."""
        return shared.args.mmproj not in [None, 'None']

    def generate_with_streaming(self, prompt, state):
        url = f"http://127.0.0.1:{self.port}/completion"
        payload = self.prepare_payload(state)

        pil_images = []

        if shared.is_multimodal:
            pil_images = self._process_images_for_generation(state)

        if pil_images:
            # Multimodal case
            IMAGE_TOKEN_COST_ESTIMATE = 600  # A safe, conservative estimate per image

            # Translate placeholders to the server's actual media marker.
            prompt = prompt.replace("<__media__>", self.media_marker)

            base64_images = [convert_pil_to_base64(img) for img in pil_images]
            payload["prompt"] = {
                "prompt_string": prompt,
                "multimodal_data": base64_images
            }

            # Calculate an estimated token count
            text_tokens = self.encode(prompt, add_bos_token=state["add_bos_token"])
            self.last_prompt_token_count = len(text_tokens) + (len(pil_images) * IMAGE_TOKEN_COST_ESTIMATE)
        else:
            # Text only case
            token_ids = self.encode(prompt, add_bos_token=state["add_bos_token"])
            self.last_prompt_token_count = len(token_ids)
            payload["prompt"] = token_ids

        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - self.last_prompt_token_count
        else:
            max_new_tokens = state['max_new_tokens']

        payload.update({
            "n_predict": max_new_tokens,
            "stream": True,
            "cache_prompt": True
        })

        if shared.args.verbose:
            logger.info("GENERATE_PARAMS=")
            printable_payload = {k: v for k, v in payload.items() if k != "prompt"}
            pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(printable_payload)
            print()

        # Make the generation request
        response = self.session.post(url, json=payload, stream=True)
        try:
            if response.status_code == 400 and response.json().get("error", {}).get("type") == "exceed_context_size_error":
                logger.error("The request exceeds the available context size, try increasing it")
                return
            else:
                response.raise_for_status()  # Raise an exception for HTTP errors

            full_text = ""
            self.last_completion_probabilities = []

            # Process the streaming response
            stop_event = state.get('stop_event')
            for line in response.iter_lines():
                if shared.stop_everything or (stop_event and stop_event.is_set()):
                    break

                if not line:
                    continue

                try:
                    line = line.decode('utf-8')

                    # Check if the line starts with "data: " and remove it
                    if line.startswith('data: '):
                        line = line[6:]  # Remove the "data: " prefix

                    # Parse the JSON data
                    data = json.loads(line)

                    # Extract the token content
                    if data.get('content', ''):
                        full_text += data['content']
                        yield full_text

                    # Capture logprobs if present
                    if 'completion_probabilities' in data:
                        self.last_completion_probabilities.extend(data['completion_probabilities'])

                    # Check if generation is complete
                    if data.get('stop', False):
                        break

                except json.JSONDecodeError as e:
                    # Log the error and the problematic line
                    print(f"JSON decode error: {e}")
                    print(f"Problematic line: {line}")
                    continue
        finally:
            response.close()

    def generate(self, prompt, state):
        output = ""
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def get_logits(self, prompt, state, n_probs=128, use_samplers=False):
        """Get the logits/probabilities for the next token after a prompt"""
        url = f"http://127.0.0.1:{self.port}/completion"

        payload = self.prepare_payload(state)
        payload.update({
            "prompt": self.encode(prompt, add_bos_token=state["add_bos_token"]),
            "n_predict": 0,
            "logprobs": True,
            "n_probs": n_probs,
            "stream": False,
            "post_sampling_probs": use_samplers,
        })

        if shared.args.verbose and use_samplers:
            logger.info("GENERATE_PARAMS=")
            printable_payload = {k: v for k, v in payload.items() if k != "prompt"}
            pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(printable_payload)
            print()

        def _try_fetch_logits():
            for retry in range(5):
                response = self.session.post(url, json=payload)
                result = response.json()

                if "completion_probabilities" in result:
                    if use_samplers:
                        return result["completion_probabilities"][0]["top_probs"]
                    else:
                        return result["completion_probabilities"][0]["top_logprobs"]

                time.sleep(0.05)
            else:
                raise Exception(f"Unexpected response format: 'completion_probabilities' not found in {result}")

        result = _try_fetch_logits()
        for entry in result:
            if not entry.get('token'):
                entry['token'] = self.decode([entry['id']])
        return result

    def get_prompt_logprob_entries(self, token_ids, n_probs=5, prompt=""):
        """Get logprob entries for prompt tokens via a single n_predict=0 request.

        Requires llama.cpp server with prompt_logprobs support.
        Returns entries in the standard format for format_completion_logprobs().
        """
        token_ids_list = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)

        url = f"http://127.0.0.1:{self.port}/completion"
        payload = {
            "prompt": token_ids_list,
            "n_predict": 0,
            "n_probs": n_probs,
            "prompt_logprobs": True,
            "stream": False,
            "cache_prompt": False,
        }

        response = self.session.post(url, json=payload)
        result = response.json()

        prompt_probs = result.get("prompt_probabilities", [])
        if not prompt_probs:
            return []

        # Null first token (no conditioning context); use empty string for BOS
        # or tokens that don't appear at the start of the prompt text.
        first_token_str = self.decode([token_ids_list[0]])
        if self.bos_token and first_token_str == self.bos_token:
            first_token_str = ""
        elif not prompt.startswith(first_token_str):
            first_token_str = ""

        entries = [{"token": first_token_str, "null_logprob": True}]
        entries.extend(prompt_probs)
        return entries

    def _get_vocabulary_size(self):
        """Get and store the model's vocabulary size."""
        url = f"http://127.0.0.1:{self.port}/v1/models"
        response = self.session.get(url).json()

        if "data" in response and len(response["data"]) > 0:
            model_info = response["data"][0]
            if "meta" in model_info and "n_vocab" in model_info["meta"]:
                self.vocabulary_size = model_info["meta"]["n_vocab"]

    def _get_bos_token(self):
        """Get and store the model's BOS token and context size."""
        url = f"http://127.0.0.1:{self.port}/props"
        response = self.session.get(url).json()
        if "bos_token" in response:
            self.bos_token = response["bos_token"]

        # Get actual n_ctx from the server (important when --fit auto-selects it)
        n_ctx = response.get("default_generation_settings", {}).get("n_ctx")
        if n_ctx:
            self.n_ctx = n_ctx

        # Get the server's media marker for multimodal support.
        # llama.cpp server generates a random marker each restart;
        # we must use it instead of the default <__media__>.
        if "media_marker" in response:
            self.media_marker = response["media_marker"]

    def _is_port_available(self, port):
        """Check if a port is available for use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                return True
            except OSError:
                return False

    def _find_available_port(self):
        """Find an available port, preferring main port + 5."""
        preferred_port = shared.args.api_port + 5
        if self._is_port_available(preferred_port):
            return preferred_port

        # Fall back to OS-assigned random port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _start_server(self):
        """Start the llama.cpp server and wait until it's ready."""
        # Determine the server path
        if self.server_path is None:
            if shared.args.ik:
                try:
                    import ik_llama_cpp_binaries
                except ImportError:
                    raise ImportError("--ik requires the ik_llama_cpp_binaries package. Install it with: pip install <ik_llama_cpp_binaries wheel URL>")

                self.server_path = ik_llama_cpp_binaries.get_binary_path()
            else:
                import llama_cpp_binaries
                self.server_path = llama_cpp_binaries.get_binary_path()

        # Build the command
        cmd = [
            self.server_path,
            "--model", self.model_path,
            "--batch-size", str(shared.args.batch_size),
            "--ubatch-size", str(shared.args.ubatch_size),
            "--port", str(self.port),
            "--no-webui",
            "--flash-attn", "on",
        ]

        if shared.args.ctx_size > 0:
            cmd += ["--ctx-size", str(shared.args.ctx_size)]
        elif shared.args.gpu_layers >= 0:
            cmd += ["--ctx-size", "8192"]

        if shared.args.gpu_layers >= 0:
            cmd += ["--gpu-layers", str(shared.args.gpu_layers), "--fit", "off"]
        else:
            cmd += ["--fit", "on"]
            cmd += ["--fit-ctx", "8192"]
            if shared.args.fit_target:
                cmd += ["--fit-target", shared.args.fit_target]

        if shared.args.threads > 0:
            cmd += ["--threads", str(shared.args.threads)]
        if shared.args.threads_batch > 0:
            cmd += ["--threads-batch", str(shared.args.threads_batch)]
        if shared.args.cpu_moe:
            cmd.append("--cpu-moe")
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
        if shared.args.split_mode != "layer":
            cmd += ["--split-mode", shared.args.split_mode]
        cache_type = "fp16"
        if shared.args.cache_type != "fp16" and shared.args.cache_type in llamacpp_valid_cache_types:
            cmd += ["--cache-type-k", shared.args.cache_type, "--cache-type-v", shared.args.cache_type]
            cache_type = shared.args.cache_type
        if shared.args.mmproj not in [None, 'None']:
            path = Path(shared.args.mmproj)
            if not path.exists():
                path = shared.user_data_dir / 'mmproj' / shared.args.mmproj

            if path.exists():
                cmd += ["--mmproj", str(path)]
        if shared.args.model_draft not in [None, 'None']:
            path = resolve_model_path(shared.args.model_draft)

            if path.is_file():
                model_file = path
            else:
                model_file = sorted(path.glob('*.gguf'))[0]

            cmd += ["--model-draft", str(model_file)]
            if shared.args.draft_max > 0:
                cmd += ["--draft-max", str(shared.args.draft_max)]
            if shared.args.gpu_layers_draft > 0:
                cmd += ["--gpu-layers-draft", str(shared.args.gpu_layers_draft)]
            if shared.args.device_draft:
                cmd += ["--device-draft", shared.args.device_draft]
            if shared.args.ctx_size_draft > 0:
                cmd += ["--ctx-size-draft", str(shared.args.ctx_size_draft)]
        if shared.args.spec_type != 'none':
            cmd += ["--spec-type", shared.args.spec_type]
            cmd += ["--draft-max", str(shared.args.draft_max)]
            cmd += ["--draft-min", "48"]
            cmd += ["--spec-ngram-size-n", str(shared.args.spec_ngram_size_n)]
            cmd += ["--spec-ngram-size-m", str(shared.args.spec_ngram_size_m)]
            cmd += ["--spec-ngram-min-hits", str(shared.args.spec_ngram_min_hits)]
        cmd += ["--parallel", str(shared.args.parallel)]
        if shared.args.streaming_llm:
            cmd += ["--cache-reuse", "1"]
            cmd += ["--swa-full"]
        if shared.args.extra_flags:
            # Clean up the input
            extra_flags = shared.args.extra_flags.strip()
            if extra_flags.startswith('"') and extra_flags.endswith('"'):
                extra_flags = extra_flags[1:-1].strip()
            elif extra_flags.startswith("'") and extra_flags.endswith("'"):
                extra_flags = extra_flags[1:-1].strip()

            if extra_flags.startswith('-'):
                # New literal format: "--jinja --rpc 1222,1222"
                cmd += shlex.split(extra_flags)
            else:
                # Legacy format: "flag1=value1,flag2,flag3=value3"
                long_form_only = {'rpc', 'fit', 'pos', 'ppl'}

                for flag_item in extra_flags.split(','):
                    flag_item = flag_item.strip()
                    if '=' in flag_item:
                        flag, value = flag_item.split('=', 1)
                        flag = flag.strip()
                        value = value.strip()
                        if len(flag) <= 3 and flag not in long_form_only:
                            cmd += [f"-{flag}", value]
                        else:
                            cmd += [f"--{flag}", value]
                    else:
                        if len(flag_item) <= 3 and flag_item not in long_form_only:
                            cmd.append(f"-{flag_item}")
                        else:
                            cmd.append(f"--{flag_item}")

        # Patch flags for ik_llama.cpp compatibility
        if shared.args.ik:
            cmd = _patch_cmd_for_ik(cmd)

        env = os.environ.copy()
        if os.name == 'posix':
            current_path = env.get('LD_LIBRARY_PATH', '')
            if current_path:
                env['LD_LIBRARY_PATH'] = f"{current_path}:{os.path.dirname(self.server_path)}"
            else:
                env['LD_LIBRARY_PATH'] = os.path.dirname(self.server_path)

        if shared.args.verbose:
            logger.info("llama-server command-line flags:")
            print(' '.join(str(item) for item in cmd[1:]))
            print()

        gpu_layers_str = "auto" if shared.args.gpu_layers < 0 else str(shared.args.gpu_layers)
        ctx_size_str = "auto" if shared.args.ctx_size == 0 and shared.args.gpu_layers < 0 else str(shared.args.ctx_size or 8192)
        logger.info(f"Using gpu_layers={gpu_layers_str} | ctx_size={ctx_size_str} | cache_type={cache_type}")
        # Start the server with pipes for output
        self.process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env
        )

        threading.Thread(target=filter_stderr_with_progress, args=(self.process.stderr,), daemon=True).start()

        # Wait for server to be healthy
        health_url = f"http://127.0.0.1:{self.port}/health"
        while True:
            # Check if process is still alive
            exit_code = self.process.poll()
            if exit_code is not None:
                raise RuntimeError(f"Server process terminated unexpectedly with exit code: {exit_code}")

            try:
                response = self.session.get(health_url)
                if response.status_code == 200:
                    break
            except Exception:
                pass

            time.sleep(1)

        # Server is now healthy, get model info
        self._get_vocabulary_size()
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
                self.process.wait(timeout=5)

            self.process = None


def filter_stderr_with_progress(process_stderr):
    """
    Reads stderr lines from a process, filters out noise, and displays progress updates
    inline (overwriting the same line) until completion.
    """
    progress_re = re.compile(r'slot update_slots: id.*progress = (\d+\.\d+)')
    ansi_re = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    log_prefix_re = re.compile(r'^[IWED] ')
    last_was_progress = False

    try:
        # Read in binary mode and decode manually
        buffer = b""
        while True:
            # Read chunks aggressively to prevent buffer overflow
            chunk = process_stderr.read(4096)
            if not chunk:
                break

            buffer += chunk

            # Process complete lines
            while b'\n' in buffer:
                line_bytes, buffer = buffer.split(b'\n', 1)
                try:
                    line = line_bytes.decode('utf-8', errors='replace').strip('\r\n')
                    line = log_prefix_re.sub('', ansi_re.sub('', line))
                    if line:  # Process non-empty lines
                        match = progress_re.search(line)

                        if match:
                            progress = float(match.group(1))

                            # Extract just the part from "prompt processing" onwards
                            prompt_processing_idx = line.find('prompt processing')
                            if prompt_processing_idx != -1:
                                display_line = line[prompt_processing_idx:]
                            else:
                                display_line = line  # fallback to full line

                            # choose carriage return for in-progress or newline at completion
                            end_char = '\r' if progress < 1.0 else '\n'
                            print(display_line, end=end_char, file=sys.stderr, flush=True)
                            last_was_progress = (progress < 1.0)

                        # skip noise lines
                        elif not (line.startswith(('srv ', 'slot ')) or 'log_server_r: request: GET /health' in line or 'No parser definition detected' in line):
                            # if we were in progress, finish that line first
                            if last_was_progress:
                                print(file=sys.stderr)

                            print(line, file=sys.stderr, flush=True)
                            last_was_progress = False

                except Exception:
                    continue

    except (ValueError, IOError):
        pass
    finally:
        try:
            process_stderr.close()
        except Exception:
            pass


def _patch_cmd_for_ik(cmd):
    """
    Rewrite upstream llama.cpp flags to ik_llama.cpp equivalents:
      --no-webui           → --webui none
      --fit off            → (removed)
      --fit on / --fit-ctx → --fit (bare flag)
      --fit-target         → --fit-margin
      --cache-reuse        → (removed, unsupported)
      --swa-full           → (removed, unsupported)
      --split-mode row     → --split-mode graph
      --split-mode tensor  → --split-mode graph
    """
    # Add Hadamard KV cache rotation when using quantized cache types.
    # This significantly improves quantized cache quality (especially q4_0)
    # and is a no-op for MLA models like DeepSeek.
    if shared.args.cache_type in ("q8_0", "q4_0"):
        cmd += ["-khad", "-vhad"]

    patched = []
    i = 0
    while i < len(cmd):
        arg = cmd[i]

        if arg == "--no-webui":
            patched += ["--webui", "none"]
        elif arg == "--fit" and i + 1 < len(cmd) and cmd[i + 1] in ("on", "off"):
            val = cmd[i + 1]
            i += 1
            if val == "on":
                patched.append("--fit")
            # "off" → drop entirely
        elif arg == "--fit-ctx":
            patched.append("--fit")
            i += 1  # skip the value
        elif arg == "--fit-target":
            patched.append("--fit-margin")
        elif arg == "--cache-reuse":
            i += 1  # skip the value
        elif arg == "--split-mode" and i + 1 < len(cmd) and cmd[i + 1] in ("row", "tensor"):
            patched += ["--split-mode", "graph"]
            i += 1  # skip the value
        elif arg == "--swa-full":
            pass  # bare flag, just drop it
        else:
            patched.append(arg)

        i += 1

    return patched
