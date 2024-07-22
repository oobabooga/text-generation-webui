import importlib
import platform

from modules import shared
from modules.cache_utils import process_llamacpp_cache

imported_module = None


def llama_cpp_lib():
    global imported_module

    # Determine the platform
    is_macos = platform.system() == 'Darwin'

    # Define the library names based on the platform
    if is_macos:
        lib_names = [
            (None, 'llama_cpp')
        ]
    else:
        lib_names = [
            ('cpu', 'llama_cpp'),
            (None, 'llama_cpp_cuda'),
            (None, 'llama_cpp')
        ]

    for arg, lib_name in lib_names:
        should_import = (arg is None or getattr(shared.args, arg))

        if should_import:
            if imported_module and imported_module != lib_name:
                # Conflict detected, raise an exception
                raise Exception(f"Cannot import `{lib_name}` because `{imported_module}` is already imported. Switching to a different version of llama-cpp-python currently requires a server restart.")

            try:
                return_lib = importlib.import_module(lib_name)
                imported_module = lib_name
                monkey_patch_llama_cpp_python(return_lib)
                return return_lib
            except ImportError:
                continue

    return None


def monkey_patch_llama_cpp_python(lib):
    if getattr(lib.Llama, '_is_patched', False):
        # If the patch is already applied, do nothing
        return

    def my_generate(self, *args, **kwargs):
        if shared.args.streaming_llm:
            new_sequence = args[0]
            past_sequence = self._input_ids

            # Do the cache trimming for StreamingLLM
            process_llamacpp_cache(self, new_sequence, past_sequence)

        for output in self.original_generate(*args, **kwargs):
            yield output

    lib.Llama.original_generate = lib.Llama.generate
    lib.Llama.generate = my_generate

    # Set the flag to indicate that the patch has been applied
    lib.Llama._is_patched = True
