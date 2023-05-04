# From https://github.com/d8ahazard/sd_dreambooth_extension/tree/main/bitsandbytes_windows

import errno
import os
from pathlib import Path
from typing import Set, Union
from warnings import warn

from bitsandbytes.cuda_setup.env_vars import get_potentially_lib_path_containing_env_vars

CUDA_RUNTIME_LIB: str = "libcudart.so" if os.name != "nt" else "cudart64_110.dll"


def extract_candidate_paths(paths_list_candidate: str) -> Set[Path]:
    return {Path(ld_path) for ld_path in paths_list_candidate.split(":") if ld_path}


def remove_non_existent_dirs(candidate_paths: Set[Path]) -> Set[Path]:
    existent_directories: Set[Path] = set()
    for path in candidate_paths:
        try:
            if path.exists():
                existent_directories.add(path)
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise exc

    non_existent_directories: Set[Path] = candidate_paths - existent_directories
    if non_existent_directories:
        warn(
            "WARNING: The following directories listed in your path were found to "
            f"be non-existent: {non_existent_directories}"
        )

    return existent_directories


def get_cuda_runtime_lib_paths(candidate_paths: Set[Path]) -> Set[Path]:
    return {
        path / CUDA_RUNTIME_LIB
        for path in candidate_paths
        if (path / CUDA_RUNTIME_LIB).is_file()
    }


def resolve_paths_list(paths_list_candidate: str) -> Set[Path]:
    """
    Searches a given environmental var for the CUDA runtime library,
    i.e. `libcudart.so`.
    """
    return remove_non_existent_dirs(extract_candidate_paths(paths_list_candidate))


def find_cuda_lib_in(paths_list_candidate: str) -> Set[Path]:
    return get_cuda_runtime_lib_paths(
        resolve_paths_list(paths_list_candidate)
    )


def warn_in_case_of_duplicates(results_paths: Set[Path]) -> None:
    if len(results_paths) > 1:
        warning_msg = (
            f"Found duplicate {CUDA_RUNTIME_LIB} files: {results_paths}.. "
            "We'll flip a coin and try one of these, in order to fail forward.\n"
            "Either way, this might cause trouble in the future:\n"
            "If you get `CUDA error: invalid device function` errors, the above "
            "might be the cause and the solution is to make sure only one "
            f"{CUDA_RUNTIME_LIB} in the paths that we search based on your env."
        )
        warn(warning_msg)


def determine_cuda_runtime_lib_path() -> Union[Path, None]:
    """
        Searches for a cuda installations, in the following order of priority:
            1. active conda env
            2. LD_LIBRARY_PATH
            3. any other env vars, while ignoring those that
                - are known to be unrelated (see `bnb.cuda_setup.env_vars.to_be_ignored`)
                - don't contain the path separator `/`

        If multiple libraries are found in part 3, we optimistically try one,
        while giving a warning message.
    """
    candidate_env_vars = get_potentially_lib_path_containing_env_vars()

    if "CONDA_PREFIX" in candidate_env_vars:
        conda_libs_path = Path(candidate_env_vars["CONDA_PREFIX"]) / "lib"

        conda_cuda_libs = find_cuda_lib_in(str(conda_libs_path))
        warn_in_case_of_duplicates(conda_cuda_libs)

        if conda_cuda_libs:
            return next(iter(conda_cuda_libs))

        warn(
            f'{candidate_env_vars["CONDA_PREFIX"]} did not contain '
            f'{CUDA_RUNTIME_LIB} as expected! Searching further paths...'
        )

    if "LD_LIBRARY_PATH" in candidate_env_vars:
        lib_ld_cuda_libs = find_cuda_lib_in(candidate_env_vars["LD_LIBRARY_PATH"])

        if lib_ld_cuda_libs:
            return next(iter(lib_ld_cuda_libs))
        warn_in_case_of_duplicates(lib_ld_cuda_libs)

        warn(
            f'{candidate_env_vars["LD_LIBRARY_PATH"]} did not contain '
            f'{CUDA_RUNTIME_LIB} as expected! Searching further paths...'
        )

    remaining_candidate_env_vars = {
        env_var: value for env_var, value in candidate_env_vars.items()
        if env_var not in {"CONDA_PREFIX", "LD_LIBRARY_PATH"}
    }

    cuda_runtime_libs = set()
    for env_var, value in remaining_candidate_env_vars.items():
        cuda_runtime_libs.update(find_cuda_lib_in(value))

    if len(cuda_runtime_libs) == 0:
        print(
            'CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching /usr/local/cuda/lib64...')
        cuda_runtime_libs.update(find_cuda_lib_in('/usr/local/cuda/lib64'))

    warn_in_case_of_duplicates(cuda_runtime_libs)

    return next(iter(cuda_runtime_libs)) if cuda_runtime_libs else None
