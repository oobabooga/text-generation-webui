# From https://github.com/d8ahazard/sd_dreambooth_extension/blob/926ae204ef5de17efca2059c334b6098492a0641/postinstall.py

import filecmp
import os
import shutil
import sysconfig


def check_bitsandbytes():
    """
    Check for "different" B&B Files and copy only if necessary
    """
    try:
        bnb_src = os.path.dirname(os.path.realpath(__file__))
        bnb_dest = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")
        filecmp.clear_cache()
        for file in os.listdir(bnb_src):
            src_file = os.path.join(bnb_src, file)
            if file == "main.py" or file == "paths.py":
                dest = os.path.join(bnb_dest, "cuda_setup")
            else:
                dest = bnb_dest
            shutil.copy2(src_file, dest)
    except:
        pass


def setup_bitsandbytes_windows():
    if os.name != "nt":
        return
    check_bitsandbytes()

    import bitsandbytes_windows.main as new_main
    import bitsandbytes.cuda_setup.main as main

    # Main
    main.check_cuda_result = new_main.check_cuda_result
    main.get_cuda_version = new_main.get_cuda_version
    main.get_cuda_lib_handle = new_main.get_cuda_lib_handle
    main.get_compute_capabilities = new_main.get_compute_capabilities
    main.get_compute_capability = new_main.get_compute_capability
    main.evaluate_cuda_setup = new_main.evaluate_cuda_setup

    import bitsandbytes_windows.paths as new_paths
    import bitsandbytes.cuda_setup.paths as paths

    # Paths
    paths.extract_candidate_paths = new_paths.extract_candidate_paths
    paths.remove_non_existent_dirs = new_paths.remove_non_existent_dirs
    paths.get_cuda_runtime_lib_paths = new_paths.get_cuda_runtime_lib_paths
    paths.resolve_paths_list = new_paths.resolve_paths_list
    paths.find_cuda_lib_in = new_paths.find_cuda_lib_in
    paths.warn_in_case_of_duplicates = new_paths.warn_in_case_of_duplicates
    paths.determine_cuda_runtime_lib_path = new_paths.determine_cuda_runtime_lib_path

    import bitsandbytes_windows.cextension as new_cext
    import bitsandbytes.cextension as cext

    # CExtension
    cext.CUDALibrary_Singleton = new_cext.CUDALibrary_Singleton

    import bitsandbytes.nn as nn
    import bitsandbytes.nn.modules as new_nn

    # nn (This is needed, stuff isn't properly set in __init__.py)
    nn.Int8Params = new_nn.Int8Params
    nn.Linear8bitLt = new_nn.Linear8bitLt
    nn.StableEmbedding = new_nn.StableEmbedding

    import bitsandbytes as bnb
    import bitsandbytes.autograd._functions as new_bnb

    import bitsandbytes.optim as optim

    # bitsandbytes stuff
    bnb.MatmulLtState = new_bnb.MatmulLtState
    bnb.bmm_cublas = new_bnb.bmm_cublas
    bnb.matmul = new_bnb.matmul
    bnb.matmul_cublas = new_bnb.matmul_cublas
    bnb.mm_cublas = new_bnb.mm_cublas
    if cext.COMPILED_WITH_CUDA:
        bnb.adam = optim.adam



