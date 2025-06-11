import argparse
import glob
import hashlib
import json
import os
import platform
import re
import signal
import site
import subprocess
import sys

# Remove the '# ' from the following lines as needed for your AMD GPU on Linux
# os.environ["ROCM_PATH"] = '/opt/rocm'
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = '10.3.0'
# os.environ["HCC_AMDGPU_TARGET"] = 'gfx1030'

# Define the required versions
TORCH_VERSION = "2.6.0"
PYTHON_VERSION = "3.11"
LIBSTDCXX_VERSION_LINUX = "12.1.0"

# Environment
script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")
state_file = '.installer_state.json'

# Command-line flags
flags = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update-wizard'])}"


def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def is_linux():
    return sys.platform.startswith("linux")


def is_windows():
    return sys.platform.startswith("win")


def is_macos():
    return sys.platform.startswith("darwin")


def is_x86_64():
    return platform.machine() == "x86_64"


def is_installed():
    site_packages_path = None
    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir and conda_env_path in sitedir:
            site_packages_path = sitedir
            break

    if site_packages_path:
        return os.path.isfile(os.path.join(site_packages_path, 'torch', '__init__.py'))
    else:
        return os.path.isdir(conda_env_path)


def cpu_has_avx2():
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return 'avx2' in info['flags']
    except:
        return True


def cpu_has_amx():
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return 'amx' in info['flags']
    except:
        return True


def load_state():
    """Load installer state from JSON file"""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_state(state):
    """Save installer state to JSON file"""
    with open(state_file, 'w') as f:
        json.dump(state, f)


def get_gpu_choice():
    """Get GPU choice from state file or ask user"""
    state = load_state()
    gpu_choice = state.get('gpu_choice')

    if not gpu_choice:
        if "GPU_CHOICE" in os.environ:
            choice = os.environ["GPU_CHOICE"].upper()
            print_big_message(f"Selected GPU choice \"{choice}\" based on the GPU_CHOICE environment variable.")
        else:
            choice = get_user_choice(
                "What is your GPU?",
                {
                    'A': 'NVIDIA - CUDA 12.4',
                    'B': 'AMD - Linux/macOS only, requires ROCm 6.2.4',
                    'C': 'Apple M Series',
                    'D': 'Intel Arc (beta)',
                    'E': 'NVIDIA - CUDA 12.8',
                    'N': 'CPU mode'
                },
            )

        # Convert choice to GPU name
        gpu_choice = {"A": "NVIDIA", "B": "AMD", "C": "APPLE", "D": "INTEL", "E": "NVIDIA_CUDA128", "N": "NONE"}[choice]

        # Save choice to state
        state['gpu_choice'] = gpu_choice
        save_state(state)

    return gpu_choice


def get_pytorch_install_command(gpu_choice):
    """Get PyTorch installation command based on GPU choice"""
    base_cmd = f"python -m pip install torch=={TORCH_VERSION} "

    if gpu_choice == "NVIDIA":
        return base_cmd + "--index-url https://download.pytorch.org/whl/cu124"
    elif gpu_choice == "NVIDIA_CUDA128":
        return "python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128"
    elif gpu_choice == "AMD":
        return base_cmd + "--index-url https://download.pytorch.org/whl/rocm6.2.4"
    elif gpu_choice in ["APPLE", "NONE"]:
        return base_cmd + "--index-url https://download.pytorch.org/whl/cpu"
    elif gpu_choice == "INTEL":
        if is_linux():
            return "python -m pip install torch==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
        else:
            return "python -m pip install torch==2.1.0a0 intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    else:
        return base_cmd


def get_pytorch_update_command(gpu_choice):
    """Get PyTorch update command based on GPU choice"""
    base_cmd = f"python -m pip install --upgrade torch=={TORCH_VERSION} "

    if gpu_choice == "NVIDIA":
        return f"{base_cmd} --index-url https://download.pytorch.org/whl/cu124"
    elif gpu_choice == "NVIDIA_CUDA128":
        return "python -m pip install --upgrade torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128"
    elif gpu_choice == "AMD":
        return f"{base_cmd} --index-url https://download.pytorch.org/whl/rocm6.2.4"
    elif gpu_choice in ["APPLE", "NONE"]:
        return f"{base_cmd} --index-url https://download.pytorch.org/whl/cpu"
    elif gpu_choice == "INTEL":
        intel_extension = "intel-extension-for-pytorch==2.1.10+xpu" if is_linux() else "intel-extension-for-pytorch==2.1.10"
        return f"{base_cmd} {intel_extension} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    else:
        return base_cmd


def get_requirements_file(gpu_choice):
    """Get requirements file path based on GPU choice"""
    requirements_base = os.path.join("requirements", "full")

    if gpu_choice == "AMD":
        file_name = f"requirements_amd{'_noavx2' if not cpu_has_avx2() else ''}.txt"
    elif gpu_choice == "APPLE":
        file_name = f"requirements_apple_{'intel' if is_x86_64() else 'silicon'}.txt"
    elif gpu_choice in ["INTEL", "NONE"]:
        file_name = f"requirements_cpu_only{'_noavx2' if not cpu_has_avx2() else ''}.txt"
    elif gpu_choice == "NVIDIA":
        file_name = f"requirements{'_noavx2' if not cpu_has_avx2() else ''}.txt"
    elif gpu_choice == "NVIDIA_CUDA128":
        file_name = f"requirements_cuda128{'_noavx2' if not cpu_has_avx2() else ''}.txt"
    else:
        raise ValueError(f"Unknown GPU choice: {gpu_choice}")

    return os.path.join(requirements_base, file_name)


def get_current_commit():
    result = run_cmd("git rev-parse HEAD", capture_output=True, environment=True)
    return result.stdout.decode('utf-8').strip()


def get_extensions_names():
    return [foldername for foldername in os.listdir('extensions') if os.path.isfile(os.path.join('extensions', foldername, 'requirements.txt'))]


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd("conda", environment=True, capture_output=True).returncode == 0
    if not conda_exist:
        print("Conda is not installed. Exiting...")
        sys.exit(1)

    # Ensure this is a new environment and not the base environment
    if os.environ.get("CONDA_DEFAULT_ENV", "") == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit(1)


def clear_cache():
    run_cmd("conda clean -a -y", environment=True)
    run_cmd("python -m pip cache purge", environment=True)


def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
    # Use the conda environment
    if environment:
        if is_windows():
            conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
            cmd = f'"{conda_bat_path}" activate "{conda_env_path}" >nul && {cmd}'
        else:
            conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
            cmd = f'. "{conda_sh_path}" && conda activate "{conda_env_path}" && {cmd}'

    # Set executable to None for Windows, bash for everything else
    executable = None if is_windows() else 'bash'

    # Run shell commands
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env, executable=executable)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print(f"Command '{cmd}' failed with exit status code '{str(result.returncode)}'.\n\nExiting now.\nTry running the start/update script again.")
        sys.exit(1)

    return result


def print_big_message(message):
    message = message.strip()
    lines = message.split('\n')
    print("\n\n*******************************************************************")
    for line in lines:
        print("*", line)

    print("*******************************************************************\n\n")


def calculate_file_hash(file_path):
    p = os.path.join(script_dir, file_path)
    if os.path.isfile(p):
        with open(p, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    else:
        return ''


def generate_alphabetic_sequence(index):
    result = ''
    while index >= 0:
        index, remainder = divmod(index, 26)
        result = chr(ord('A') + remainder) + result
        index -= 1

    return result


def get_user_choice(question, options_dict):
    print()
    print(question)
    print()

    for key, value in options_dict.items():
        print(f"{key}) {value}")

    print()

    choice = input("Input> ").upper()
    while choice not in options_dict.keys():
        print("Invalid choice. Please try again.")
        choice = input("Input> ").upper()

    return choice


def update_pytorch_and_python():
    print_big_message("Checking for PyTorch updates.")
    gpu_choice = get_gpu_choice()
    install_cmd = get_pytorch_update_command(gpu_choice)
    run_cmd(install_cmd, assert_success=True, environment=True)


def clean_outdated_pytorch_cuda_dependencies():
    patterns = ["cu121", "cu122", "torch2.4", "torchvision", "torchaudio"]
    result = run_cmd("python -m pip list --format=freeze", capture_output=True, environment=True)
    matching_packages = []

    for line in result.stdout.decode('utf-8').splitlines():
        if "==" in line:
            pkg_name, version = line.split('==', 1)
            if any(pattern in version for pattern in patterns):
                matching_packages.append(pkg_name)

    if matching_packages:
        print(f"\nUninstalling: {', '.join(matching_packages)}\n")
        run_cmd(f"python -m pip uninstall -y {' '.join(matching_packages)}", assert_success=True, environment=True)

    return matching_packages


def install_webui():
    if os.path.isfile(state_file):
        os.remove(state_file)

    # Get GPU choice and save it to state
    gpu_choice = get_gpu_choice()

    # Write a flag to CMD_FLAGS.txt for CPU mode
    if gpu_choice == "NONE":
        cmd_flags_path = os.path.join(script_dir, "user_data", "CMD_FLAGS.txt")
        with open(cmd_flags_path, 'r+') as cmd_flags_file:
            if "--cpu" not in cmd_flags_file.read():
                print_big_message("Adding the --cpu flag to user_data/CMD_FLAGS.txt.")
                cmd_flags_file.write("\n--cpu\n")

    # Handle CUDA version display
    elif any((is_windows(), is_linux())) and gpu_choice == "NVIDIA":
        print("CUDA: 12.4")
    elif any((is_windows(), is_linux())) and gpu_choice == "NVIDIA_CUDA128":
        print("CUDA: 12.8")

    # No PyTorch for AMD on Windows (?)
    elif is_windows() and gpu_choice == "AMD":
        print("PyTorch setup on Windows is not implemented yet. Exiting...")
        sys.exit(1)

    # Install Git and then Pytorch
    print_big_message("Installing PyTorch.")
    install_pytorch = get_pytorch_install_command(gpu_choice)
    run_cmd(f"conda install -y ninja git && {install_pytorch} && python -m pip install py-cpuinfo==9.0.0", assert_success=True, environment=True)

    if gpu_choice == "INTEL":
        # Install oneAPI dependencies via conda
        print_big_message("Installing Intel oneAPI runtime libraries.")
        run_cmd("conda install -y -c https://software.repos.intel.com/python/conda/ -c conda-forge dpcpp-cpp-rt=2024.0 mkl-dpcpp=2024.0", environment=True)
        # Install libuv required by Intel-patched torch
        run_cmd("conda install -y libuv", environment=True)

    # Install the webui requirements
    update_requirements(initial_installation=True, pull=False)


def update_requirements(initial_installation=False, pull=True):
    # Create .git directory if missing
    if not os.path.exists(os.path.join(script_dir, ".git")):
        run_cmd(
            "git init -b main && git remote add origin https://github.com/oobabooga/text-generation-webui && "
            "git fetch && git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/main && "
            "git reset --hard origin/main && git branch --set-upstream-to=origin/main",
            environment=True,
            assert_success=True
        )

    current_commit = get_current_commit()
    wheels_changed = not os.path.exists(state_file)
    if not wheels_changed:
        state = load_state()
        if 'wheels_changed' in state or state.get('last_installed_commit') != current_commit:
            wheels_changed = True

    gpu_choice = get_gpu_choice()
    requirements_file = get_requirements_file(gpu_choice)

    if pull:
        # Read .whl lines before pulling
        before_pull_whl_lines = []
        if os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                before_pull_whl_lines = [line for line in f if '.whl' in line]

        print_big_message('Updating the local copy of the repository with "git pull"')

        # Hash files before pulling
        files_to_check = [
            'start_linux.sh', 'start_macos.sh', 'start_windows.bat', 'start_wsl.bat',
            'update_wizard_linux.sh', 'update_wizard_macos.sh', 'update_wizard_windows.bat', 'update_wizard_wsl.bat',
            'one_click.py'
        ]
        before_hashes = {file: calculate_file_hash(file) for file in files_to_check}

        # Perform the git pull
        run_cmd("git pull --autostash", assert_success=True, environment=True)

        # Check hashes after pulling
        after_hashes = {file: calculate_file_hash(file) for file in files_to_check}
        if os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                after_pull_whl_lines = [line for line in f if '.whl' in line]

        wheels_changed = wheels_changed or (before_pull_whl_lines != after_pull_whl_lines)

        # Check for changes to installer files
        for file in files_to_check:
            if before_hashes[file] != after_hashes[file]:
                print_big_message(f"File '{file}' was updated during 'git pull'. Please run the script again.")

                # Save state before exiting
                state = load_state()
                if wheels_changed:
                    state['wheels_changed'] = True
                save_state(state)
                sys.exit(1)

    # Save current state
    state = load_state()
    state['last_installed_commit'] = current_commit
    state.pop('wheels_changed', None)  # Remove wheels_changed flag
    save_state(state)

    if os.environ.get("INSTALL_EXTENSIONS", "").lower() in ("yes", "y", "true", "1", "t", "on"):
        install_extensions_requirements()

    if is_linux():
        run_cmd(f"conda install -y -c conda-forge libstdcxx-ng=={LIBSTDCXX_VERSION_LINUX}", assert_success=True, environment=True)

    # Update PyTorch
    if not initial_installation:
        update_pytorch_and_python()
        clean_outdated_pytorch_cuda_dependencies()

    print_big_message(f"Installing webui requirements from file: {requirements_file}")
    print(f"GPU Choice: {gpu_choice}\n")

    # Prepare the requirements file
    textgen_requirements = open(requirements_file).read().splitlines()

    if not initial_installation and not wheels_changed:
        textgen_requirements = [line for line in textgen_requirements if '.whl' not in line]

    with open('temp_requirements.txt', 'w') as file:
        file.write('\n'.join(textgen_requirements))

    # Workaround for git+ packages not updating properly.
    git_requirements = [req for req in textgen_requirements if req.startswith("git+")]
    for req in git_requirements:
        url = req.replace("git+", "")
        package_name = url.split("/")[-1].split("@")[0].rstrip(".git")
        run_cmd(f"python -m pip uninstall -y {package_name}", environment=True)
        print(f"Uninstalled {package_name}")

    # Install/update the project requirements
    run_cmd("python -m pip install -r temp_requirements.txt --upgrade", assert_success=True, environment=True)

    # Clean up
    os.remove('temp_requirements.txt')
    clear_cache()


def install_extensions_requirements():
    print_big_message("Installing extensions requirements.\nSome of these may fail on Windows.\nDon\'t worry if you see error messages, as they will not affect the main program.")
    extensions = get_extensions_names()
    for i, extension in enumerate(extensions):
        print(f"\n\n--- [{i + 1}/{len(extensions)}]: {extension}\n\n")
        extension_req_path = os.path.join("extensions", extension, "requirements.txt")
        run_cmd(f"python -m pip install -r {extension_req_path} --upgrade", assert_success=False, environment=True)


def launch_webui():
    run_cmd(f"python server.py {flags}", environment=True)


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--update-wizard', action='store_true', help='Launch a menu with update options.')
    args, _ = parser.parse_known_args()

    if args.update_wizard:
        while True:
            choice = get_user_choice(
                "What would you like to do?",
                {
                    'A': 'Update the web UI',
                    'B': 'Install/update extensions requirements',
                    'C': 'Revert local changes to repository files with \"git reset --hard\"',
                    'N': 'Nothing (exit)'
                },
            )

            if choice == 'A':
                update_requirements()
            elif choice == 'B':
                choices = {'A': 'All extensions'}
                for i, name in enumerate(get_extensions_names()):
                    key = generate_alphabetic_sequence(i + 1)
                    choices[key] = name

                choice = get_user_choice("What extension?", choices)

                if choice == 'A':
                    install_extensions_requirements()
                else:
                    extension_req_path = os.path.join("extensions", choices[choice], "requirements.txt")
                    run_cmd(f"python -m pip install -r {extension_req_path} --upgrade", assert_success=False, environment=True)

                update_requirements(pull=False)
            elif choice == 'C':
                run_cmd("git reset --hard", assert_success=True, environment=True)
            elif choice == 'N':
                sys.exit()
    else:
        if not is_installed():
            install_webui()
            os.chdir(script_dir)

        if os.environ.get("LAUNCH_AFTER_INSTALL", "").lower() in ("no", "n", "false", "0", "f", "off"):
            print_big_message("Will now exit due to LAUNCH_AFTER_INSTALL.")
            sys.exit()

        # Check if a model has been downloaded yet
        if '--model-dir' in flags:
            # Splits on ' ' or '=' while maintaining spaces within quotes
            flags_list = re.split(' +(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)|=', flags)
            model_dir = [flags_list[(flags_list.index(flag) + 1)] for flag in flags_list if flag == '--model-dir'][0].strip('"\'')
        else:
            model_dir = 'user_data/models'

        if len([item for item in glob.glob(f'{model_dir}/*') if not item.endswith(('.txt', '.yaml'))]) == 0:
            print_big_message("You haven't downloaded any model yet.\nOnce the web UI launches, head over to the \"Model\" tab and download one.")

        # Workaround for llama-cpp-python loading paths in CUDA env vars even if they do not exist
        conda_path_bin = os.path.join(conda_env_path, "bin")
        if not os.path.exists(conda_path_bin):
            os.mkdir(conda_path_bin)

        # Launch the webui
        launch_webui()
