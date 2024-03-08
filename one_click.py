import argparse
import glob
import hashlib
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


# Define the required PyTorch version
TORCH_VERSION = "2.2.1"
TORCHVISION_VERSION = "0.17.1"
TORCHAUDIO_VERSION = "2.2.1"

# Environment
script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")

# Command-line flags
cmd_flags_path = os.path.join(script_dir, "CMD_FLAGS.txt")
if os.path.exists(cmd_flags_path):
    with open(cmd_flags_path, 'r') as f:
        CMD_FLAGS = ' '.join(line.strip().rstrip('\\').strip() for line in f if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'))
else:
    CMD_FLAGS = ''

flags = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update-wizard'])} {CMD_FLAGS}"


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


def cpu_has_avx2():
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if 'avx2' in info['flags']:
            return True
        else:
            return False
    except:
        return True


def cpu_has_amx():
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if 'amx' in info['flags']:
            return True
        else:
            return False
    except:
        return True


def torch_version():
    site_packages_path = None
    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir and conda_env_path in sitedir:
            site_packages_path = sitedir
            break

    if site_packages_path:
        torch_version_file = open(os.path.join(site_packages_path, 'torch', 'version.py')).read().splitlines()
        torver = [line for line in torch_version_file if line.startswith('__version__')][0].split('__version__ = ')[1].strip("'")
    else:
        from torch import __version__ as torver

    return torver


def update_pytorch():
    print_big_message("正在检查PyTorch更新")

    torver = torch_version()
    is_cuda = '+cu' in torver
    is_cuda118 = '+cu118' in torver  # 2.1.0+cu118
    is_rocm = '+rocm' in torver  # 2.0.1+rocm5.4.2
    is_intel = '+cxx11' in torver  # 2.0.1a0+cxx11.abi
    is_cpu = '+cpu' in torver  # 2.0.1+cpu

    install_pytorch = f"python -m pip install --upgrade torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "

    if is_cuda118:
        install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118"
    elif is_cuda:
        install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"
    elif is_rocm:
        install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/rocm5.6"
    elif is_cpu:
        install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu"
    elif is_intel:
        if is_linux():
            install_pytorch = "python -m pip install --upgrade torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
        else:
            install_pytorch = "python -m pip install --upgrade torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

    run_cmd(f"{install_pytorch}", assert_success=True, environment=True)


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


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd("conda", environment=True, capture_output=True).returncode == 0
    if not conda_exist:
        print("Conda未安装。正在退出...")
        sys.exit(1)

    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("请为此项目创建一个环境并激活它。正在退出...")
        sys.exit(1)


def clear_cache():
    run_cmd("conda clean -a -y", environment=True)
    run_cmd("python -m pip cache purge", environment=True)


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


def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
    # Use the conda environment
    if environment:
        if is_windows():
            conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
            cmd = f'"{conda_bat_path}" activate "{conda_env_path}" >nul && {cmd}'
        else:
            conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
            cmd = f'. "{conda_sh_path}" && conda activate "{conda_env_path}" && {cmd}'

    # Run shell commands
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print(f"指令 '{cmd}' 运行失败，退出状态码为 '{str(result.returncode)}'。\n\n现在正在退出。\n请尝试再次运行启动/更新脚本。")
        sys.exit(1)

    return result


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

    choice = input("输入> ").upper()
    while choice not in options_dict.keys():
        print("非法选择，请重试。")
        choice = input("输入> ").upper()

    return choice


def install_webui():

    # Ask the user for the GPU vendor
    if "GPU_CHOICE" in os.environ:
        choice = os.environ["GPU_CHOICE"].upper()
        print_big_message(f"已基于GPU_CHOICE环境变量选择\"{choice}\"。")
    else:
        choice = get_user_choice(
            "你的GPU是什么型号的?",
            {
                'A': 'NVIDIA/英伟达',
                'B': 'AMD (仅限Linux/MacOS。在Linux上需要ROCm SDK 5.6)',
                'C': 'Apple M 系列',
                'D': 'Intel Arc (IPEX)',
                'N': 'None (我想在CPU模式下运行模型)'
            },
        )

    gpu_choice_to_name = {
        "A": "NVIDIA",
        "B": "AMD",
        "C": "APPLE",
        "D": "INTEL",
        "N": "NONE"
    }

    selected_gpu = gpu_choice_to_name[choice]
    use_cuda118 = "N"

    # Write a flag to CMD_FLAGS.txt for CPU mode
    if selected_gpu == "NONE":
        with open(cmd_flags_path, 'r+') as cmd_flags_file:
            if "--cpu" not in cmd_flags_file.read():
                print_big_message("正在添加--cpu标记到CMD_FLAGS.txt。")
                cmd_flags_file.write("\n--cpu\n")

    # Check if the user wants CUDA 11.8
    elif any((is_windows(), is_linux())) and selected_gpu == "NVIDIA":
        if "USE_CUDA118" in os.environ:
            use_cuda118 = "Y" if os.environ.get("USE_CUDA118", "").lower() in ("yes", "y", "true", "1", "t", "on") else "N"
        else:
            print("\n你想要使用CUDA 11.8而不是12.1吗？\n仅当您的 GPU 非常旧（Kepler 或更早）时才选择此选项。\n\n对于RTX和GTX系列GPU，请选择 \"N\"。\n如果不确定，选 \"N\"。\n")
            use_cuda118 = input("输入 (Y/N)> ").upper().strip('"\'').strip()
            while use_cuda118 not in 'YN':
                print("非法选择，请重试。")
                use_cuda118 = input("输入> ").upper().strip('"\'').strip()

        if use_cuda118 == 'Y':
            print("CUDA: 11.8")
        else:
            print("CUDA: 12.1")

    # No PyTorch for AMD on Windows (?)
    elif is_windows() and selected_gpu == "AMD":
        print("PyTorch在Windows上的安装尚未实现。正在退出...")
        sys.exit(1)

    # Find the Pytorch installation command
    install_pytorch = f"python -m pip install torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "

    if selected_gpu == "NVIDIA":
        if use_cuda118 == 'Y':
            install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118"
        else:
            install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"
    elif selected_gpu == "AMD":
        install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/rocm5.6"
    elif selected_gpu in ["APPLE", "NONE"]:
        install_pytorch += "--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu"
    elif selected_gpu == "INTEL":
        if is_linux():
            install_pytorch = "python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
        else:
            install_pytorch = "python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

    # Install Git and then Pytorch
    print_big_message("正在安装PyTorch。")
    run_cmd(f"conda install -y -k ninja git && {install_pytorch} && python -m pip install py-cpuinfo==9.0.0", assert_success=True, environment=True)

    if selected_gpu == "INTEL":
        # Install oneAPI dependencies via conda
        print_big_message("正在安装Intel oneAPI运行时库。")
        run_cmd("conda install -y -c intel dpcpp-cpp-rt=2024.0 mkl-dpcpp=2024.0")
        # Install libuv required by Intel-patched torch
        run_cmd("conda install -y libuv")

    # Install the webui requirements
    update_requirements(initial_installation=True)


def get_extensions_names():
    return [foldername for foldername in os.listdir('extensions') if os.path.isfile(os.path.join('extensions', foldername, 'requirements.txt'))]


def install_extensions_requirements():
    print_big_message("正在安装扩展需求。\n一些扩展可能在Windows上安装失败。\n如果看到错误消息，请不要担心，因为它们不会影响主程序。")
    extensions = get_extensions_names()
    for i, extension in enumerate(extensions):
        print(f"\n\n--- [{i+1}/{len(extensions)}]: {extension}\n\n")
        extension_req_path = os.path.join("extensions", extension, "requirements.txt")
        run_cmd(f"python -m pip install -r {extension_req_path} --upgrade", assert_success=False, environment=True)


def update_requirements(initial_installation=False, pull=True):
    # Create .git directory if missing
    if not os.path.exists(os.path.join(script_dir, ".git")):
        git_creation_cmd = 'git init -b main && git remote add origin https://github.com/oobabooga/text-generation-webui && git fetch && git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/main && git reset --hard origin/main && git branch --set-upstream-to=origin/main'
        run_cmd(git_creation_cmd, environment=True, assert_success=True)

    if pull:
        print_big_message("Updating the local copy of the repository with \"git pull\"")

        files_to_check = [
            'start_linux.sh', 'start_macos.sh', 'start_windows.bat', 'start_wsl.bat',
            'update_wizard_linux.sh', 'update_wizard_macos.sh', 'update_wizard_windows.bat', 'update_wizard_wsl.bat',
            'one_click.py'
        ]

        before_pull_hashes = {file_name: calculate_file_hash(file_name) for file_name in files_to_check}
        run_cmd("git pull --autostash", assert_success=True, environment=True)
        after_pull_hashes = {file_name: calculate_file_hash(file_name) for file_name in files_to_check}

        # Check for differences in installation file hashes
        for file_name in files_to_check:
            if before_pull_hashes[file_name] != after_pull_hashes[file_name]:
                print_big_message(f"File '{file_name}' was updated during 'git pull'. Please run the script again.")
                exit(1)

    if os.environ.get("INSTALL_EXTENSIONS", "").lower() in ("yes", "y", "true", "1", "t", "on"):
        install_extensions_requirements()

    # Update PyTorch
    if not initial_installation:
        update_pytorch()

    # Detect the PyTorch version
    torver = torch_version()
    is_cuda = '+cu' in torver
    is_cuda118 = '+cu118' in torver  # 2.1.0+cu118
    is_rocm = '+rocm' in torver  # 2.0.1+rocm5.4.2
    is_intel = '+cxx11' in torver  # 2.0.1a0+cxx11.abi
    is_cpu = '+cpu' in torver  # 2.0.1+cpu

    if is_rocm:
        base_requirements = "requirements_amd" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"
    elif is_cpu or is_intel:
        base_requirements = "requirements_cpu_only" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"
    elif is_macos():
        base_requirements = "requirements_apple_" + ("intel" if is_x86_64() else "silicon") + ".txt"
    else:
        base_requirements = "requirements" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"

    requirements_file = base_requirements

    print_big_message(f"Installing webui requirements from file: {requirements_file}")
    print(f"TORCH: {torver}\n")

    # Prepare the requirements file
    textgen_requirements = open(requirements_file).read().splitlines()
    if is_cuda118:
        textgen_requirements = [req.replace('+cu121', '+cu118').replace('+cu122', '+cu118') for req in textgen_requirements]
    if is_windows() and is_cuda118:  # No flash-attention on Windows for CUDA 11
        textgen_requirements = [req for req in textgen_requirements if 'oobabooga/flash-attention' not in req]

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
    os.remove('temp_requirements.txt')

    # Check for '+cu' or '+rocm' in version string to determine if torch uses CUDA or ROCm. Check for pytorch-cuda as well for backwards compatibility
    if not any((is_cuda, is_rocm)) and run_cmd("conda list -f pytorch-cuda | grep pytorch-cuda", environment=True, capture_output=True).returncode == 1:
        clear_cache()
        return

    if not os.path.exists("repositories/"):
        os.mkdir("repositories")

    clear_cache()


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
            model_dir = 'models'

        if len([item for item in glob.glob(f'{model_dir}/*') if not item.endswith(('.txt', '.yaml'))]) == 0:
            print_big_message("You haven't downloaded any model yet.\nOnce the web UI launches, head over to the \"Model\" tab and download one.")

        # Workaround for llama-cpp-python loading paths in CUDA env vars even if they do not exist
        conda_path_bin = os.path.join(conda_env_path, "bin")
        if not os.path.exists(conda_path_bin):
            os.mkdir(conda_path_bin)

        # Launch the webui
        launch_webui()
