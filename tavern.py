import concurrent.futures
from dotenv import load_dotenv
import os


def run_program(program):
    os.system(program)


if __name__ == "__main__":
    env_path = os.path.expanduser('~/.config/text-generation-webui/.env')
    load_dotenv(dotenv_path=env_path)
    dir1 = str(os.getenv("text_generation_webui_path"))
    venv_path = str(os.getenv("venv_path"))
    dir2 = str(os.getenv("silly_tavern_path"))
    model_path = str(os.getenv("model_path"))
    model = str(os.getenv("model_name"))
    loader = str(os.getenv("loader"))
    msl = int(os.getenv("max_seq_len", 2048))
    host = str(os.getenv("host", "localhost"))
    port = int(os.getenv("port", 7860))

    programs = [
        f"cd {dir1} && \
          {venv_path}/bin/python3 {dir1}/server.py --api \
              --model-dir={model_path} \
                --model={model} \
                  --loader={loader} \
                    --max_seq_len={msl} \
                      --compress_pos_emb={msl//2048} \
                        --listen \
                          --listen-host={host} \
                            --listen-port={port}",
        f"cd {dir2} && node server.js",
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_program, programs)
