from modules import launch_utils
import argparse
import json
import sys
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the webui")
    parser.add_argument("--check", action="store_ture")
    parser.add_argument("--no-history", action="store_true")
    args = parser.parse_args()
    
    python = launch_utils.python

    args_history = []
    if not os.path.exists(".history.json"):
        with open(".history.json", "w") as f:
            json.dump([], f)
    args_history = json.load(open(".history.json", "r"))
    
    launch_utils.check_environment()

    if args.check:
        launch_utils.prepare_packages()
        exit()

    if args.no_history or len(args_history) == 0:
        print("Performing zero arguments")
        os.system(f"{python} server.py")
    
    elif len(sys.argv) == 1:
        print("Performing history arguments")
        os.system(f"{python} server.py {' '.join(args_history)}")
    
    elif len(sys.argv) > 1:
        print("Performing arguments: " + str(sys.argv[1:]))
        args_history = sys.argv[1:]
        json.dump(args_history, open(".history.json", "w"))
        os.system(f"{python} server.py {' '.join(sys.argv[1:])}")