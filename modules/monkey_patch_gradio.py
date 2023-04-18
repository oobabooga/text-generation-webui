import importlib
import pathlib


file_path = pathlib.Path(importlib.util.find_spec("gradio").submodule_search_locations[0]) / "strings.py"

with open(file_path, 'r') as f:
    contents = f.read()

new_contents = contents.replace('threading.Thread(target=get_updated_messaging, args=(en,)).start()', '')
if contents != new_contents:
    print('Patching gradio to prevent it from calling home every time it gets imported...')
    try:
        # This is dirty but harmless. 
        with open(file_path, 'w') as f:
            f.write(new_contents)
    except:
        pass
