## Contributing guidelines

1) Please submit your changes to the `dev` branch of the repository.
2) Lint the changes that you have made. I use the following tools:

```
pyflakes file.py  # To check for errors
pycodestyle file.py  # To check for style problems
```

The following pycodestyle errors can be ignored:

* E402 module level import not at top of file
* E501 line too long
* E722 do not use bare 'except'

The remaining errors should be fixed.

3) Thoroughly self-test your code. Can you think of a scenario where it might not work as expected? Does it interfere with other parts of the program?

4) Keep it simple, structured, and organized. Don't add more lines than you have to.

### Scope

The main focus of this project is the Gradio UI, so you should familiarize yourself with Gradio: https://www.gradio.app/docs/interface

**API**

The UI takes precedence over the API: you should not add features exclusively to the API if they could be added to the UI first.

### Extensions

As a rule of thumb, new extensions should be submitted to https://github.com/oobabooga/text-generation-webui-extensions. **You are highly encouraged to submit your extensions to that list!**

New built-in extensions can be accepted in cases where they would be useful to a large percentage of the user base, preferably while adding few or no additional dependencies.

### Installation methods

There are two main installation methods for this project:

1) The one-click-installers.
2) Manual installation as described in the README.

Some Docker files are available in the repository, but I never use Docker. Pull requests about Docker should contain straightforward fixes or updates only.

### Some important variables

* `shared.settings` contains default values for Gradio components. It can be customized through a `settings.yaml` file.
* `shared.args` contains the command-line arguments. They represent variables that need to be changed often.
* `shared.gradio` contains the UI elements, like sliders and dropdowns. When defining Gradio event handlers, the `gradio` function in `modules.utils` can be used to write

```python
gradio('history', 'character_menu', 'mode')
```

instead of 

```python
[shared.gradio[k] for k in ['history', 'character_menu', 'mode']]
```

* The UI values are not passed directly to the generation functions. Instead, they are first fed into the `shared.gradio['interface_state']` state variable. This variable receives the name `state` when used as input to backend functions. The code for updating `shared.gradio['interface_state']` with the current UI values is the following (see `server.py` for several examples):

```python
ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')
```

* The chat history is represented as a dictionary with the following structure:

```python
{
    'internal': [['hi', 'hey'], ['how are you?', "i'm fine, thanks!"]], 
    'visible': [['hi', 'hey'], ['how are you?', "i'm fine, thanks!"]]
}

```

Each row is in the format `[input, reply]`. `history['visible']` contains the messages as they will appear in the UI, and `history['internal']` contains the messages as they appear in the prompt. When no extension is used, the two will be identical, but many extensions add images, audio widgets, or translations to `history['visible']`.