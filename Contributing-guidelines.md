# Contributing guidelines

Your help with improving text-generation-webui is welcome and appreciated. Here you can find some general guidelines to make the process easier.

## Requirements

Before submitting a Pull Request, make sure to:

1) Familiarize yourself with Gradio, in particular, its components and methods. The documentation is very easy to follow: https://gradio.app/docs
2) Lint the changes that you have made. I use the following tools:

```
pyflakes file.py  # To check for errors
pycodestyle file.py  # To check for style problems
```

The following pycodestyle errors can be ignored:

* E501 line too long
* E402 module level import not at top of file
* E722 do not use bare 'except'

The remaining errors should be fixed.

3) Thoroughly self-test your code. Can you think of a scenario where it might not work as expected? Does it interfere with other parts of the program?

4) Keep it simple, structured, and organized.

## Scope

This project aims to provide a web interface for interacting with Large Language Models. As such, improvements to the UI are of high priority, including:

* Improving the various UI functionalities.
* Improving the arrangement of UI elements.
* Improving the CSS styles under `text-generation-webui/css`, in particular, those of chat styles.

**API**

The API exists to make it possible to automate text generation actions available in the UI.

* You should not add features to the API if they could be added to the UI first.
* Providing an API with extensive and niche features is not a priority.

## Extensions

As a rule of thumb, extensions should be submitted to https://github.com/oobabooga/text-generation-webui-extensions. Exceptions can be made for extensions that do not add any dependencies, or extensions that would be useful to a large percentage of the user base.

## Installation methods

There are two main installation methods for this project:

1) Manual installation as described in the README.
2) The one-click-installers available at https://github.com/oobabooga/one-click-installers.

Some Docker files are available in the repository, but I do not use Docker. Pull requests about Docker should contain straightforward fixes or updates only.

## Some important variables

* The `shared.settings` variable contains default values for Gradio components. It can be customized through a `settings.yaml` file.
* The `shared.args` variable contains the command-line arguments. They represent variables that need to be changed often.
* The chat history is represented as a dictionary with the following structure:

```python
{
    'internal': [['hi', 'hey'], ['how are you?', "i'm fine, thanks!"]], 
    'visible': [['hi', 'hey'], ['how are you?', "i'm fine, thanks!"]]
}

```

Each row is in the format `[input, reply]`. `history['visible']` contains the messages as they will appear in the UI, and `history['internal']` contains the messages as they appear in the prompt. When no extension is used, the two will be identical, but many extensions add images, audio widgets, or translations to `history['visible']`.