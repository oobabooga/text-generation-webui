Here you can restart the UI with new settings.

* **Available extensions**: shows a list of extensions available under `text-generation-webui/extensions`.
* **Boolean command-line flags**: shows command-line flags of bool (true/false) type.

After selecting your desired flags and extensions, you can restart the UI by clicking on **Apply flags/extensions and restart**.

## Install or update an extension

In this field, you can enter the GitHub URL for an extension and press enter to either install it (i.e. cloning it into `text-generation-webui/extensions`) or update it with `git pull` in case it is already cloned.

Note that some extensions may include additional Python requirements. In this case, to install those you have to run the command

```
pip install -r extensions/extension-name/requirements.txt
```

or

```
pip install -r extensions\extension-name\requirements.txt
```

if you are on Windows.

If you used the one-click installer, this command should be executed in the terminal window that appears when you run the "cmd_" script for your OS.

## Saving UI defaults

The **Save UI defaults to settings.yaml** button gathers the visible values in the UI and saves them to settings.yaml so that your settings will persist across multiple restarts of the UI.

Note that preset parameters like temperature are not individually saved, so you need to first save your preset and select it in the preset menu before saving the defaults.
