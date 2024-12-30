Here you can restart the UI with new settings.

* **Available extensions**: shows a list of extensions available under `text-generation-webui/extensions`.
* **Boolean command-line flags**: shows command-line flags of bool (true/false) type.

After selecting your desired flags and extensions, you can restart the UI by clicking on **Apply flags/extensions/localization and restart**.

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

## Localization

You can get a localization file from community, put the json file under localization folder, and then click on **Apply flags/extensions/localization and restart** button to apply the localization you have selected.  

If there's no localization in your language, you can make your own! Just press the **Download localization template** button, change the values, and save the file in the localization folder. Then you can select your localization from the dropdown menu. Remember to rename localization.json to your language code (e.g. en.json, de.json, etc.) so others can know which language it is.