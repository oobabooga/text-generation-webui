The one-click-installers, previously downloaded as .zip archives, are now part of the repository ([#4028](https://github.com/oobabooga/text-generation-webui/pull/4028)).

Before this change, the installers used to not be automatically updated, causing people to run outdated versions that were much more likely to break after a simple update.

## Migrating from an old one-click install

The process is very simple, and you will keep all your models and settings.

1) Run your existing `update` script, and make sure that the following files exist inside `text-generation-webui` after the update: `start_windows.bat`, `start_linux.sh`.

2) Move your `text-generation-webui` folder one folder up. For instance, mine was at `Desktop\oobabooga_windows\text-generation-webui`:

![Untitled](https://github.com/oobabooga/text-generation-webui/assets/112222186/b8d4576f-75d2-459c-b4c6-54381061b54e)

So I moved it to `Desktop`:

![Untitled-2](https://github.com/oobabooga/text-generation-webui/assets/112222186/cde6fe04-3909-43c2-9cdf-cb6580312d6d)

If yours is at `oobabooga_windows\oobabooga_windows\text-generation-webui`, move it two folders up.

3) Enter the `text-generation-webui` folder and run the correct `start_` script for your OS. This will create a new `installer_files` folder inside `text-generation-webui` with the project requirements.

![Untitled-3](https://github.com/oobabooga/text-generation-webui/assets/112222186/23dd4d58-40ed-4616-8a49-2969912323e2)

4) Delete your previous `oobabooga_windows` folder (or similar), as it is no longer necessary.

Make sure that you have completed steps 1 to 3 and that `text-generation-webui` is no longer inside this folder before deleting it. You may want to backup your `CMD_FLAGS.txt` file.

![Untitled-4](https://github.com/oobabooga/text-generation-webui/assets/112222186/58506d07-5497-47fa-ba84-869aea9d550e)

5) Migration completed!