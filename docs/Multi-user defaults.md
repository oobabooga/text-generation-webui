# Default settings in multi-user mode

`--multi-user` can use the same YAML interface defaults as a single-user server.
Use this to choose the initial chat mode, character, and enabled tools for new
browser sessions. Users can still change these choices in their own session.

## Configure the defaults

Create a file named `team-defaults.yaml` in the TextGen directory:

```yaml
mode: chat-instruct
character: Assistant
selected_tools:
  - calculate
  - get_datetime
```

Start TextGen with your usual launcher and add these flags:

```text
--multi-user --settings team-defaults.yaml
```

For a manual Python installation, run from the TextGen directory:

```sh
python server.py --multi-user --settings team-defaults.yaml
```

For a one-click installation, add the flags to `CMD_FLAGS.txt`. An absolute path
to the YAML file also works. Restart the server after editing the file, then
open a new browser session to check the defaults.

Alternatively, put the configuration in `settings.yaml` inside your active user
data directory. TextGen loads that file automatically when no existing file is
selected with `--settings`. A custom settings file is loaded instead of the
user-data `settings.yaml`, not merged with it; unspecified keys retain built-in
defaults. Copy any other defaults you need into the custom file.

## Choose the values

- `mode` accepts `instruct`, `chat-instruct`, or `chat`.
- `character` is the character filename without its extension, such as
  `Assistant` for `characters/Assistant.yaml` in the user data directory. Character
  settings apply in `chat` and `chat-instruct` modes.
- `selected_tools` is a YAML list of tool names shown in the Tools section. The
  example selects two bundled tools. Use `selected_tools: []` to start with none.
  Tools are available in `instruct` and `chat-instruct` modes; `chat` mode hides
  the tool controls. Selecting a tool does not add tool-calling support to a model
  that does not support it.

The YAML keys are `character` and `selected_tools`; they are not the internal UI
name `character_menu` or new `--character` / `--tools-enable` command-line flags.

## Session behavior

Multi-user mode starts fresh chat histories instead of loading saved histories,
and it disables automatic saving of interface settings. Changes made by one
session do not replace these server defaults for subsequent sessions. It does
not provide persistent per-account preferences: configure initial defaults on
the server and let users adjust their own sessions.

As described in the command-line help, this mode is intended for small trusted
teams. These settings choose initial UI values, not access permissions for tools
or models.
