This web UI supports extensions. They can be used to modify your prompt or the model's output.

## Creating an extension

In order to create an extension, create a folder under `extensions` and put a file named `script.py` in that folder. For instance, `extensions/my_extension`. 

## Extension functions

The `script.py` file may contain three functions:

#### input_modifier

Internally changes your input string before it enters the model.

#### output_modifier

Modifies the model output before it is presented in the web UI.

#### bot_prefix_modifier

In chat mode, modifies the prefix for a new bot message. For instance, if your bot is named `Marie Antoinette`, the default prefix for a new message will be

```
Marie Antoinette:
```

Using `bot_prefix_modifier`, you can change it to:

```
Marie Antoinette: *I am very enthusiastic*
```
 
Marie Antoinette will become very enthusiastic in all her messages.

## Extension parameters

The `script.py` file may also contain a `params` dict in the format

```
params = {
    'description1': value1,
    'description1': value2,
    ...
}
```

where the values must be numbers, strings, or booleans. These web UI will create fields for you to change these parameters interactively.

## Activating an extension

In order to use your extension, start the web UI with the `--extensions` flag. For instance, if your extension was created in the `extensions/enthusiasm` folder, use

`python server.py --extensions enthusiasm`

You can activate more than one extension at a time. In this case, they will be applied in the order that you specify:

```
python server.py --extensions enthusiasm,pretty # First apply enthusiasm, then pretty
python server.py --extensions pretty,enthusiasm # First apply pretty, then enthusiasm
```

## Examples

|Extension|Description|
|---------|-----------|
|[google_translate](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/google_translate/script.py)| Automatically translates inputs and outputs using Google Translate|
|[example](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/example/script.py)| Just a very simple example|
