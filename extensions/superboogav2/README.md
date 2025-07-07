# SuperboogaV2

Enhance your LLM with additional information from text, URLs, and files for more accurate and context-aware responses.

---



## Installation and Activation

1. Start the conda environment by running `cmd_windows.bat` or the equivalent for your system in the root directory of `text-generation-webui`.
2. Install the necessary packages:
   ```
   pip install -r extensions/superboogav2/requirements.txt
   ```
3. Activate the extension in the `Session` tab of the web UI.
4. Click on `Apply flags/extensions and restart`. Optionally save the configuration by clicking on `Save UI defaults to settings.yaml`.

## Usage and Features

After activation, you can scroll further down in the chat UI to reveal the SuperboogaV2 interface. Here, you can add extra information to your chats through text input, multiple URLs, or by providing multiple files subject to the context window limit of your model.

The extra information and the current date and time are provided to the model as embeddings that persist across conversations. To clear them, click the `Clear Data` button and start a new chat. You can adjust the text extraction parameters and other options in the `Settings`.

## Supported File Formats

SuperboogaV2 utilizes MuPDF, pandas, python-docx, and python-pptx to extract text from various file formats, including:

- TXT
- PDF
- EPUB
- HTML
- CSV
- ODT/ODS/ODP
- DOCX/PPTX/XLSX

## Additional Information

SuperboogaV2 processes your data into context-aware chunks, applies cleaning techniques, and stores them as embeddings to minimize redundant computations. Relevance is determined using distance calculations and prioritization of recent information.

For a detailed description and more information, refer to the comments in this pull request: [https://github.com/oobabooga/text-generation-webui/pull/3272](https://github.com/oobabooga/text-generation-webui/pull/3272)
