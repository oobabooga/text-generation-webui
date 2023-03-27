# Text Generation Web UI with Long-Term Memory

Welcome to the experimental repository for the Text Generation Web UI with a long-term memory (LTM) extension. The goal of the LTM extension is to enable the chatbot to "remember" conversations long-term. Please note that this is an early-stage experimental project, and perfect results should not be expected.

## How to Run
1. Follow the instructions in [oobabooga's  original repository](https://github.com/oobabooga/text-generation-webui) until you can chat with a chatbot.
2. Within the `textgen` conda environment (from the linked instructions), run the following commands to install dependencies and run tests:
```bash
pip install -r extensions/long_term_memory/requirements.txt
python -m pytest -v extensions/long_term_memory/
```
3. Run the server with the LTM extension. If all goes well, you should see it reporting "ok"
```bash
python server.py --chat --extensions long_term_memory
```
4. Chat normally with the chatbot and observe the console for LTM write/load status. Please note that LTM-stored memories will only be visible to the chatbot during your NEXT session. Additionally please use the same name for yourself across sessions, otherwise the chatbot may get confused when trying to understand memories (example: if you have used "anon" as your name in the past, don't use "Anon" in the future)
5. Memories will be saved in `extensions/long_term_memory/user_data/bot_memories/`. Back them up if you plan to mess with the code.

## Tips (credit to Anons from /g/'s /lmg/)
- If you're running on Windows, the LTM's extensions's dependencies may override the version of pytorch needed to run your LLMs. If this is the case, try reinstalling the original version of pytorch manually:
```bash
pip install torch-1.12.0+cu113 # or whichever version of pytorch was uninstalled
```

## Limitations
- This project has been tested on Ubuntu LTS 22.04. Compatibility with Windows or macOS is unknown.
- There's one universal LTM database, so it's recommended to stick with just one character. If you don't, all characters will see the memories of others.
- The system can only load one "memory" at any given time, and each memory sticks around for one message.
- Memories themselves are past raw conversations filtered solely on length, and some may be irrelevant or filler text.
- Limited scalability: Appending to the persistent LTM database is reasonably efficient, but we currently load all LTM embeddings in RAM, which consumes memory. Additionally, we perform a linear search across all embeddings during each chat round.

## How the Chatbot Sees the LTM
Chatbots are typically given a fixed, "context" text block that persists across the entire chat. The LTM extension augments this context block by dynamically injecting a relevant long-term memory.

### Example of a typical context block:
```markdown
The following is a conversation between Anon and Miku. Miku likes Anon but is very shy.
```

### Example of an augmented context block:
```markdown
Miku's memory log:
3 days ago, Miku said:
"So Anon, your favorite color is blue? That's really cool!"

During conversations between Anon and Miku, Miku will try to remember the memory described above and naturally integrate it with the conversation.
The following is a conversation between Anon and Miku. Miku likes Anon but is very shy.
```

## How It Works Behind the Scenes
### Database
- [zarr](https://zarr.readthedocs.io/en/stable/) is used to store embedding vectors on disk.
- [SQLite](https://www.sqlite.org/index.html) is used to store the actual memory text and additional attributes.
- [numpy](https://numpy.org/) is used to load the embedding vectors into RAM.

### Semantic Search
- Embeddings are generated using an SBERT model with the [SentenceTransformers](https://www.sbert.net/) library, specifically [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
- We use [scikit-learn](https://scikit-learn.org/) to perform a linear search against the loaded embedding vectors to find the single closest LTM given the user's input text.

## How You Can Help
- We need assistance with prompt engineering experimentation. How should we formulate the LTM injection?
- Test the system and try to break it, report any bugs you find.

## Roadmap
The roadmap will be driven based on user feedback. Potential updates include:

### New Features
- N-gram analysis for "higher quality memories".
- Scaling up memory bank size (with a limit of, perhaps, 4).

### Quality of Life Improvements
- Limit the size of each memory so it doesn't overwhelm the context.
- Other simple hacks to improve the end user-experience.

### Longer-Term (depending on interest/level of use)
- Integrate the system with [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Use a Large Language Model (LLM) to summarize raw text into more useful memories directly. This may be challenging just as an oobabooga extension.
- Scaling the backend.
