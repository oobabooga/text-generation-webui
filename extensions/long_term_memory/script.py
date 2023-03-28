"""Extension that allows us to fetch and store memories from/to LTM."""

import pathlib
import pprint

import gradio as gr

import modules.shared as shared
from modules.chat import generate_chat_prompt
from modules.html_generator import fix_newlines

from extensions.long_term_memory.core.memory_database import LtmDatabase
from extensions.long_term_memory.utils.chat_parsing import clean_character_message
from extensions.long_term_memory.utils.timestamp_parsing import (
    get_time_difference_message,
)


# === Constants (feel free to play around with these) ===
# The LTM sub-context to be injected into the bot's fixed context.
FETCHED_MEMORY_TEMPLATE = """
{name2}'s memory log:
{time_difference}, {memory_name} said:
"{memory_message}"

During conversations between {name1} and {name2}, {name2} will try to remember the memory described above and naturally integrate it with the conversation.
"""

# How long a message must be for it to be considered for LTM storage.
# Lower this value to allow "shorter" memories to get recorded by LTM.
MINIMUM_MESSAGE_LENGTH_FOR_LTM = 100

# Controls how "similar" your last message has to be to some LTM message to
# be loaded into the context. It represents the cosine distance, where "lower"
# means "more similar".
# Lower this value to increase the strictness of the check.
MEMORY_SCORE_THRESHOLD = 0.60


# === Internal constants (don't change these without good reason) ===
_MIN_ROWS_TILL_RESPONSE = 5
_LAST_BOT_MESSAGE_INDEX = -3


# === Module-level variables ===
memory_database = LtmDatabase(
    pathlib.Path("./extensions/long_term_memory/user_data/bot_memories/")
)
# This bias string is currently unused, feel free to try using it
params = {
    "activate": False,
    "bias string": " *I got a new memory! I'll try bringing it up in conversation!*",
}


# === Display important notes to the user ===
print()
print("-----------------------------------------")
print("IMPORTANT LONG TERM MEMORY NOTES TO USER:")
print("-----------------------------------------")
print(
    "Please remember that LTM-stored memories will only be visible to "
    "the bot during your NEXT session. This prevents the loaded memory "
    "from being flooded with messages from the current conversation which "
    "would defeat the original purpose of this module."
)
print(
    f"Messages shorter than {MINIMUM_MESSAGE_LENGTH_FOR_LTM} chars will NOT "
    "be stored. This can be adjusted in extensions/long_term_memory/script.py"
)
print("-----------------------------------------")


# === Hooks to oobaboogs UI ===
def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """
    if params["activate"]:
        bias_string = params["bias string"].strip()
        return f"{string} {bias_string} "
    return string


def ui():
    """Adds the LTM-specific settings."""
    with gr.Accordion("Long Term Memory settings", open=True):
        update = gr.Button("Force reload memories")
        with gr.Row():
            destroy = gr.Button("Destroy all memories")
            destroy_confirm = gr.Button(
                "THIS IS IRREVERSIBLE, ARE YOU SURE?", variant="stop", visible=False
            )
            destroy_cancel = gr.Button("Cancel", visible=False)
            destroy_elems = [destroy_confirm, destroy, destroy_cancel]

    # Update memories
    update.click(memory_database.reload_embeddings_from_disk, [], [])

    # Clear memory with confirmation
    destroy.click(
        lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)],
        None,
        destroy_elems,
    )
    destroy_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None,
        destroy_elems,
    )
    destroy_confirm.click(memory_database.destroy_all_memories, [], [])
    destroy_cancel.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None,
        destroy_elems,
    )


def custom_generate_chat_prompt(
    user_input,
    max_new_tokens,
    name1,
    name2,
    context,
    chat_prompt_size,
    impersonate=False,
):
    """Main hook that allows us to fetch and store memories from/to LTM."""
    print("=" * 60)

    user_input = fix_newlines(user_input)

    # === Fetch the "best" memory from LTM, if there is one ===
    (fetched_memory, distance_score) = memory_database.query(user_input)
    memory_context = None
    if fetched_memory and distance_score < MEMORY_SCORE_THRESHOLD:
        time_difference = get_time_difference_message(fetched_memory["timestamp"])
        memory_context = FETCHED_MEMORY_TEMPLATE.format(
            name1=name1,
            name2=name2,
            time_difference=time_difference,
            memory_name=fetched_memory["name"],
            memory_message=fetched_memory["message"],
        )
        print("----------------------------")
        print("NEW MEMORY LOADED IN CHATBOT")
        pprint.pprint(fetched_memory)
        print("score", distance_score)
        print("----------------------------")

    # === Call oobabooga's original generate_chat_prompt ===
    augmented_context = (
        f"{memory_context.strip()} {context.strip()}\n"
        if memory_context is not None
        else context
    )

    (prompt, prompt_rows) = generate_chat_prompt(
        user_input,
        max_new_tokens,
        name1,
        name2,
        augmented_context,
        chat_prompt_size,
        impersonate,
        also_return_rows=True,
    )

    # === Clean and add new messages to LTM ===
    # Store the bot's last message.
    # Avoid storing any of the baked-in bot template responses
    if len(prompt_rows) >= _MIN_ROWS_TILL_RESPONSE:
        bot_message = prompt_rows[_LAST_BOT_MESSAGE_INDEX]
        clean_bot_message = clean_character_message(name2, bot_message)

        # Store bot message into LTM
        if len(clean_bot_message) >= MINIMUM_MESSAGE_LENGTH_FOR_LTM:
            memory_database.add(name2, clean_bot_message)
            print("-----------------------")
            print("NEW MEMORY SAVED to LTM")
            print("-----------------------")
            print("name:", name2)
            print("message:", clean_bot_message)
            print("-----------------------")

    # Store Anon's input directly into LTM
    if len(user_input) >= MINIMUM_MESSAGE_LENGTH_FOR_LTM:
        memory_database.add(name1, user_input)
        print("-----------------------")
        print("NEW MEMORY SAVED to LTM")
        print("-----------------------")
        print("name:", name1)
        print("message:", user_input)
        print("-----------------------")

    return prompt
