"""Extension that allows us to fetch and store memories from/to LTM."""

import json
import pathlib
import pprint
from typing import List, Tuple

import gradio as gr

import modules.shared as shared
from modules.chat import generate_chat_prompt
from modules.html_generator import fix_newlines

from extensions.long_term_memory.core.memory_database import LtmDatabase
from extensions.long_term_memory.utils.chat_parsing import clean_character_message
from extensions.long_term_memory.utils.timestamp_parsing import (
    get_time_difference_message,
)


# === Internal constants (don't change these without good reason) ===
_CONFIG_PATH = "ltm_config.json"
_MIN_ROWS_TILL_RESPONSE = 5
_LAST_BOT_MESSAGE_INDEX = -3
_LTM_STATS_TEMPLATE = """{num_memories_seen_by_bot} memories are loaded in the bot
{num_memories_in_ram} memories are loaded in RAM
{num_memories_on_disk} memories are saved to disk"""
with open(_CONFIG_PATH, "rt") as handle:
    _CONFIG = json.load(handle)


# === Module-level variables ===
debug_texts = {
    "current_memory_text": "(None)",
    "num_memories_loaded": 0,
    "current_context_block": "(None)",
}
memory_database = LtmDatabase(
    pathlib.Path("./extensions/long_term_memory/user_data/bot_memories/"),
    num_memories_to_fetch=_CONFIG["ltm_reads"]["num_memories_to_fetch"],
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
    "would defeat the original purpose of this module. This can be overridden "
    "by pressing 'Force reload memories'"
)
print("----------")
print("LTM CONFIG")
print("----------")
print("change these values in ltm_config.json")
pprint.pprint(_CONFIG)
print("----------")
print("-----------------------------------------")


def _get_current_memory_text() -> str:
    return debug_texts["current_memory_text"]


def _get_num_memories_loaded() -> int:
    return debug_texts["num_memories_loaded"]


def _get_current_ltm_stats() -> str:
    ltm_stats = {
        "num_memories_seen_by_bot": _get_num_memories_loaded(),
        "num_memories_in_ram": memory_database.message_embeddings.shape[0],
        "num_memories_on_disk": memory_database.disk_embeddings.shape[0],
    }
    ltm_stats_str = _LTM_STATS_TEMPLATE.format(**ltm_stats)
    return ltm_stats_str


def _get_current_context_block() -> str:
    return debug_texts["current_context_block"]


def _build_augmented_context(memory_context: str, original_context: str) -> str:
    injection_location = _CONFIG["ltm_context"]["injection_location"]
    if injection_location == "BEFORE_NORMAL_CONTEXT":
        augmented_context = f"{memory_context.strip()}\n{original_context.strip()}"
    elif injection_location == "AFTER_NORMAL_CONTEXT_BUT_BEFORE_MESSAGES":
        if "<START>" not in original_context:
            raise ValueError(
                "Cannot use AFTER_NORMAL_CONTEXT_BUT_BEFORE_MESSAGES, "
                "<START> token not found in context. Please make sure you're "
                "using a proper character json and that you're NOT using the "
                "generic 'Assistant' sample character"
            )

        split_index = original_context.index("<START>")
        augmented_context = original_context[:split_index] + \
                memory_context.strip() + "\n" + original_context[split_index:]
    else:
        raise ValueError(f"Invalid injection_location: {injection_location}")

    return augmented_context


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
        with gr.Row():
            update = gr.Button("Force reload memories")
    with gr.Accordion(
        "Long Term Memory debug status (must manually refresh)", open=True
    ):
        with gr.Row():
            current_memory = gr.Textbox(
                value=_get_current_memory_text(),
                label="Current memory loaded by bot",
            )
            current_ltm_stats = gr.Textbox(
                value=_get_current_ltm_stats(),
                label="LTM statistics",
            )
        with gr.Row():
            current_context_block = gr.Textbox(
                value=_get_current_context_block(),
                label="Current FIXED context block (ONLY includes example convos)"
            )
        with gr.Row():
            refresh_debug = gr.Button("Refresh")
    with gr.Accordion("Long Term Memory DANGER ZONE", open=False):
        with gr.Row():
            destroy = gr.Button("Destroy all memories", variant="stop")
            destroy_confirm = gr.Button(
                "THIS IS IRREVERSIBLE, ARE YOU SURE?", variant="stop", visible=False
            )
            destroy_cancel = gr.Button("Do Not Delete", visible=False)
            destroy_elems = [destroy_confirm, destroy, destroy_cancel]

    # Update memories
    update.click(memory_database.reload_embeddings_from_disk, [], [])

    # Update debug info
    refresh_debug.click(fn=_get_current_memory_text, outputs=[current_memory])
    refresh_debug.click(fn=_get_current_ltm_stats, outputs=[current_ltm_stats])
    refresh_debug.click(fn=_get_current_context_block, outputs=[current_context_block])

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


def _build_memory_context(fetched_memories: List[Tuple[str, float]], name1: str, name2: str):
    memory_length_cutoff = _CONFIG["ltm_reads"]["memory_length_cutoff_in_chars"]

    # Build all the individual memory strings
    memory_strs = []
    distance_scores = []
    debug_texts["current_memory_text"] = "(None)"
    debug_texts["num_memories_loaded"] = 0
    for (fetched_memory, distance_score) in fetched_memories:
        if fetched_memory and distance_score < _CONFIG["ltm_reads"]["max_cosine_distance"]:
            time_difference = get_time_difference_message(fetched_memory["timestamp"])
            memory_str = _CONFIG["ltm_context"]["memory_template"].format(
                time_difference=time_difference,
                memory_name=fetched_memory["name"],
                memory_message=fetched_memory["message"][:memory_length_cutoff],
            )
            memory_strs.append(memory_str)
            distance_scores.append(distance_score)

    # No memories fetched, we'll have no memory_context
    if not memory_strs:
        return None

    # Now inject all memory strings into the wider memory context
    joined_memory_strs = "\n".join(memory_strs)
    memory_context = _CONFIG["ltm_context"]["memory_context_template"].format(
        name1=name1,
        name2=name2,
        all_memories=joined_memory_strs,
    )

    # Report debugging info to user
    print("------------------------------")
    print("NEW MEMORIES LOADED IN CHATBOT")
    pprint.pprint(joined_memory_strs)
    debug_texts["current_memory_text"] = joined_memory_strs
    debug_texts["num_memories_loaded"] = len(memory_strs)
    print("scores (in order)", distance_scores)
    print("------------------------------")
    return memory_context


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
    fetched_memories = memory_database.query(
        user_input,
    )
    memory_context = _build_memory_context(fetched_memories, name1, name2)

    # === Call oobabooga's original generate_chat_prompt ===
    augmented_context = context
    if memory_context is not None:
        augmented_context = _build_augmented_context(memory_context, context)
    debug_texts["current_context_block"] = augmented_context

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
        if len(clean_bot_message) >= _CONFIG["ltm_writes"]["min_message_length"]:
            memory_database.add(name2, clean_bot_message)
            print("-----------------------")
            print("NEW MEMORY SAVED to LTM")
            print("-----------------------")
            print("name:", name2)
            print("message:", clean_bot_message)
            print("-----------------------")

    # Store Anon's input directly into LTM
    if len(user_input) >= _CONFIG["ltm_writes"]["min_message_length"]:
        memory_database.add(name1, user_input)
        print("-----------------------")
        print("NEW MEMORY SAVED to LTM")
        print("-----------------------")
        print("name:", name1)
        print("message:", user_input)
        print("-----------------------")

    return prompt
