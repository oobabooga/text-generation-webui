"""Utils that parse chat logs."""


def clean_character_message(name: str, message: str) -> str:
    """
    Sometimes the chatbot will respond multiple times in a single
    message, each response being prefixed with '{bot_name}: '.
    This function parses each sub-message and returns them as a single
    continuous sentence.
    """
    name_header = f"{name}: "

    # The character isn't saying anything, return an empty list
    if name_header not in message:
        return ""

    # The character may be saying something, parse and return all messages
    split_message = message.split(name_header)
    messages = [line.strip() for line in split_message]
    messages = [line for line in messages if line]
    clean_message = " ".join(messages).strip()

    return clean_message
