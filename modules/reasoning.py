import html as html_module

# Thinking block format definitions: (start_tag, end_tag, content_start_tag)
# Use None for start_tag to match from beginning (end-only formats should be listed last)
THINKING_FORMATS = [
    ('<think>', '</think>', None),
    ('<|channel|>analysis<|message|>', '<|end|>', '<|start|>assistant<|channel|>final<|message|>'),
    ('<seed:think>', '</seed:think>', None),
    ('<|think|>', '<|end|>', '<|content|>'),  # Solar Open
    ('Thinking Process:', '</think>', None),  # Qwen3.5 verbose thinking outside tags
    (None, '</think>', None),  # End-only variant (e.g., Qwen3-next)
]


def extract_reasoning(text, html_escaped=False):
    """Extract reasoning/thinking blocks from the beginning of a string.

    When html_escaped=True, tags are HTML-escaped before searching
    (for use on already-escaped UI strings).

    Returns (reasoning_content, final_content) where reasoning_content is
    None if no thinking block is found.
    """
    if not text:
        return None, text

    esc = html_module.escape if html_escaped else lambda s: s

    for start_tag, end_tag, content_tag in THINKING_FORMATS:
        end_esc = esc(end_tag)
        content_esc = esc(content_tag) if content_tag else None

        if start_tag is None:
            # End-only format: require end tag, start from beginning
            end_pos = text.find(end_esc)
            if end_pos == -1:
                continue
            thought_start = 0
        else:
            # Normal format: require start tag
            start_esc = esc(start_tag)
            start_pos = text.find(start_esc)
            if start_pos == -1:
                continue
            thought_start = start_pos + len(start_esc)
            end_pos = text.find(end_esc, thought_start)

        if end_pos == -1:
            # End tag missing - check if content tag can serve as fallback
            if content_esc:
                content_pos = text.find(content_esc, thought_start)
                if content_pos != -1:
                    thought_end = content_pos
                    content_start = content_pos + len(content_esc)
                else:
                    thought_end = len(text)
                    content_start = len(text)
            else:
                thought_end = len(text)
                content_start = len(text)
        else:
            thought_end = end_pos
            if content_esc:
                content_pos = text.find(content_esc, end_pos)
                content_start = content_pos + len(content_esc) if content_pos != -1 else end_pos + len(end_esc)
            else:
                content_start = end_pos + len(end_esc)

        return text[thought_start:thought_end], text[content_start:]

    return None, text
