import html as html_module

# Thinking block format definitions: (start_tag, end_tag, content_start_tag)
# Use None for start_tag to match from beginning (end-only formats should be listed last)
THINKING_FORMATS = [
    ('<think>', '</think>', None),
    ('<|channel|>analysis<|message|>', '<|end|>', '<|channel|>final<|message|>'),
    ('<|channel|>commentary<|message|>', '<|end|>', '<|channel|>final<|message|>'),
    ('<seed:think>', '</seed:think>', None),
    ('<|channel>thought', '<channel|>', None),  # Gemma 4
    ('<|think|>', '<|end|>', '<|content|>'),  # Solar Open
    # ('Thinking Process:', '</think>', None),  # Qwen3.5 verbose thinking outside tags -- removed: too prone to false positives in streaming
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
                # During streaming, the start tag may be arriving partially.
                # If the text is a prefix of a start tag, return empty content
                # to prevent the partial tag from leaking.
                stripped = text.strip()
                if stripped and start_esc.startswith(stripped):
                    return '', ''
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
                if content_pos != -1:
                    content_start = content_pos + len(content_esc)
                else:
                    # Content tag not present yet.  In GPT-OSS the region
                    # between <|end|> and the content tag contains internal
                    # markup (<|start|>assistant…) that must not be shown.
                    # Suppress it to prevent tag leaks during streaming.
                    remainder = text[end_pos + len(end_esc):].lstrip()
                    framing_token = esc('<|start|>')
                    if not remainder or remainder.startswith(framing_token) or framing_token.startswith(remainder):
                        content_start = len(text)
                    else:
                        content_start = end_pos + len(end_esc)
            else:
                content_start = end_pos + len(end_esc)

        return text[thought_start:thought_end], text[content_start:].lstrip()

    # Handle standalone GPT-OSS final channel marker without a preceding
    # analysis/commentary block (the model skipped thinking entirely).
    for marker in ['<|start|>assistant<|channel|>final<|message|>', '<|channel|>final<|message|>']:
        marker_esc = esc(marker)
        pos = text.find(marker_esc)
        if pos != -1:
            before = text[:pos].strip()
            after = text[pos + len(marker_esc):]
            return (before if before else None), after

    return None, text
