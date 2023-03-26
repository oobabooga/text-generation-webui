"""Tests for the chat_parsing module."""

from extensions.long_term_memory.utils.chat_parsing import (
    clean_character_message,
)


def test_clean_character_message():
    """Ensures clean_character_message works as expected."""

    # Single response
    expected_result = "Hai ^_^"
    assert expected_result == clean_character_message("Miku", "Miku: Hai ^_^")

    # Multiple responses
    expected_result = "Hai ^_^ Did you do your best today?"
    assert expected_result == clean_character_message(
        "Miku", "Miku: Hai ^_^ Miku: Did you do your best today?"
    )

    # No responses
    assert "" == clean_character_message("Miku", "Invalid message")

    # Empty input
    assert "" == clean_character_message("Miku", "")

    # Message with only bot name and no text
    assert "" == clean_character_message("Miku", "Miku: ")

    # Leading and trailing whitespaces
    expected_result = "iToddlers BTFO HAHAHAHA"
    assert expected_result == clean_character_message(
        "Satania", "\n   Satania: iToddlers    Satania: BTFO HAHAHAHA  \n "
    )

    # Bot message with special characters
    expected_result = "/think *he likes me!* (◕ω◕) yay!!11"
    assert expected_result == clean_character_message(
        "Miku", "Miku: /think *he likes me!* (◕ω◕) Miku: yay!!11"
    )
