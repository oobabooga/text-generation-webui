"""Tests for the timestamp_parsing module."""

from datetime import datetime, timedelta

import pytest

from extensions.long_term_memory.utils.timestamp_parsing import (
    get_time_difference_message,
)


def test_get_time_difference_message():
    """Ensures get_time_difference_message works as expected."""

    now = datetime.utcnow()

    # The time between now and now is 0 days.
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    assert get_time_difference_message(timestamp) == "0 days ago"

    # Round down if we haven't passed a full day.
    one_day_ago = now - timedelta(days=0, hours=8)
    timestamp = one_day_ago.strftime("%Y-%m-%d %H:%M:%S")
    assert get_time_difference_message(timestamp) == "0 days ago"

    # Ensure we actually see multiple days.
    five_days_ago = now - timedelta(days=5, hours=16)
    timestamp = five_days_ago.strftime("%Y-%m-%d %H:%M:%S")
    assert get_time_difference_message(timestamp) == "5 days ago"

    # Ensure we raise on error.
    with pytest.raises(ValueError):
        get_time_difference_message("invalid timestamp")
