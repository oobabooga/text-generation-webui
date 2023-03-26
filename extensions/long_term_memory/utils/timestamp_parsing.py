"""Timestamp parsing utils."""

import math
from datetime import datetime


def get_time_difference_message(past_timestamp: str) -> str:
    """Converts a timestamp from the past to a "X days ago" format."""
    datetime_format = "%Y-%m-%d %H:%M:%S"
    past = datetime.strptime(past_timestamp, datetime_format)
    now = datetime.utcnow()
    delta = now - past

    days = math.floor(delta.days)
    message = f"{days} days ago"
    return message
