"""Tests the LTM database."""

import random
import string

import pytest

from extensions.long_term_memory.core.memory_database import (
    LtmDatabase,
)

from extensions.long_term_memory.constants import (
    DATABASE_NAME,
    EMBEDDINGS_NAME,
)


# Single query test data
MEMORY_LIST = [
    ("Anon", "a"),
    ("Anon", "hello"),
    ("Anon", "What are you doing with my computer RAM Miku, playing games?"),
    ("Miku", "You're cute!"),
    ("Miku", "Thank you for the apple pie! It tasted delicious!"),
    ("Miku", "EEEKKK! Why is there a spider on the window??!"),
    ("Miku", "The Discrete Fourier Transform"),
]

QUERY_MESSAGES = [
    "Thanks Anon, the spider is gone!",
    "The FFT is a fast algorithm of",
    "Do you remember the time I gave you my apple pie, did it taste okay?",
    "How much RAM do you need for a gaming computer?",
]
EXPECTED_MEMORY_INDICES = [5, 6, 4, 2]

# Multi-query test data
MEMORY_LIST_FOR_MULTI_FETCH = [
    ("Miku", "You shouldn't be fetching this message!"),
    ("Miku", "EEEKKK! Why is there a spider on the window??!"),
    ("Miku", "THE SPIDER IS STILL ON THE WINDOW DO SOMETHING ABOUT IT!"),
    ("Miku", "I actually like spiders!"),
]
QUERY_MESSAGE_FOR_MULTI_FETCH = "There's a spider on the window"
TEST_PARAMS_FOR_MULTI_FETCH = [
    {
        "num_memories_to_fetch": 2,
        "expected_indices": [1, 2],
    },
    {
        "num_memories_to_fetch": 3,
        "expected_indices": [1, 2, 3],
    }
]

# Additional testing params
NUM_RANDOM_MESSAGES = 50
RANDOM_MESSAGE_LENGTH = 300


def _get_single_response(ltm_database, actual_message):
    query_responses = ltm_database.query(actual_message)
    assert 1 == len(query_responses)
    return query_responses[0]

def _validate_memories(ltm_database):
    # Testing helper that ensures proper behavior when database contains
    # all the memories from MEMORY_LIST

    # Sanity check: verify we can find exact matches
    for actual_name, actual_message in MEMORY_LIST:
        (query_response, score) = _get_single_response(ltm_database, actual_message)
        assert query_response
        assert actual_name == query_response["name"]
        assert actual_message == query_response["message"]
        assert pytest.approx(0, abs=0.001) == score

    # Verify we can find similar messages in a fuzzy manner
    for query_text, memory_index in zip(QUERY_MESSAGES, EXPECTED_MEMORY_INDICES):
        (query_response, _) = _get_single_response(ltm_database, query_text)
        (actual_name, actual_message) = MEMORY_LIST[memory_index]
        assert actual_name == query_response["name"]
        assert actual_message == query_response["message"]


def _validate_database_integrity(tmp_path, num_expected_elems):
    # Testing helper that validates integrity of the database

    # Re-attach to the database to simulate a restart
    ltm_database = LtmDatabase(tmp_path)

    # Ensure we have the correct number of disk embeddings
    assert num_expected_elems == ltm_database.disk_embeddings.shape[0]

    # Ensure we have the correct number of text memories
    with ltm_database.sql_conn as cursor:
        (response,) = cursor.execute("SELECT COUNT(*) FROM long_term_memory").fetchone()
    assert num_expected_elems == response

    # Validate the indices of the text memories
    with ltm_database.sql_conn as cursor:
        response = cursor.execute(
            "SELECT id FROM long_term_memory ORDER BY timestamp ASC"
        ).fetchall()

    assert num_expected_elems == len(response)
    for expected_index, (actual_index,) in enumerate(response):
        assert expected_index == actual_index


def test_typical_usage(tmp_path):
    """Ensures LTM database operates as expected."""

    ### Mock user session 1 ###
    # Attach to the database (will create a new one)
    ltm_database = LtmDatabase(tmp_path)

    # Querying LTM should return an empty list
    # since we have no LTMs yet.
    query_response = ltm_database.query(QUERY_MESSAGES[0])
    assert not query_response

    # Add some memories
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    # Querying LTM should STILL return an empty list since no LTM is
    # actually queryable yet. Once the user restarts their session,
    # all LTMs will be queryable
    query_response = ltm_database.query(QUERY_MESSAGES[0])
    assert not query_response

    ### Mock user session 2 ###
    _validate_memories(LtmDatabase(tmp_path))

    ### Ensure integrity of the LTM database ###
    _validate_database_integrity(tmp_path, len(MEMORY_LIST))


def test_duplicate_messages(tmp_path):
    """Ensures we gracefully reject duplicate messages."""

    ### Mock user session 1 ###
    # Attach to the database (will create a new one)
    ltm_database = LtmDatabase(tmp_path)

    # Add some memories
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    # Add the same memories again
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    ### Mock user session 2 ###
    _validate_memories(LtmDatabase(tmp_path))

    ### Ensure integrity of the LTM database ###
    _validate_database_integrity(tmp_path, len(MEMORY_LIST))


def test_inconsistent_state(tmp_path):
    """Ensures we error out when the database is in an inconsistent state."""

    # Test when only the database file exists
    (tmp_path / DATABASE_NAME).touch()
    with pytest.raises(RuntimeError):
        LtmDatabase(tmp_path)

    # Test when only the embeddings file exists
    (tmp_path / DATABASE_NAME).unlink()
    (tmp_path / EMBEDDINGS_NAME).touch()
    with pytest.raises(RuntimeError):
        LtmDatabase(tmp_path)


def test_extended_usage(tmp_path):
    """Ensures system works in more difficult conditions."""

    ### Mock User Session 1 ###
    # Attach to the database (will create a new one)
    ltm_database = LtmDatabase(tmp_path)

    # Add some real memories
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    # Add a bunch of randomly-generated junk
    for _ in range(NUM_RANDOM_MESSAGES):
        message = "".join(
            random.choice(string.ascii_letters) for _ in range(RANDOM_MESSAGE_LENGTH)
        )
        ltm_database.add("RandomBot", message)

    # Try adding the same memories again
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    # Add more randomly-generated junk
    for _ in range(NUM_RANDOM_MESSAGES):
        message = "".join(
            random.choice(string.ascii_letters) for _ in range(RANDOM_MESSAGE_LENGTH)
        )
        ltm_database.add("RandomBot", message)

    ### Mock user session 2 ###
    _validate_memories(LtmDatabase(tmp_path))

    ### Ensure integrity of the LTM database ###
    num_expected_elems = 2 * NUM_RANDOM_MESSAGES + len(MEMORY_LIST)
    _validate_database_integrity(tmp_path, num_expected_elems)


def test_reload_embeddings_from_disk(tmp_path):
    """Ensures LTM database can reload embeddings from disk correctly."""

    ### Mock user session 1 ###
    # Attach to the database (will create a new one)
    ltm_database = LtmDatabase(tmp_path)

    # Add some memories
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    # Querying LTM should STILL return an empty list since no LTM is
    # actually queryable yet. Once the user restarts their session,
    # all LTMs will be queryable
    query_responses = ltm_database.query(QUERY_MESSAGES[0])
    assert not query_responses

    # Reload embeddings from disk, now all LTMs should be queryable
    ltm_database.reload_embeddings_from_disk()
    # NOTE: we reuse the original ltm_database object for this check
    _validate_memories(ltm_database)

    ### Ensure integrity of the LTM database ###
    _validate_database_integrity(tmp_path, len(MEMORY_LIST))


def test_destroy_fake_memories(tmp_path):
    """Ensures LTM database can destroy all (fake) memories.

    Your actual memories are safe, this test does not touch them.
    """

    ### Populating all memories ###
    # Attach to the database (will create a new one)
    ltm_database = LtmDatabase(tmp_path)

    # Destroy all memories on fresh database, shouldn't change anything
    ltm_database.destroy_all_memories()

    # Add some memories
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    # Reload embeddings from disk, now all LTMs should be queryable
    ltm_database.reload_embeddings_from_disk()
    _validate_memories(ltm_database)

    # Ensure integrity of the LTM database
    _validate_database_integrity(tmp_path, len(MEMORY_LIST))

    ### Destroying all memories ###
    # Destroy all memories on a populated database
    ltm_database.destroy_all_memories()

    # Validate all memories are actually destroyed
    # Vectors in-memory
    query_responses = ltm_database.query(QUERY_MESSAGES[0])
    assert not query_responses

    # Vectors on-disk
    _validate_database_integrity(tmp_path, 0)

    ### Populating all memories again ###
    # Attach to the database (will create a new one)
    ltm_database = LtmDatabase(tmp_path)

    # Add some memories
    for name, message in MEMORY_LIST:
        ltm_database.add(name, message)

    # Reload embeddings from disk, now all LTMs should be queryable
    ltm_database.reload_embeddings_from_disk()
    _validate_memories(ltm_database)

    # Ensure integrity of the LTM database
    _validate_database_integrity(tmp_path, len(MEMORY_LIST))


def test_multi_fetch(tmp_path):
    """Verify we can fetch multiple messages at once."""
    # Add all data
    ltm_database = LtmDatabase(tmp_path)
    for name, message in MEMORY_LIST_FOR_MULTI_FETCH:
        ltm_database.add(name, message)

    # Query to validate
    for test_params in TEST_PARAMS_FOR_MULTI_FETCH:
        expected_responses = [MEMORY_LIST_FOR_MULTI_FETCH[i][1] \
                for i in test_params["expected_indices"]]

        val_ltm_database = LtmDatabase(tmp_path, test_params["num_memories_to_fetch"])
        query_responses = val_ltm_database.query(QUERY_MESSAGE_FOR_MULTI_FETCH)
        assert test_params["num_memories_to_fetch"] == len(query_responses)

        for (query_response, _) in query_responses:
            assert query_response["message"] in expected_responses
