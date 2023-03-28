"""Sqlite queries."""

# NOTE: we shouldn't be attaching a semantic meaning to pk, fix this later
CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS long_term_memory(
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        message TEXT NOT NULL UNIQUE,
        timestamp TEXT NOT NULL
    )
    """

DROP_TABLE_QUERY = "DROP TABLE IF EXISTS long_term_memory"

INSERT_DATA_QUERY = """
    INSERT INTO long_term_memory (id, name, message, timestamp)
    VALUES(?, ?, ?, CURRENT_TIMESTAMP)
    """

FETCH_DATA_QUERY = """
    SELECT name, message, timestamp FROM long_term_memory
    WHERE id = ?
    """
