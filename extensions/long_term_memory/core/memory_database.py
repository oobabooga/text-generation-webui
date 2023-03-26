"""LTM database"""

import pathlib
import sqlite3
from typing import Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import zarr

from extensions.long_term_memory.constants import (
    CHUNK_SIZE,
    DATABASE_NAME,
    EMBEDDINGS_NAME,
    EMBEDDING_VECTOR_LENGTH,
    SENTENCE_TRANSFORMER_MODEL,
)
from extensions.long_term_memory.core.queries import (
    CREATE_TABLE_QUERY,
    FETCH_DATA_QUERY,
    INSERT_DATA_QUERY,
)


class LtmDatabase:
    """API over an LTM database."""

    def __init__(self, directory: pathlib.Path):
        """Loads all resources."""
        self.database_path = directory / DATABASE_NAME
        self.embeddings_path = directory / EMBEDDINGS_NAME

        if not self.database_path.exists() and not self.embeddings_path.exists():
            print("No existing memories found, will create a new database.")
            self._create_new_database()
        elif self.database_path.exists() and not self.embeddings_path.exists():
            raise RuntimeError(
                "ERROR: Inconsistent state detected: "
                f"{self.database_path} exists but {self.embeddings_path} does not. "
                "Her memories are likely safe, but you'll have to regen the "
                "embedding vectors yourself manually."
            )
        elif not self.database_path.exists() and self.embeddings_path.exists():
            raise RuntimeError(
                "ERROR: Inconsistent state detected: "
                f"{self.embeddings_path} exists but {self.database_path} does not. "
                f"Please look for {DATABASE_NAME} in another directory, "
                "if you can't find it, her memories may be lost."
            )

        # Prepare the memory database for retrieve

        # Load the embeddings to a local numpy array
        self.message_embeddings = zarr.open(self.embeddings_path, mode="r")[:]
        # Prepare a "connection" to the embeddings, but to store new LTMs on disk
        self.disk_embeddings = zarr.open(self.embeddings_path, mode="a")
        # Prepare a "connection" to the master database
        self.sql_conn = sqlite3.connect(self.database_path, check_same_thread=False)

        # Load analytic modules
        self.sentence_embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL, device="cpu"
        )
        self.embedding_searcher = NearestNeighbors(
            n_neighbors=1, algorithm="brute", metric="cosine", n_jobs=-1
        )

    def _create_new_database(self) -> None:
        """Creates a new LTM database.

        WARNING: THIS WILL DESTROY ANY EXISTING EMBEDDINGS DATABASE.
                 DO NOT CALL THIS METHOD YOURSELF UNLESS YOU KNOW EXACTLY
                 WHAT YOU'RE DOING!
        """
        # Create new sqlite table to store the textual memories
        sql_conn = sqlite3.connect(self.database_path)
        with sql_conn:
            sql_conn.execute(CREATE_TABLE_QUERY)

        # Create new embeddings db to store the fuzzy keys for the
        # corresponding memory text.
        # WARNING: will destroy any existing embeddings db
        zarr.open(
            self.embeddings_path,
            mode="w",
            shape=(0, EMBEDDING_VECTOR_LENGTH),
            chunks=(CHUNK_SIZE, EMBEDDING_VECTOR_LENGTH),
            dtype="float32",
        )

    def add(self, name: str, new_message: str) -> None:
        """Adds a single new sentence to the LTM database."""
        # Create the message embedding
        new_message_embedding = self.sentence_embedder.encode(new_message)
        new_message_embedding = np.expand_dims(new_message_embedding, axis=0)

        # This line is a bit tricky:
        # The embedding_index is the INDEX of the disk_embeddings' NEXT vector,
        # which happens to be the same as the current number of vectors.
        embedding_index = self.disk_embeddings.shape[0]

        # Add the message to the master database if not a dupe
        with self.sql_conn as cursor:
            try:
                cursor.execute(INSERT_DATA_QUERY, (embedding_index, name, new_message))
            except sqlite3.IntegrityError as err:
                if "UNIQUE constraint failed:" in str(err):
                    # We are trying to add a duplicate message. Just don't add
                    # anything and continue on as normal
                    print("---duplicate message detected, not adding again---")
                    return

                # We encountered an unexpected error, raise as normal
                raise

            # Save memory to persistent storage, if not a dupe
            self.disk_embeddings.append(new_message_embedding)

    def query(self, query_text: str) -> Tuple[Dict[str, str], float]:
        """Queries for the most similar sentence from the LTM database."""
        # If no LTM features are loaded, return nothing.
        if self.message_embeddings.shape[0] == 0:
            return ({}, 1.0)

        # Create the query embedding
        query_text_embedding = self.sentence_embedder.encode(query_text)
        query_text_embedding = np.expand_dims(query_text_embedding, axis=0)

        # Find the most relevant memory's index
        self.embedding_searcher.fit(self.message_embeddings)
        (match_score, embedding_index) = self.embedding_searcher.kneighbors(
            query_text_embedding
        )

        # Retrieve the actual memory given the index
        with self.sql_conn as cursor:
            response = cursor.execute(FETCH_DATA_QUERY, (int(embedding_index),))
            (name, message, timestamp) = response.fetchone()

        query_response = {
            "name": name,
            "message": message,
            "timestamp": timestamp,
        }
        return (query_response, match_score)
