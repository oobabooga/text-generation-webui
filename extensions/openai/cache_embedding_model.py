#!/usr/bin/env python3
# preload the embedding model, useful for Docker images to prevent re-download on config change
# Dockerfile:
# ENV OPENEDAI_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"  # Optional
# RUN python3 cache_embedded_model.py
import os

import sentence_transformers

st_model = os.environ.get("OPENEDAI_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
model = sentence_transformers.SentenceTransformer(st_model)
