#!/usr/bin/env python3
# preload the embedding model, useful for Docker images to prevent re-download on config change
# Dockerfile:
# ENV OPENEDAI_EMBEDDING_MODEL=all-mpnet-base-v2 # Optional
# RUN python3 cache_embedded_model.py
import os, sentence_transformers
st_model = os.environ["OPENEDAI_EMBEDDING_MODEL"] if "OPENEDAI_EMBEDDING_MODEL" in os.environ else "all-mpnet-base-v2"
model = sentence_transformers.SentenceTransformer(st_model)
