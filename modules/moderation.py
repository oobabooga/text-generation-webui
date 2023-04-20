import lancedb
import pandas as pd
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer


# used for both training and querying
def embed_func(batch, model):
    return [model.encode(sentence) for sentence in batch]


# this returns a pyarrow Table with the original data + a new vector column
# pass in the first 1000 rows for the sake of time
def create_embeddings(df, func=embed_func):
    data = with_embeddings(func, df, column="comment_text",
                           wrap_api=False, batch_size=1)
    return data


# data is the output of create_embeddings
def create_lancedb_table(data, uri="~/.lancedb", name="jigsaw"):
    db = lancedb.connect(uri)
    tbl = db.create_table(name, data)
    return tbl

def connect_lancedb_table(uri="~/.lancedb", name="jigsaw"):
    db = lancedb.connect(uri)
    tbl = db.name    
    return tbl



# tbl is a LanceDB Table instance
def run_query(tbl, query, topk=5):
    emb = embed_func([query])[0]
    return tbl.search(emb).limit(topk).to_df()
