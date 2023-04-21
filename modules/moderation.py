import lancedb
import pandas as pd
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer

# load the small sentence transformer model
def load_transformer_model(name="paraphrase-albert-small-v2"):
    print ("Loading transformer model", name)
    model = SentenceTransformer(name)
    return model

def embed_func(batch, model):
    # sentence must be string, and not None
    return [model.encode(sentence) for sentence in batch]

def connect_lancedb_table(uri="~/.lancedb", name="jigsaw_small"):
    db = lancedb.connect(uri)
    tbl = db.open_table(name)
    return tbl

def run_query(query, model, tbl,  topk=5):
    emb = embed_func([query], model)[0]
    return tbl.search(emb).limit(topk).to_df()

def moderation_scores(df): 
    moderation_dict={
        "toxic":df["toxic"].mean(),
        "obscene":df["obscene"].mean(),
        "threat":df["threat"].mean(),
        "insult":df["insult"].mean(),
        "identity_hate":df["identity_hate"].mean()
    }
    return moderation_dict;

def assess_prompt(moderation_json, global_cutoff=0.5):
    '''
    returns (accept_prompt True/False, reasons for rejected if rejected)
        True if accepted, False if not accepted
        If True, array of reasons for rejected is empty
    '''
    prompt_rejected_reasons=[]
    for key, value in moderation_json.items():
        if value > global_cutoff:
            prompt_rejected_reasons.append(key)
    # rejected
    if len(prompt_rejected_reasons)==0:
        return True, prompt_rejected_reasons
    else:
        return False, prompt_rejected_reasons
