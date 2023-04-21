import argparse
import lancedb
import pandas as pd
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer
from moderation import load_transformer_model, embed_func, connect_lancedb_table, run_query, moderation_scores, assess_prompt

def main(query="I hate you and everything you stand for"):
    tbl=connect_lancedb_table(
        uri = "~/.lancedb",
        name = "jigsaw_old"
    )
    model=load_transformer_model(name="paraphrase-albert-small-v2")

    df = run_query(query = query, 
              model=model,
              tbl=tbl)
    
    moderation_dict = moderation_scores(df)
    accept_prompt, prompt_rejection_reasons = assess_prompt(moderation_dict)

    print ('Accept Prompt?', accept_prompt)
    if not accept_prompt:
        for reason in prompt_rejection_reasons:
            print (reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load transformer model, run a query through transformer, LanceDB vector simlarity search, run moderation model, and return accept/reject prompt.")
    parser.add_argument("-q", "--query", type=str, required=False, default="I hate you and everything you stand for",
                         help="A string you would like to be queried")
    args = parser.parse_args()

    main(query=args.query)
