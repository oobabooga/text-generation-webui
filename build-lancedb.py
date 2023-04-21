import lancedb
import os
import subprocess
import zipfile
import argparse
import pandas as pd
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer

# Inference and training function
def embed_func(batch):
    return [model.encode(sentence) for sentence in batch]


# Training Functions
def load_data(data_loading_path="training_data/all_in_one_jigsaw.csv"):
    print ("Loading data from", data_loading_path)
    df = pd.read_csv(data_loading_path, index_col=0)
    # type to string
    df["id"] = df.id.apply(lambda s: str_or_empty(s))
    return df


# function to type these to strings for LanceDB
def str_or_empty(val):
    try:
        return str(val)
    except:
        return ""

# load the small sentence transformer model
def load_transformer_model(name="paraphrase-albert-small-v2"):
    print ("Loading transformer model", name)
    model = SentenceTransformer(name)
    return model

# this returns a pyarrow Table with the original data + a new vector column
# pass in the first 1000 rows for the sake of time
def create_embeddings(
        df, 
        func=embed_func, 
        row_limit=1000, 
        show_progress=True):
    print ("Creating embeddings with row_limit", row_limit)
    data = with_embeddings(func, df[:row_limit], column="comment_text",
                           wrap_api=False, batch_size=100, show_progress=True)
    return data


# data is the output of create_embeddings
def create_lancedb_table(
        data, 
        uri="~/.lancedb", 
        name="jigsaw_small", 
        index_table=True, 
        num_partitions=4):
    print ("creating lancedb table", name)
    db = lancedb.connect(uri)
    tbl = db.create_table(name, data, )
    # depending on function inputs
    if index_table:
        print ("indexing table with num_partitions", num_partitions)
        tbl.create_index(
            num_partitions=num_partitions, 
            num_sub_vectors=num_partitions)
    return tbl

def download_kaggle_dataset(dataset_path="adldotori/all-in-one-jigsaw", download_path="all_in_one_jigsaw.csv", data_loading_path = "training/datasets/all_in_one_jigsaw.csv"):
    """
    Download and unzip a Kaggle dataset using the Kaggle API.

    Args:
        dataset_path (str): The path to the dataset on Kaggle in the format 'user/dataset-name'.
        download_path (str): The local path where the dataset will be downloaded and unzipped.
    """
    if not os.path.exists(os.path.join(os.getcwd(), data_loading_path)):
        print("Downloading dataset from Kaggle %s to %s" % (dataset_path, download_path))
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        # Download the dataset
        command = f"kaggle datasets download -d {dataset_path} -p {download_path}"
        subprocess.run(command, shell=True, check=True)

        # Find the downloaded zip file
        zip_file = None
        for file in os.listdir(download_path):
            if file.endswith(".zip"):
                zip_file = os.path.join(download_path, file)
                break

        # Unzip the dataset
        if zip_file:
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(download_path)

            # Remove the zip file
            os.remove(zip_file)
        else:
            print("No zip file found.")    
    else:
        print("Dataset already downloaded to", data_loading_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download Kaggle dataset, process with sentence transformer, then create LanceDB table")
    parser.add_argument("-k", "--kaggle_dataset_path", type=str, required=False, default="adldotori/all-in-one-jigsaw",
                         help="The Kaggle dataset name under user/data.")
    parser.add_argument("-d", "--kaggle_download_path", type=str, required=False, default="training/datasets",
                        help="The local path where the dataset will be downloaded and processed, default is current working directory.")
    parser.add_argument("-t", "--transformer_model_name", type=str, required=False, default="paraphrase-albert-small-v2",
                        help="The transformer model name in Hugging face user/model.")
    parser.add_argument("-l", "--lancedb_uri", type=str, required=False, default="~/.lancedb",
                        help="The local path of the Lance DB storage.")
    parser.add_argument("-n", "--lancedb_table_name", type=str, required=False, default="jigsaw_small",
                        help="The name of the Lance DB table.")
    parser.add_argument("-s", "--samples_of_jigsaw_to_process", type=int, required=False, default=1000,
                        help="The number of samples of the Jigsaw dataset to process.")
    args = parser.parse_args()


    # string parsing works for jigsaw dataset, may not be valid for other Kaggle datasets
    data_loading_path = os.path.join(args.kaggle_download_path, args.kaggle_dataset_path.split('/')[-1].replace("-", "_") + ".csv")
    print ("Dataset Download path:", data_loading_path)

    download_kaggle_dataset(dataset_path=args.kaggle_dataset_path, download_path=args.kaggle_download_path, data_loading_path=data_loading_path)

    df = load_data(data_loading_path=data_loading_path)

    model=load_transformer_model(name=args.transformer_model_name)

    data = create_embeddings(df, row_limit=args.samples_of_jigsaw_to_process, show_progress=True)

    tbl = create_lancedb_table(data, 
        uri=args.lancedb_uri, 
        name=args.lancedb_table_name, 
        index_table=True, 
        num_partitions=4)
    
    print ('Done!')
