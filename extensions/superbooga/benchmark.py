"""
This module implements a benchmark function to evaluate the performance of the embedding pipeline. It expects a configuration JSON file. It must have questions and expected retrieved text.
For each question, it's essential to have variants of that question. Language is fluid and each person might have their own spin on how they may ask it.

At the end, it will save the results inside a benchmark_{sysdate}.txt file in the main directory.

The benchmark function will return the score as an integer.
"""
import datetime
import json
import os

from pathlib import Path

from .data_processor import process_and_add_to_collector, preprocess_text
from .parameters import get_chunk_count, get_max_token_count
from .utils import create_metadata_source

def benchmark(config_path, collector):
    # Get the current system date
    sysdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{sysdate}.txt"
    
    # Open the log file in append mode
    with open(filename, 'a') as log:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        total_points = 0
        max_points = 0

        for item in data:
            filepath = item["text"]
            corpus = ""

            # Check if the file exists
            if os.path.isfile(Path(filepath)):
                # Open the file and read its content
                with open(Path(filepath), 'r') as file:
                    corpus = file.read()
                process_and_add_to_collector(corpus, collector, True, create_metadata_source('benchmark'))
            else:
                raise f'Cannot find specified file {filepath}.'

            for question_group in item["questions"]:
                question_variants = question_group["question_variants"]
                criteria = question_group["criteria"]
                
                for q in question_variants:
                    max_points += len(criteria)
                    processed_text = preprocess_text(q)

                    # Get the most similar chunks
                    results = collector.get_sorted_by_dist(processed_text, n_results=get_chunk_count(), max_token_count=get_max_token_count())

                    points = 0
                    
                    for c in criteria:
                        for p in results:
                            if c in p:
                                points += 1
                                total_points += 1
                                break

                    info = f"The question '{q}' scored {points}/{len(criteria)} points."
                    print(info, file=log)

                print('\n---\n', file=log)

        print(f'##Total points:\n\n{total_points}/{max_points}', file=log)

    return total_points, max_points