import datetime
import json
import math
import os

from .data_processor import process_and_add_to_collector, preprocess_text

def benchmark(config_path, n_results, max_token_count, chunk_len, context_len, chunk_regex, chunk_sep, collector):
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
            if os.path.isfile(filepath):
                # Open the file and read its content
                with open(filepath, 'r') as file:
                    corpus = file.read()
                for i in process_and_add_to_collector(corpus, chunk_len, context_len, chunk_regex, chunk_sep, collector):
                    yield i
            else:
                raise f'Cannot find specified file {filepath}.'

            for question_group in item["questions"]:
                question_variants = question_group["question_variants"]
                criteria = question_group["criteria"]
                
                for q in question_variants:
                    max_points += len(criteria)
                    processed_text = preprocess_text(q)

                    # Get the most similar chunks
                    results = collector.get_sorted_by_dist(processed_text, n_results=n_results, max_token_count=int(max_token_count))

                    points = 0
                    
                    for c in criteria:
                        for p in results:
                            if c in p:
                                points += 1
                                total_points += 1
                                break

                    info = f"The question '{q}' scored {points} points."
                    print(info, file=log)

                    yield info
                print(file=log)

    yield f'Total points: {total_points}/{max_points}.'

    return total_points