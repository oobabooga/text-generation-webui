"""
This module is responsible for processing the corpus and feeding it into chromaDB. It will receive a corpus of text. 
It will then split it into chunks of specified length. For each of those chunks, it will append surrounding context.
It will only include full words.
"""

import re
import bisect

import extensions.superboogav2.parameters as parameters

from .data_preprocessor import TextPreprocessorBuilder, TextSummarizer
from .chromadb import ChromaCollector

def preprocess_text_no_summary(text) -> str:
    builder = TextPreprocessorBuilder(text)
    if parameters.should_to_lower():
        builder.to_lower()

    if parameters.should_remove_punctuation():
        builder.remove_punctuation()

    if parameters.should_remove_specific_pos():
        builder.remove_specific_pos()

    if parameters.should_remove_stopwords():
        builder.remove_stopwords

    if parameters.should_lemmatize():
        builder.lemmatize()

    if parameters.should_merge_spaces():
        builder.merge_spaces

    if parameters.should_strip():
        builder.strip()

    if parameters.get_num_conversion_strategy():
        if parameters.get_num_conversion_strategy() == parameters.NUM_TO_WORD_METHOD:
            builder.num_to_word(parameters.get_min_num_length())
        elif parameters.get_num_conversion_strategy() == parameters.NUM_TO_CHAR_METHOD:
            builder.num_to_char(parameters.get_min_num_length())
        elif parameters.get_num_conversion_strategy() == parameters.NUM_TO_CHAR_LONG_METHOD:
            builder.num_to_char_long(parameters.get_min_num_length())
    
    return builder.build()


def preprocess_text(text) -> list[str]:
    important_sentences = TextSummarizer.process_long_text(text, parameters.get_min_num_sentences())
    return [preprocess_text_no_summary(sent) for sent in important_sentences]


def _create_chunks_with_context(corpus, chunk_len, context_left, context_right):
    """
    This function takes a corpus of text and splits it into chunks of a specified length, 
    then adds a specified amount of context to each chunk. The context is added by first 
    going backwards from the start of the chunk and then going forwards from the end of the 
    chunk, ensuring that the context includes only whole words and that the total context length 
    does not exceed the specified limit. This function uses binary search for efficiency.

    Returns:
    chunks (list of str): The chunks of text.
    chunks_with_context (list of str): The chunks of text with added context.
    chunk_with_context_start_indices (list of int): The starting indices of each chunk with context in the corpus.
    """
    words = re.split('(\\s+)', corpus)
    word_start_indices = [0]
    current_index = 0

    for word in words:
        current_index += len(word)
        word_start_indices.append(current_index)

    chunks, chunk_lengths, chunk_start_indices, chunk_with_context_start_indices = [], [], [], []
    current_length = 0
    current_index = 0
    chunk = []

    for word in words:
        if current_length + len(word) > chunk_len:
            chunks.append(''.join(chunk))
            chunk_lengths.append(current_length)
            chunk_start_indices.append(current_index - current_length)
            chunk = [word]
            current_length = len(word)
        else:
            chunk.append(word)
            current_length += len(word)
        current_index += len(word)

    if chunk:
        chunks.append(''.join(chunk))
        chunk_lengths.append(current_length)
        chunk_start_indices.append(current_index - current_length)

    chunks_with_context = []
    for start_index, chunk_length in zip(chunk_start_indices, chunk_lengths):
        context_start_index = bisect.bisect_right(word_start_indices, start_index - context_left)
        context_end_index = bisect.bisect_left(word_start_indices, start_index + chunk_length + context_right)

        # Combine all the words in the context range (before, chunk, and after)
        chunk_with_context = ''.join(words[context_start_index:context_end_index])
        chunks_with_context.append(chunk_with_context)
        
        # Determine the start index of the chunk with context
        chunk_with_context_start_index = word_start_indices[context_start_index]
        chunk_with_context_start_indices.append(chunk_with_context_start_index)

    return chunks, chunks_with_context, chunk_with_context_start_indices


def _clear_chunks(data_chunks, data_chunks_with_context, data_chunk_starting_indices):
    distinct_data_chunks = []
    distinct_data_chunks_with_context = []
    distinct_data_chunk_starting_indices = []

    seen_chunks = dict()

    for chunk, context, index in zip(data_chunks, data_chunks_with_context, data_chunk_starting_indices):
        # Skip the chunk if it does not contain any alphanumeric characters
        if not any(char.isalnum() for char in chunk):
            continue

        seen_chunk_start = seen_chunks.get(chunk)
        if seen_chunk_start:
            # If we've already seen this exact chunk, and the context around it it very close to the seen chunk, then skip it.
            if abs(seen_chunk_start-index) < parameters.get_delta_start():
                continue
        
        distinct_data_chunks.append(chunk)
        distinct_data_chunks_with_context.append(context)
        distinct_data_chunk_starting_indices.append(index)

        seen_chunks[chunk] = index

    return distinct_data_chunks, distinct_data_chunks_with_context, distinct_data_chunk_starting_indices


def process_and_add_to_collector(corpus: str, collector: ChromaCollector, clear_collector_before_adding: bool, metadata: dict):
    # Defining variables
    chunk_lens = [int(len.strip()) for len in parameters.get_chunk_len().split(',')]
    context_len = [int(len.strip()) for len in parameters.get_context_len().split(',')]
    if len(context_len) >= 3:
        raise f"Context len has too many values: {len(context_len)}"
    if len(context_len) == 2:
        context_left = context_len[0]
        context_right = context_len[1]
    else:
        context_left = context_right = context_len[0]

    data_chunks = []
    data_chunks_with_context = []
    data_chunk_starting_indices = []

    # Handling chunk_regex
    if parameters.get_chunk_regex():
        if parameters.get_chunk_separator():
            cumulative_length = 0  # This variable will store the length of the processed corpus
            sections = corpus.split(parameters.get_chunk_separator())
            for section in sections:
                special_chunks = list(re.finditer(parameters.get_chunk_regex(), section))
                for match in special_chunks:
                    chunk = match.group(0)
                    start_index = match.start()
                    end_index = start_index + len(chunk)
                    context = section[max(0, start_index - context_left):min(len(section), end_index + context_right)]
                    data_chunks.append(chunk)
                    data_chunks_with_context.append(context)
                    data_chunk_starting_indices.append(cumulative_length + max(0, start_index - context_left))
                cumulative_length += len(section) + len(parameters.get_chunk_separator())  # Update the length of the processed corpus
        else:
            special_chunks = list(re.finditer(parameters.get_chunk_regex(), corpus))
            for match in special_chunks:
                chunk = match.group(0)
                start_index = match.start()
                end_index = start_index + len(chunk)
                context = corpus[max(0, start_index - context_left):min(len(corpus), end_index + context_right)]
                data_chunks.append(chunk)
                data_chunks_with_context.append(context)
                data_chunk_starting_indices.append(max(0, start_index - context_left))

    for chunk_len in chunk_lens:
        # Breaking the data into chunks and adding those to the db
        if parameters.get_chunk_separator():
            cumulative_length = 0  # This variable will store the length of the processed corpus
            sections = corpus.split(parameters.get_chunk_separator())
            for section in sections:
                chunks, chunks_with_context, context_start_indices = _create_chunks_with_context(section, chunk_len, context_left, context_right)
                context_start_indices = [cumulative_length + i for i in context_start_indices]  # Add the length of the processed corpus to each start index
                data_chunks.extend(chunks)
                data_chunks_with_context.extend(chunks_with_context)
                data_chunk_starting_indices.extend(context_start_indices)
                cumulative_length += len(section) + len(parameters.get_chunk_separator())  # Update the length of the processed corpus
        else:
            chunks, chunks_with_context, context_start_indices = _create_chunks_with_context(corpus, chunk_len, context_left, context_right)
            data_chunks.extend(chunks)
            data_chunks_with_context.extend(chunks_with_context)
            data_chunk_starting_indices.extend(context_start_indices)

    data_chunks = [preprocess_text_no_summary(chunk) for chunk in data_chunks]

    data_chunks, data_chunks_with_context, data_chunk_starting_indices = _clear_chunks(
        data_chunks, data_chunks_with_context, data_chunk_starting_indices
    )

    if clear_collector_before_adding:
        collector.clear()
    collector.add(data_chunks, data_chunks_with_context, data_chunk_starting_indices, [metadata]*len(data_chunks) if metadata is not None else None)