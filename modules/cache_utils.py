import torch
from numba import njit

from modules import shared


def process_llamacpp_cache(model, new_sequence, past_sequence):
    if len(past_sequence) == 0 or len(new_sequence) == 0:
        return past_sequence

    i1, i2, j1, j2 = find_longest_common_substring_indices(past_sequence, new_sequence)
    overlap_length = i2 - i1 + 1

    # Do StreamingLLM if i1 > 0 (ie the longest common subsequence is not a prefix)
    # and the overlap length is sufficiently long.
    if i1 > 0 and overlap_length > 0.2 * len(new_sequence):

        new_sequence = torch.tensor(new_sequence)
        past_sequence = torch.tensor(past_sequence)

        prefix_length = find_prefix_length(past_sequence[:i1], new_sequence[:j1])
        sink_length = max(prefix_length, shared.args.attention_sink_size)
        removed_length = i1 - sink_length

        if removed_length <= 0:
            return past_sequence.tolist()

        matching_prefix = past_sequence[:prefix_length]
        removed_chunk = past_sequence[sink_length:i1]
        overlapping_sequence = new_sequence[j1:j2 + 1]
        added_chunk = new_sequence[j2 + 1:]

        # print(past_sequence.tolist())
        # print(new_sequence.tolist())

        print()
        print('MATCHING PREFIX=', repr(shared.tokenizer.decode(matching_prefix)))
        print('ADDED CHUNK=', repr(shared.tokenizer.decode(added_chunk)))
        print('REMOVED CHUNK=', repr(shared.tokenizer.decode(removed_chunk)))
        print('REMOVED LENGTH=', removed_length)
        print()

        # Remove interval [sink_length, sink_length + removed_length) from the context
        # Update model.n_tokens
        model._ctx.kv_cache_seq_rm(0, sink_length, sink_length + removed_length)
        model._ctx.kv_cache_seq_shift(0, sink_length + removed_length, -1, -removed_length)

        new_sequence = new_sequence.tolist()
        model.input_ids[:j2 + 1] = new_sequence[:j2 + 1]
        model.n_tokens = j2 + 1

        return new_sequence[:j2 + 1]
    else:
        return past_sequence


def find_prefix_length(past_seq, seq_tensor):
    '''
    Given two torch tensors, finds the length of the longest
    common prefix between the two.
    '''
    min_length = min(past_seq.shape[0], seq_tensor.shape[0])
    indices = torch.nonzero(~torch.eq(past_seq[:min_length], seq_tensor[:min_length]))
    if len(indices) > 0:
        prefix_length = indices[0].item()
    else:
        prefix_length = min_length

    return prefix_length


@njit
def find_longest_common_substring_indices(list1, list2):
    '''
    Given two lists, solves the Longest Common Substring problem.

    It returns the indices where the substring starts and ends in
    s1 and s2.

    Example:

    ir, jr, ir2, jr2 = find_longest_common_substring_indices(s1, s2)
    print(s1[ir:jr + 1])
    print(s2[ir2:jr2 + 1])

    Adapted from
    https://rosettacode.org/wiki/Longest_common_substring#Python
    '''

    len_list1, len_list2 = len(list1), len(list2)
    start_index_list1, end_index_list1 = 0, -1
    start_index_list2, end_index_list2 = 0, -1

    # for index1 in tqdm(range(0, len_list1), desc="StreamingLLM prompt comparison", leave=False):
    for index1 in range(0, len_list1):
        try:
            index2 = list2.index(list1[index1])
        except:
            continue

        while index2 >= 0:
            temp_index1, temp_index2 = index1, index2
            while temp_index1 < len_list1 and temp_index2 < len_list2 and list2[temp_index2] == list1[temp_index1]:
                if temp_index1 - index1 >= end_index_list1 - start_index_list1:
                    start_index_list1, end_index_list1 = index1, temp_index1
                    start_index_list2, end_index_list2 = index2, temp_index2

                temp_index1 += 1
                temp_index2 += 1
            try:
                index2 = list2.index(list1[index1], index2 + 1)
            except:
                break

    return start_index_list1, end_index_list1, start_index_list2, end_index_list2
