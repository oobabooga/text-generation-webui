import torch

from modules import shared


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


def find_longest_common_substring_indices(s1, s2):
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

    len1, len2 = len(s1), len(s2)
    ir, jr = 0, -1
    ir2, jr2 = 0, -1
    for i1 in range(len1):
        i2 = s2.index(s1[i1]) if s1[i1] in s2 else -1
        while i2 >= 0:
            j1, j2 = i1, i2

            while True:
                walked = False

                j1_bak = j1
                j2_bak = j2

                while j1 < len1 and j2 < len2 and s2[j2] == s1[j1]:
                    if j1 - i1 >= jr - ir:
                        ir, jr = i1, j1
                        ir2, jr2 = i2, j2

                    j1 += 1
                    j2 += 1
                    walked = True

                if walked:
                    continue

                # The normal function would not have the next 2 while loops.
                # They are there to account for different tokenizations of the
                # same sequence. For instance, in the Llama tokenizer,
                # [306, 311] and [13001] are the same thing.

                j1 = j1_bak + 1
                j2 = j2_bak + 2
                while j1 < len1 and j2 < len2 and s2[j2] == s1[j1]:
                    if j1 - i1 >= jr - ir:
                        ir, jr = i1, j1
                        ir2, jr2 = i2, j2

                    j1 += 1
                    j2 += 1
                    walked = True

                if walked:
                    continue

                j1 = j1_bak + 2
                j2 = j2_bak + 1
                while j1 < len1 and j2 < len2 and s2[j2] == s1[j1]:
                    if j1 - i1 >= jr - ir:
                        ir, jr = i1, j1
                        ir2, jr2 = i2, j2

                    j1 += 1
                    j2 += 1
                    walked = True

                if not walked:
                    break

            try:
                i2 = s2.index(s1[i1], i2 + 1)
            except ValueError:
                i2 = -1

    return ir, jr, ir2, jr2


def handle_llamacpp_prefix_and_streamingllm(model, past_seq, seq, seq_tensor):
    reset = True

    if shared.args.streaming_llm:
        i1, i2, j1, j2 = find_longest_common_substring_indices(past_seq.tolist(), seq)
        overlap_length = i2 - i1 + 1

        # A removed chunk has been found.
        # i1 > 0 means that the longest common substring is not the prefix.
        # short overlaps of few tokens are ignored (like a single '\n' character)
        if i1 > 0 and overlap_length > 0.2 * seq_tensor.shape[-1]:
            reset = False

            prefix_length = find_prefix_length(past_seq[:i1], seq_tensor[:j1])
            sink_length = prefix_length
            if sink_length < shared.args.attention_sink_size:
                sink_length = shared.args.attention_sink_size

            removed_length = i1 - sink_length

            matching_prefix = past_seq[:prefix_length]
            removed_chunk = past_seq[sink_length:i1]
            overlapping_sequence = seq_tensor[j1:j2 + 1]
            added_chunk = seq_tensor[j2 + 1:]

            print(past_seq)
            print(seq_tensor)

            print('\n\n')
            print('MATCHING PREFIX=', repr(shared.tokenizer.decode(matching_prefix)))
            print('REMOVED CHUNK=', repr(shared.tokenizer.decode(removed_chunk)))
            # print('OVERLAPPING SEQUENCE=', repr(shared.tokenizer.decode(overlapping_sequence)))
            print('ADDED CHUNK=', repr(shared.tokenizer.decode(added_chunk)))
            print('\n\n')

            print('------------------')

            # Remove interval [sink_length, sink_length + removed_length) from the context
            # Subtract removed_length from model.n_tokens
            model._ctx.kv_cache_seq_rm(0, sink_length, sink_length + removed_length)
            model._ctx.kv_cache_seq_shift(0, sink_length + removed_length, -1, -removed_length)

            model.n_tokens -= removed_length
            model.eval(seq[prefix_length + overlap_length:])

        # No removed chunk has been found
        else:
            prefix_length = find_prefix_length(past_seq, seq_tensor)
            if prefix_length > 0:
                reset = False
                model.n_tokens = prefix_length
                if len(seq_tensor) - prefix_length > 0:
                    model.eval(seq[prefix_length:])

    else:
        prefix_length = find_prefix_length(past_seq, seq_tensor)
        if prefix_length > 0:
            reset = False
            model.n_tokens = prefix_length
            if len(seq_tensor) - prefix_length > 0:
                model.eval(seq[prefix_length:])

    return reset
