import torch


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
            while j1 < len1 and j2 < len2 and s2[j2] == s1[j1]:
                if j1 - i1 >= jr - ir:
                    ir, jr = i1, j1
                    ir2, jr2 = i2, j2

                j1 += 1
                j2 += 1

            try:
                i2 = s2.index(s1[i1], i2 + 1)
            except ValueError:
                i2 = -1

    return ir, jr, ir2, jr2
