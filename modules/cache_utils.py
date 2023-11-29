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


def find_subsequence_index(past_seq, seq_tensor):
    '''
    Looks for the location of seq_tensor in past_seq.
    If seq_tensor is not a subsequence of past_seq, returns -1.
    '''
    # Find the indices where seq_tensor is equal to elements in past_seq
    indices = torch.where(past_seq == seq_tensor[0])

    # Check each potential starting index
    for index in indices[0]:
        # Extract the subsequence from past_seq starting at the potential index
        subsequence = past_seq[index:index + len(seq_tensor)]

        # Check if the extracted subsequence is equal to seq_tensor
        if torch.equal(subsequence, seq_tensor):
            return index.item()  # Return the starting index as a Python integer

    # If no match is found, return -1
    return -1


def find_streamingllm_lengths(past_seq, seq_tensor):
    for overlap_length in range(seq_tensor.shape[-1], 0, -1):
        removed_length = find_subsequence_index(past_seq, seq_tensor[:overlap_length])
        if removed_length != -1:
            return removed_length, overlap_length
    else:
        return -1, -1
