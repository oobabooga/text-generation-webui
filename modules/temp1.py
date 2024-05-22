import torch
import time
import random

#################################################################################################
#### TODO: Delete this file before merging PR
####
#### This file is intended to help reviewers test this PR
#### See the bottom of this file for how to run your own test cases,
#### 1. to verify that the optimized function produces same results as the original function
#### 2. to measure the difference in runtime for different test cases
#################################################################################################

def dry_optimized(input_ids_row, allowed_length, multiplier, base, sequence_breakers):
    penalties = {}

    # Use normal Python data types for improved performance
    s = input_ids_row.tolist()
    last_token = s[-1]

    if last_token in sequence_breakers:
        # This check does not affect results, but it saves a tiny bit of compute.
        return penalties

    # Create z array where each value indicates match length starting from that index going backwards.
    # For performance reasons this is implemented with a variant of the "Z algorithm", which runs in linear time.
    z = [0] * len(s)
    end = len(s) - 1
    # we will move a window through the input tokens (excluding the last token)
    # using a two pointer technique where the pointers only move in one direction (thus guaranteeing linear runtime)
    right = end - 1
    left = end - 1
    while right >= 0:
        while left == right and left >= 0:
            # We are looking for the start of a new match
            if s[right] == s[end]:
                # Start of new match found
                break
            else:
                # Match not found
                right -= 1
                left -= 1

        while left >= 0 and s[left] == s[end - (right - left)]:
            # Current match continues, update match_length starting from token at right pointer
            z[right] += 1
            # Move left pointer to expand the window
            left -= 1

        helper = right
        while right > left:
            # Move right pointer to contract the window
            right -= 1
            # Check if window is collapsed to size 0
            if left == right:
                break
            # This is the magical step in z algorithm where we use previously computed z values to avoid quadratic runtime!
            z[right] = min(z[end - (helper - right)], right - left)
            # Check if the result we got is complete result or partial result
            if left >= 0 and right - z[right] <= left:
                # We can't know if the match extends beyond current window, so we need to expand the window one or more times and possibly update the result
                break

    # Sequence breakers: find the first sequence breaker from the end of input, so that we can count how long is the maximum repeatable sequence
    max_match_length = 0
    while max_match_length < len(s) and s[len(s) - max_match_length - 1] not in sequence_breakers:
        max_match_length += 1

    # Sequence breakers: cap all match_length values so that none of the sequences cross over a sequence breaker
    z = [min(match_length, max_match_length) for match_length in z]

    # Apply penalties (excluding the last token)
    for idx, match_length in enumerate(z[:-1]):
        # No penalty unless match length exceeds allowed length
        if match_length >= allowed_length:
            # idx is the last token of the repeated sequence, and we want to attribute the penalty to the next token after that one
            token = s[idx + 1]
            # Penalty formula is unchanged from original DRY implementation
            penalty = multiplier * base ** (match_length - allowed_length)
            # If the same token appears multiple times in input, take the maximum penalty of these occurrences
            penalties[token] = max(penalty, penalties.get(token, 0))
    
    return penalties

# This is the original DRY function, for comparison purposes (modified slightly to facilitate testing)
def dry_original(input_ids_row, allowed_length, multiplier, base, sequence_breakers):
    penalties = {}

    # Raw integer must be extracted here to check for set membership.
    last_token = input_ids_row[-1].item()

    if last_token in sequence_breakers:
        return penalties

    # Exclude the last token as it always matches.
    match_indices = (input_ids_row[:-1] == last_token).nonzero()

    # Stores the maximum matching sequence length
    # for each token immediately following the sequence in the input.
    match_lengths = {}

    for i in match_indices:
        next_token = input_ids_row[i+1].item()

        if next_token in sequence_breakers:
            return penalties

        # We have already found that `last_token` matches at this index,
        # so the match is at least of length 1.
        match_length = 1

        # Extend the match backwards as far as possible.
        while True:
            j = i - match_length
            if j < 0:
                # Start of input reached.
                break

            previous_token = input_ids_row[-(match_length+1)].item()
            if input_ids_row[j] != previous_token:
                # Start of match reached.
                break

            if previous_token in sequence_breakers:
                # Sequence-breaking token reached.
                break

            match_length += 1

        if next_token in match_lengths:
            match_lengths[next_token] = max(match_length, match_lengths[next_token])
        else:
            match_lengths[next_token] = match_length

    # Apply penalties.
    for token, match_length in match_lengths.items():
        if match_length >= allowed_length:
            penalty = multiplier * base ** (match_length - allowed_length)
            penalties[token] = penalty
    
    return penalties

# Helper function to run tests
def test(input_ids_list, allowed_length = 2, multiplier = 0.8, base = 1.75, sequence_breakers = [], verbose = True):
    input_ids_row = torch.tensor(input_ids_list)
    if verbose:
        print('\n**********************************************************')
        print('Input: ' + str(input_ids_row if len(input_ids_row) < 20 else (str(input_ids_row[:15])[:-2] + ', ...] (' + str(len(input_ids_row)) + ' tokens)')))

    # Run input with optimized function and time it
    time_opti_start = time.time()
    penalties_opti = dry_optimized(input_ids_row, allowed_length, multiplier, base, sequence_breakers)
    time_opti_end = time.time()
    if verbose:
        print(str(round(time_opti_end - time_opti_start, 2)) + ' seconds runtime for dry_optimized')

    # Run input with original function and time it
    time_orig_start = time.time()
    penalties_orig = dry_original(input_ids_row, allowed_length, multiplier, base, sequence_breakers)
    time_orig_end = time.time()
    if verbose:
        print(str(round(time_orig_end - time_orig_start, 2)) + ' seconds runtime for dry_original')

    # Compare penalties from optimized and original
    if (penalties_opti != penalties_orig):
        print("Penalties from dry_optimized: " + str(penalties_opti))
        print("Penalties from dry_original: " + str(penalties_orig))
        raise ValueError("Test FAIL! Penalties from original and optimized do not match.")

    if verbose:
        print("Test PASS. Penalties from dry_optimized and dry_original are the same: " + str(penalties_opti))

def generateTest(token_count, vocab_size):
    input_ids_list = []
    for i in range(token_count):
        v = round(random.random() * vocab_size)
        input_ids_list.append(v)
    return input_ids_list    

def runGeneratedTestsForever():
    MAX_SIZE = 100
    i = 0
    while True:
        i += 1
        token_count = 1 + round(random.random() * MAX_SIZE)
        vocab_size = 1 + round(random.random() * MAX_SIZE)
        input_ids_list = generateTest(token_count, vocab_size)
        test(input_ids_list, verbose = False)
        if i % 1000 == 0:
            print(f'{i} tests PASS...')

#################################################################################################
#### Define your test cases below
#################################################################################################

test([1337])
test([1, 1], allowed_length = 1)
test([5, 2, 2, 2])
test([1, 2, 2, 1, 2], allowed_length = 1)

# Very good test for Z algorithm
test([1,4,3,1,2,3,1,2,3,1,4,3,1,2,3,1], allowed_length = 1)
#     1 0 0 2 0 0 7 0 0 5 0 0 2 0 0 - (expected z array)

# Sequence breaker test
test([1,2,3,4,5,999,6,7,8,9,1,2,3,4,5,999,6,7,8,9], sequence_breakers = [999, 666])

# "Worst case" adversarial input where the same token is repeated many times
test([42] * 1024)

# Uncomment this to run test generator
#runGeneratedTestsForever()