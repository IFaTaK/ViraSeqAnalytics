"""
This module provides functions for DNA sequence analysis, particularly focusing on the comparison, 
creation, and merging of DNA reads. It includes methods for comparing DNA sequences, generating 
simulated DNA reads, and using a greedy algorithm to merge these reads based on overlaps. 

The module utilizes a combination of basic string manipulation and more complex algorithms 
like greedy merging and overlap detection to facilitate these tasks. These functions are 
useful in bioinformatics for tasks such as sequence alignment and genome assembly.

Functions:
    compare_long_sequences: Compares long DNA sequences in segments.
    compare_sequences: Compares two DNA sequences and prints their alignment.
    create_reads: Creates simulated reads from a given DNA sequence.
    experimental_merge_reads: Attempts to merge sequencing reads based on coverage and overlap criteria.
    find_longest_overlap: Finds the longest overlap between any two reads in a list.
    overlap_length: Finds the longest overlap where the suffix of one read matches the prefix of another.
    greedy_sanger_merge: Merges reads using a greedy approach.
    correct_offset: Corrects the offset in a read sequence compared to the original sequence.
    read_sequence: Simulates the reading of a DNA sequence, creating reads and merging them.
"""
from itertools import combinations_with_replacement, permutations
import numpy as np

def compare_long_sequences(reference, comparison):
    """
    Compares two long DNA sequences in segments and prints the comparison.

    Args:
        reference (str): The reference DNA sequence.
        comparison (str): The DNA sequence to compare against the reference.

    Returns:
        None: The function outputs the comparison result to the console.
    """
    print("\n")
    for k in range(0,len(reference),150):
        compare_sequences(reference[k:min(len(reference),150+k)],comparison[k:min(len(comparison),150+k)])
        print('\n')

def compare_sequences(reference, comparison):
    """
    Compares two DNA sequences and prints their alignment.

    This function aligns two sequences (reference and comparison) and
    visually represents the differences between them. Matching bases are
    replaced with underscores, and mismatches are shown with the base from
    the comparison sequence.

    Args:
        reference (str): The reference DNA sequence.
        comparison (str): The DNA sequence to compare against the reference.

    Returns:
        None: The function outputs the comparison result to the console.
    """
    # Define the format for printing the sequences
    format_str = "{0:7}{1:150}"
    aligned_sequence = ""

    # Create a string representing the alignment
    for k in range(min(150, len(reference), len(comparison))):
        aligned_sequence += "_" if reference[k] == comparison[k] else comparison[k]

    # Prepare the data for printing
    lines_to_print = [
        ["Ref : ", reference],
        ["Comp : ", aligned_sequence],
        ["N° : ", "".join([str(k % 10) for k in range(150)])]
    ]

    # Print the aligned sequences and numbering
    for line in lines_to_print:
        print(format_str.format(*line))

def create_reads(sequence, reads_len, num_samples):
    """
    Creates simulated reads from a given DNA sequence.

    Args:
        seq (str): The original DNA sequence.
        reads_len (int): Length of each read.
        number_sample (int): Number of reads to generate.

    Returns:
        list: A list of simulated DNA reads.
    """
    reads = []
    sequence_read = sequence + sequence[:reads_len]
    for _ in range(num_samples):
        start = np.random.randint(0,reads_len)
        idx = start
        while idx < len(sequence):
            reads.append(sequence_read[idx:min(idx+reads_len,len(sequence_read))])
            idx += reads_len
    return reads

def experimental_merge_reads(reads, coverage=3, epsilon=2):
    """
    [Experimental Function]
    
    Attempts to merge Sanger sequencing reads based on coverage and overlap criteria.
    This function is currently in the experimental stage and may not perform as expected.
    
    The algorithm explores different combinations and permutations of the reads, attempting
    to merge them based on prefix-suffix matches and coverage criteria. It is designed to 
    handle overlapping sequencing reads but has limitations in its current form.
    
    Args:
        reads (list of str): List of sequencing reads to be merged.
        coverage (int, optional): Expected coverage for reads. Defaults to 3 (for testing).
        epsilon (int, optional): Tolerance level for deviation from expected coverage. Defaults to 2 (for testing).
    
    Returns:
        list: A list of merged read sequences, representing potential assembly outcomes. 
              In its current implementation, this result may not be optimal or accurate.
    
    Known Limitations:
    - The function does not handle complex overlaps or sequencing errors effectively.
    - Computationally intensive for large datasets due to exhaustive combinatorial approach.
    - The accuracy and performance have not been extensively validated.
    
    Note:
    This function is part of ongoing research and development and is subject to change.
    Use with caution and in experimental contexts only.
    """
    def has_expected_coverage(p):
        phi = K*L - sum(p)
        expected = 1-np.exp(-coverage)
        return expected - phi <= epsilon*expected
    def has_expected_number_island(p):
        gp0 = sum([idx if p[idx]==0 else 0 for idx in range(K)])
        expected = K*np.exp(-coverage)
        return expected - gp0 <= epsilon*expected
    def is_prefix_suffix_match(read1,read2,rank):
        def suffix(text,rank):
            return "" if rank==0 else text[-rank:]
        def prefix(text,rank):
            return text[:rank]
        return suffix(read1,rank) == prefix(read2,rank)
    def merge(read1,read2,rank):
        return read1 + read2[rank+1:]
    K = len(reads)
    L = len(reads[0])
    res = []
    m = K
    for partition_vector in list(combinations_with_replacement(range(L+1),K)):
        if has_expected_coverage(partition_vector) and has_expected_number_island(partition_vector):
            for permutation in permutations(range(K)):
                test = True
                for idx,pi in enumerate(partition_vector):
                    if idx == K-1:
                        test = test and pi == 0
                        break
                    # print("read1 : {}   read2: {}   rank: {}".format(reads[permutation[idx]],reads[permutation[idx+1]],partition_vector[idx]))
                    test = test and is_prefix_suffix_match(reads[permutation[idx]],reads[permutation[idx+1]],pi)
                    if not test:
                        break
                if not test:
                    break
                island_list = []
                str_overlap_list = []
                for idx in range(K-1):
                    if partition_vector[idx] == 0:
                        str_overlap_list.append((reads[permutation[idx]],partition_vector[idx]))
                        str_temp = str_overlap_list[0][0]
                        for i in range(1,len(str_overlap_list)-1):
                            str_temp = merge(str_temp,str_overlap_list[i][0],str_overlap_list[i-1][1])
                        # if not str_temp in island_list:
                        island_list.append(str_temp)
                        str_overlap_list = []
                    else:
                        str_overlap_list.append((reads[permutation[idx]],partition_vector[idx]))
                if  island_list not in res:
                    if len(island_list) < m:
                        res.append(island_list)
                        m = len(island_list)
                    # res.append(island_list)
    return res

def find_longest_overlap(reads, min_overlap=3):
    """
    Finds the longest overlap between any two reads in a list.

    Args:
        reads (list of str): The list of reads.
        min_overlap (int): Minimum length of overlap to consider.

    Returns:
        tuple: Indices of the two reads with the longest overlap and the overlap string.
    """
    max_overlap = 0
    read1, read2 = None, None
    overlap_text = ''
    for i,_ in enumerate(reads):
        for j,_ in enumerate(reads):
            if i != j:
                overlap_len, overlap = overlap_length(reads[i], reads[j], min_overlap)
                if overlap_len > max_overlap:
                    max_overlap, read1, read2 = overlap_len, i, j
                    overlap_text = overlap
    return read1, read2, overlap_text

def overlap_length(read1, read2, min_overlap):
    """
    Finds the longest overlap between any two reads in a list.

    Args:
        reads (list of str): The list of reads.
        min_overlap (int): Minimum length of overlap to consider.

    Returns:
        tuple: Indices of the two reads with the longest overlap and the overlap string.
    """
    start = 0
    while True:
        start = read1.find(read2[:min_overlap], start)
        if start == -1:  # No overlap found
            return 0, ''
        if read2.startswith(read1[start:]):
            return len(read1) - start, read1[start:]
        start += 1

def greedy_sanger_merge(reads, min_overlap=3):
    """
    Merges reads using a greedy approach, prioritizing the longest overlaps.

    Args:
        reads (list of str): List of reads to merge.
        min_overlap (int): Minimum length of overlap to consider for merging.

    Returns:
        list of str: Merged reads.
    """
    while True:
        read1, read2, overlap = find_longest_overlap(reads, min_overlap)
        if read1 is None:  # No overlaps found
            break
        # Merge reads
        reads[read1] += reads[read2][len(overlap):]
        del reads[read2]
    return reads[0]

def correct_offset(original, readed, reads_length):
    """
    Corrects the offset in a read sequence compared to the original sequence.

    Args:
        original (str): The original DNA sequence.
        read (str): The read sequence to correct.
        read_length (int): Length of the reads.

    Returns:
        str: Corrected read sequence.
    """
    res = readed
    res = 2*res[:min(len(res),len(original))]
    max_overlap = 0
    offset = 0
    for idx in range(reads_length):
        if original[:idx] in res and idx > max_overlap:
            offset = res.index(original[:idx])
            max_overlap = idx
    res = res[offset:] + res[:offset]
    res = res[:len(readed)]
    return res

def read_sequence(sequence, reads_length, num_samples):
    """
    Simulates the reading of a DNA sequence, creating reads and merging them.

    Args:
        sequence (str): The original DNA sequence.
        read_length (int): Length of each read.
        num_samples (int): Number of reads to generate.

    Returns:
        str: The read sequence after merging.
    """
    reads = create_reads(sequence,reads_length,num_samples)
    seq_with_offset = greedy_sanger_merge(reads)
    read_seq = correct_offset(sequence,seq_with_offset,reads_length)
    return read_seq
