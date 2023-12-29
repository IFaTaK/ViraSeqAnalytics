"""
This module provides tools for analyzing specific gene regions in a DNA sequence by comparing them to a reference genome sequence. It utilizes the FM-index, an efficient data structure for text searching, to find and analyze gene segments within the sequence.

The module contains two main functions:
- `extract_gene`: Extracts segments of a specific gene from the DNA sequence.
- `data_from_genome`: Analyzes the match rate of specified gene regions in a sequence against the reference genome.

The FM-index of the reference genome is precomputed and used for efficient searching and matching of gene segments. This approach allows for the rapid identification of matching rates in the specified gene regions, which is crucial in bioinformatics for comparative genomic studies and variant analysis.

Attributes:
    FMINDEX_REFERENCE (FmIndex): An FM-index object created from the reference genome sequence.

Functions:
    extract_gene(sequence, gene_start, gene_end, len_extraction): Extracts segments of a gene.
    data_from_genome(sequence, fmindex_reference, list_idxs_genes, len_extraction): Analyzes gene regions.
"""
try:
    from genome import REFERENCE_SEQUENCE
    from fmindex import FmIndex
except ImportError:
    from src.genome import REFERENCE_SEQUENCE
    from src.fmindex import FmIndex

# Creating an FM-index object for the reference genome sequence.
FMINDEX_REFERENCE = FmIndex(REFERENCE_SEQUENCE)

def extract_gene(sequence, gene_start, gene_end, len_extraction=100):
    """
    Extracts segments of a gene from a given sequence.

    Args:
        sequence (str): The DNA sequence from which the gene is to be extracted.
        gene_start (int): The starting index of the gene in the sequence.
        gene_end (int): The ending index of the gene in the sequence.
        len_extraction (int): Length of each segment to be extracted. Defaults to 100.

    Returns:
        list: A list of extracted gene segments.

    Raises:
        IndexError: If the gene_end index is beyond the length of the sequence.
    """
    if len(sequence) < gene_end:
        raise IndexError("Gene end index is out of the sequence range.")

    list_extraction = []
    for k in range(gene_start, gene_end + len_extraction, len_extraction):
        list_extraction.append(sequence[k:min(k + len_extraction, gene_end + 1)])
    return list_extraction

def data_from_genome(sequence, fmindex_reference=FMINDEX_REFERENCE, list_idxs_genes=None, len_extraction=500):
    """
    Analyzes the match rate of specified gene regions in a given sequence against a reference genome.

    Args:
        sequence (str): The DNA sequence to be analyzed.
        fmindex_reference (FmIndex): FM-index object of the reference genome.
        list_idxs_genes (list of tuples): List of tuples where each tuple contains the start and end indices of gene regions.
        len_extraction (int): Length of segments for extraction and analysis. Defaults to 100.

    Returns:
        list: A list of match rates for each specified gene region.

    Note:
        The match rate is calculated as the proportion of segments in the gene region that match the reference genome.
    """
    def is_suffix_in_gene(list_idx_suffix, start, end, rank):
        return any(start <= idx_suffix <= end + 1 - rank for idx_suffix in list_idx_suffix)

    if not list_idxs_genes:
        # Default gene regions to analyze
        list_idxs_genes = [(266, 13484), (21563, 25385), (25393, 26221), (26245, 26472), (26523, 27191), (27202, 27387), (27394, 27759), (27756, 27887), (27894, 28259), (28274, 29533), (29558, 29674)]

    match_rate_genes = []
    for start, end in list_idxs_genes:
        gene_sequence_split = extract_gene(sequence, start, end, len_extraction)
        count = 0
        for extraction in gene_sequence_split:
            for rank in range(len(extraction)):
                list_idx_suffix = fmindex_reference.occurrences(extraction[:rank + 1])
                if is_suffix_in_gene(list_idx_suffix, start, end, rank):
                    count += 1

        match_rate_genes.append(count / (end - start + 1))

    return match_rate_genes
