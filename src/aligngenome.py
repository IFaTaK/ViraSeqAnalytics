import numpy as np
import itertools
from genome import OV447556

def LongCompSeq(Ref,Comp):
    print("\n")
    for k in range(0,len(Ref),150):
        CompSeq(Ref[k:min(len(Ref),150+k)],Comp[k:min(len(Comp),150+k)])
        print('\n')

def CompSeq(Ref,Comp):
    form = "{0:7}{1:150}"
    Res = ""
    for k in range(min(150,len(Ref),len(Comp))):
        if Ref[k] == Comp[k]:
            Res += "_"
        else:
            Res += Comp[k]

    prt = [["Ref : ",Ref],["Comp : ", Res],["N° : ", "".join([str(k%10) for k in range(0,150)])]]
    for val in prt:
        print(form.format(*val))

def reads(seq, read_len, number_sample):
    Res = [] 
    seq_read = seq + seq[:read_len] 
    for _ in range(number_sample):
        start = np.random.randint(0,read_len)
        idx = start
        while idx < len(seq):
            Res.append(seq_read[idx:min(idx+read_len,len(seq_read))])
            idx += read_len
    return Res


def find_overlap(read1, read2, min_length=3):
    """Find the length of the overlap between two reads."""
    start = 0  # Start at the beginning of read1
    while True:
        start = read1.find(read2[:min_length], start)
        if start == -1:  # No more occurrences to the right
            return 0
        # Found a potential overlap, verify if it's true
        if read2.startswith(read1[start:]):
            return len(read1) - start
        start += 1

def create_layout(reads, min_overlap_length):
    """Create a layout of reads based on overlaps."""
    graph = {}
    for read_a, read_b in itertools.permutations(reads, 2):
        overlap_length = find_overlap(read_a, read_b, min_overlap_length)
        if overlap_length > 0:
            graph[read_a] = (read_b, overlap_length)
    return graph

def derive_consensus(graph):
    """Derive the consensus sequence from a layout graph with cycles."""
    if not graph:
        return ""

    # Find the start read (a read with no incoming edges)
    all_targets = set(target for _, target in graph.values())
    start_read = next(read for read in graph if read not in all_targets)
    consensus = start_read

    visited = set()
    def find_next_read(read, visited):
        """Find the next read, avoiding cycles."""
        if read not in graph or read in visited:
            return None, 0
        next_read, overlap_len = graph[read]
        return next_read, overlap_len

    current_read = start_read
    while current_read:
        next_read, overlap_len = find_next_read(current_read, visited)
        if next_read:
            consensus += next_read[overlap_len:]
            visited.add(current_read)
        current_read = next_read

    return consensus


def merge_sanger_reads(reads, min_overlap_length=3):
    """Merge Sanger reads using Overlap-Layout-Consensus approach."""
    layout = create_layout(reads, min_overlap_length)
    consensus = derive_consensus(layout)
    return consensus

# Example usage
# np.random.seed(1)
test = OV447556[:10_000]
# test = "anticonstitutionnellement"
# reads_test = ['ACGT', 'CGTTG', 'TTGCA']
reads_test = reads(test,1000,1000)
consensus = merge_sanger_reads(reads_test)
# print(f"Consensus sequence: {consensus}")

def correct_offset(original, readed, read_len):
    res = readed
    res = res[:min(len(res),len(original))]
    max_overlap = 0
    offset = 0
    for idx in range(read_len):
        if original[:idx] in res and idx > max_overlap:
            offset = res.index(original[:idx])
            idx = max_overlap
    res = res[offset:] + res[:offset]
    return res

res = correct_offset(test,consensus,1000)
LongCompSeq(test,res)
# print(f"corrected consensus sequence: {res}")
