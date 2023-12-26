import numpy as np
from itertools import *

def merge_reads(reads, coverage = 1, epsilon=79):
    def has_expected_coverage(p):
        phi = K*L - sum(p)
        expected = (1-np.exp(-coverage))
        return abs(phi-expected) <= epsilon*expected
    def has_expected_number_island(p):
        G_p_0 = sum([idx if p[idx]==0 else 0 for idx in range(K)])
        expected = K*np.exp(-coverage)
        return abs(G_p_0 - expected) <= epsilon*expected
    def has_not_too_much_zero(p):
        return p.count(0) <= 3
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
        # if has_expected_coverage(partition_vector) and has_expected_number_island(partition_vector):
        # if has_not_too_much_zero(partition_vector):
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
                str_res = ""
                for idx in range(K-1):
                    if partition_vector[idx] == 0:
                        str_res += reads[permutation[idx]]
                        island_list.append(str_res)
                        str_res = ""
                    else: 
                        str_res.append(merge(reads[permutation[idx]],reads[permutation[idx+1]],partition_vector[idx]))
                if len(island_list) < m:
                    res.append(island_list)
                    m = len(island_list)   
                # res.append(island_list)
    return res                     
                
    
np.random.seed(0)
str_test = "anticonstitutionnellement"
reads_test = reads(str_test,7,2)
print(reads_test)
loloi = merge_reads(reads_test)
print(loloi)