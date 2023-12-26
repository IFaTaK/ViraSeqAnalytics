import unittest
from src.aligngenome import compare_long_sequences, compare_sequences, create_reads, \
                       experimental_merge_reads, find_longest_overlap, overlap_length, \
                       greedy_sanger_merge, correct_offset, read_sequence

class TestAlignGenome(unittest.TestCase):

    def test_compare_long_sequences(self):
        # Test comparing long sequences
        reference = 'ACGT' * 40
        comparison = 'ACGT' * 40
        # Expected: Should print the alignment without errors
        compare_long_sequences(reference, comparison)

    def test_compare_sequences(self):
        # Test comparing two sequences
        reference = 'ACGTACGT'
        comparison = 'ACGTTTGT'
        # Expected: Should print the alignment with some mismatches
        compare_sequences(reference, comparison)

    def test_create_reads(self):
        # Test creating reads from a sequence
        sequence = 'ACGT' * 10
        reads_length = 10
        num_samples = 5
        result = create_reads(sequence, reads_length, num_samples)
        self.assertEqual(len(result), len(sequence)//reads_length * num_samples)
        for read in result:
            self.assertTrue(len(read) <= reads_length)

    def test_experimental_merge_reads(self):
        # Test merging reads experimentally
        reads = ['ACGT', 'CGTA', 'GTAC']
        merged = experimental_merge_reads(reads)
        self.assertIsInstance(merged, list)

    def test_find_longest_overlap(self):
        # Test finding the longest overlap
        reads = ['ACGT', 'CGTA', 'GTAC']
        read1, read2, overlap = find_longest_overlap(reads)
        self.assertIsNotNone(read1)
        self.assertIsNotNone(read2)
        self.assertNotEqual(overlap, '')

    def test_overlap_length(self):
        # Test calculating overlap length
        read1 = 'ACGT'
        read2 = 'CGTA'
        length, overlap = overlap_length(read1, read2, 2)
        self.assertGreater(length, 0)
        self.assertNotEqual(overlap, '')

    def test_greedy_sanger_merge(self):
        # Test greedy merging of reads
        reads = ['ACGT', 'CGTA', 'GTAC']
        result = greedy_sanger_merge(reads)
        self.assertIsInstance(result, str)

    def test_correct_offset(self):
        # Test correcting offset in a read
        original = 'ACGTACGT'
        read = 'GTACGTAC'
        corrected = correct_offset(original, read, 4)
        self.assertEqual(corrected, original)

    def test_read_sequence(self):
        # Test simulating read sequence
        sequence = 'ACGT' * 10
        read_length = 10
        num_samples = 5
        result = read_sequence(sequence, read_length, num_samples)
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()
