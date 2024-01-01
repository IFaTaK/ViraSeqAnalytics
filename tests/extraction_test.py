import unittest
from src.extraction import extract_gene, data_from_genome, FMINDEX_REFERENCE

class TestGeneExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample data setup
        cls.sample_sequence = "ATCG" * 1000  # Example sequence
        cls.sample_gene_start = 100
        cls.sample_gene_end = 200
        cls.len_extraction = 50
        cls.list_idxs_genes = [(100, 200)]

    def test_extract_gene_valid_input(self):
        # Test extraction with valid inputs
        result = extract_gene(self.sample_sequence, self.sample_gene_start, self.sample_gene_end, self.len_extraction)
        print(result)
        self.assertEqual(len(result), 3)  # Check if the correct number of segments are extracted

    def test_extract_gene_invalid_input(self):
        # Test extraction with invalid input (e.g., end index greater than sequence length)
        with self.assertRaises(IndexError):
            extract_gene(self.sample_sequence, 100, len(self.sample_sequence) + 100)

    def test_data_from_genome(self):
        # Test data_from_genome function
        match_rates = data_from_genome(self.sample_sequence, FMINDEX_REFERENCE, self.list_idxs_genes)
        self.assertIsInstance(match_rates, list)  # Check if the result is a list
        self.assertTrue(all(0 <= rate <= 1 for rate in match_rates))  # Check if all match rates are between 0 and 1

    # Additional tests for edge cases and specific behaviors can be added here

if __name__ == '__main__':
    unittest.main()
