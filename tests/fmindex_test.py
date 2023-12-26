
import unittest
from src.fmindex import FmCheckpoints, FmIndex

class TestFmCheckpoints(unittest.TestCase):

    def setUp(self):
        self.bw = 'annb$aa'
                  #0123456
        self.checkpoints = FmCheckpoints(self.bw, cp_i_val=2)

    def test_rank(self):
        # Test rank function for various characters and positions
        self.assertEqual(self.checkpoints.rank(self.bw, 'a', 5), 2)
        self.assertEqual(self.checkpoints.rank(self.bw, 'n', 3), 2)
        self.assertEqual(self.checkpoints.rank(self.bw, 'b', 3), 1)
        self.assertEqual(self.checkpoints.rank(self.bw, 'a', 0), 1)

class TestFmIndex(unittest.TestCase):

    def setUp(self):
        self.text = 'banana$'
        self.fm_index = FmIndex(self.text)

    def test_count(self):
        self.assertEqual(self.fm_index.count('b'), 4)
        self.assertEqual(self.fm_index.count('$'), 0)

    def test_range(self):
        """
        0 $banana
        1 a$banan
        2 ana$ban
        3 anana$b
        4 banana$
        5 na$bana
        6 nana$ba
        """
        self.assertEqual(self.fm_index.range('ana'), (2, 4))

    def test_resolve(self):
        self.assertEqual(self.fm_index.resolve(3), 1)

    def test_has_substring(self):
        self.assertTrue(self.fm_index.has_sub_string('ana'))
        self.assertFalse(self.fm_index.has_sub_string('abc'))

    def test_has_suffix(self):
        # Testing if the method correctly identifies suffixes of the text
        self.assertTrue(self.fm_index.has_suffix('banana'))
        self.assertFalse(self.fm_index.has_suffix('banan'))
        self.assertTrue(self.fm_index.has_suffix('anana'))
        self.assertTrue(self.fm_index.has_suffix('a'))
        self.assertTrue(self.fm_index.has_suffix(''))

    def test_occurrences(self):
        self.assertEqual(sorted(self.fm_index.occurrences('ana')), [1, 3])

if __name__ == '__main__':
    unittest.main()
