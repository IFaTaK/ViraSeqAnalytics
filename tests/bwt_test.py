import unittest
from src.bwt import rotations, bwm, bwt_from_bwm, suffix_array, bwt_from_sa, BWTransform

class TestBWTFunctions(unittest.TestCase):
    def setUp(self):
        # Setting up various text strings to be used in the test cases
        self.text1 = "banana$"
        self.text2 = "mississippi$"
        self.text3 = "abcd$"
        self.text4 = "ttttt$"
        self.text5 = "$"

    def test_rotations(self):
        # Test to ensure all rotations of the string are generated correctly
        expected1 = ["banana$", "anana$b", "nana$ba", "ana$ban", "na$bana", "a$banan", "$banana"]
        self.assertEqual(rotations(self.text1), expected1)

        expected2 = ['mississippi$', 'ississippi$m', 'ssissippi$mi', 'sissippi$mis', 'issippi$miss', 'ssippi$missi', 'sippi$missis', 'ippi$mississ', 'ppi$mississi', 'pi$mississip', 'i$mississipp', '$mississippi']
        self.assertEqual(rotations(self.text2), expected2)

    def test_bwm(self):
        # Test to ensure the Burrows-Wheeler Matrix is created correctly
        expected1 = ['$banana', 'a$banan', 'ana$ban', 'anana$b', 'banana$', 'na$bana', 'nana$ba']
        self.assertEqual(bwm(self.text1), expected1)

        expected2 = ['$mississippi', 'i$mississipp', 'ippi$mississ', 'issippi$miss', 'ississippi$m', 'mississippi$', 'pi$mississip', 'ppi$mississi', 'sippi$missis', 'sissippi$mis', 'ssippi$missi', 'ssissippi$mi']
        self.assertEqual(bwm(self.text2), expected2)

    def test_bwt_from_bwm(self):
        # Test to ensure the Burrows-Wheeler Transform from Matrix is created correctly
        self.assertEqual(bwt_from_bwm(self.text1), "annb$aa")
        self.assertEqual(bwt_from_bwm(self.text2), "ipssm$pissii")

    def test_suffix_array(self):
        # Test to ensure the Suffix Array is created correctly
        expected1 = [6, 5, 3, 1, 0, 4, 2]
        self.assertEqual(list(suffix_array(self.text1)), expected1)

        expected2 = [11, 10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]
        self.assertEqual(list(suffix_array(self.text2)), expected2)

    def test_bwt_from_sa(self):
        # Test to ensure the Burrows-Wheeler Transform from Suffix Array is created correctly
        self.assertEqual(bwt_from_sa(self.text1), "annb$aa")
        self.assertEqual(bwt_from_sa(self.text2), "ipssm$pissii")

    def test_BWTransform_inverse(self):
        # Testing the inverse transformation of the BWTransform class
        bwt1 = BWTransform(self.text1)
        self.assertEqual(bwt1.inverse_transform(), "banana$")

        bwt2 = BWTransform(self.text2)
        self.assertEqual(bwt2.inverse_transform(), "mississippi$")

        bwt3 = BWTransform(self.text3)
        self.assertEqual(bwt3.inverse_transform(), "abcd$")

        bwt4 = BWTransform(self.text4)
        self.assertEqual(bwt4.inverse_transform(), "ttttt$")

        bwt5 = BWTransform(self.text5)
        self.assertEqual(bwt5.inverse_transform(), "$")

    def test_BWTransform_repr(self):
        # Testing the string representation (__repr__) of BWTransform class
        bwt = BWTransform(self.text1)
        self.assertEqual(bwt.__repr__(), "annb$aa")

if __name__ == '__main__':
    unittest.main()
