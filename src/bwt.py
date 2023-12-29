"""
This module implements the Burrows-Wheeler Transform (BWT) and related functions.
It's designed to transform strings into their BWT form and vice versa, 
facilitating data compression algorithms.

Functions:
- rotations(text): Generates all rotations of a given text.
- bwm(text): Constructs the Burrows-Wheeler Matrix (BWM) from the text.
- bwt_from_bwm(text): Generates BWT from BWM.
- suffix_array(text): Constructs the suffix array for a given text.
- bwt_from_sa(text): Generates BWT using the suffix array.
- BWTransform: Class that encapsulates BWT functionalities, including the inverse transform.
"""

def rotations(text):
    """
    Generate all rotations of a given text, appending '$' if not already present.
    Each rotation is a version of the text starting from a different character.

    Args:
    - text (str): The input text to rotate.

    Returns:
    - list: A list of all rotations of the text.
    """
    if text[-1] != "$":
        text = text + "$"

    return [(2*text)[idx:idx+len(text)] for idx in range(len(text))]

def bwm(text):
    """
    Construct the Burrows-Wheeler Matrix (BWM) from the text.
    This involves generating all rotations and then sorting them.

    Args:
    - text (str): The input text.

    Returns:
    - list: The sorted list of all rotations, forming the BWM.
    """
    return sorted(rotations(text))

def bwt_from_bwm(text):
    """
    Generate the Burrows-Wheeler Transform (BWT) from the BWM.

    Args:
    - text (str): The input text.

    Returns:
    - str: The BWT of the text.
    """
    return ''.join(map(lambda x: x[-1], bwm(text)))

def suffix_array(text):
    """
    Construct the suffix array for a given text, appending '$' if not present.
    The suffix array is an array of integers providing the starting positions 
    of suffixes of a string in lexicographical order.

    Args:
    - text (str): The input text.

    Returns:
    - iterator: An iterator of the suffix array.
    """
    if text[-1] != "$":
        text = text + "$"

    return map(lambda x: x[1], sorted([(text[idx:], idx) for idx in range(len(text))]))

def bwt_from_sa(text):
    """
    Generate the Burrows-Wheeler Transform (BWT) using the suffix array.

    Args:
    - text (str): The input text.

    Returns:
    - str: The BWT of the text.
    """
    bwt = []
    for suffix_idx in suffix_array(text):
        if suffix_idx == 0:
            bwt.append('$')
        else:
            bwt.append(text[suffix_idx-1])
    return ''.join(bwt)

class BWTransform:
    """
    A class encapsulating the functionalities related to Burrows-Wheeler Transform.

    Attributes:
    - bwt (str): The Burrows-Wheeler Transform of the given text.
    - bwm (list): The Burrows-Wheeler Matrix of the text.
    - suffix_array (iterator): The suffix array of the text.

    Methods:
    - inverse_transform(): Reconstructs the original text from its BWT.
    """

    def __init__(self, text):
        """
        Initialize the BWTransform object with the given text.

        Args:
        - text (str): The input text to be transformed.
        """
        self.bwt = bwt_from_sa(text)
        self.bwm = bwm(text)
        self.suffix_array = suffix_array(text)

    def __repr__(self):
        """
        Provide a string representation of the BWTransform object.
        """
        return self.bwt

    def inverse_transform(self):
        """
        Reconstructs the original text from its BWT using the inverse Burrows-Wheeler Transform.

        Returns:
        - str: The reconstructed original text.
        """
        table = [""]*len(self.bwt)
        for _ in range(len(self.bwt)):
            table = sorted([self.bwt[idx] + table[idx]
                           for idx in range(len(self.bwt))])
        return [row for row in table if row.endswith("$")].pop()
