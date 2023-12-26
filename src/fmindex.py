"""
This module implements the FM-index, an efficient and compact data structure that utilizes the 
Burrows-Wheeler Transform (BWT) for pattern matching in text. The FM-index is particularly useful 
in bioinformatics for tasks such as genome sequence alignment and search.

Classes:
- FmCheckpoints: Manages rank checkpoints in the BWT for fast rank queries.
- FmIndex: Implements the FM-index with functionality for substring search and pattern matching.

The module provides capabilities for pattern matching, including identifying substrings and suffixes
in the text, and locating all occurrences of a pattern.
"""

try:
    from bwt import suffix_array, bwt_from_sa
except ImportError:
    from src.bwt import suffix_array, bwt_from_sa

class FmCheckpoints(object):
    """
    Manages rank checkpoints in the Burrows-Wheeler Transform (BWT) and handles rank queries.
    Rank checkpoints are used to efficiently calculate the rank of characters in the BWT, 
    a fundamental operation in the FM-index.

    Attributes:
    - cps (dict): Checkpoints storing the cumulative count of each character.
    - cp_i_val (int): Spacing between checkpoints.

    Methods:
    - rank(bw, char, row): Returns the number of occurrences of 'char' in the BWT up to a given row.
    """

    def __init__(self, bw, cp_i_val=4):
        """
        Initializes the FmCheckpoints object by scanning the BWT and creating periodic checkpoints.

        Args:
        - bw (str): The Burrows-Wheeler Transform of a string.
        - cp_i_val (int, optional): Interval between checkpoints. Defaults to 4.
        """
        self.cps = {}        # Checkpoints for each character
        self.cp_i_val = cp_i_val # Spacing between checkpoints
        char_count = {}       # Counter for occurrences of each character

        # Initialize character count and checkpoints for each unique character in BWT
        for char in set(bw):
            char_count[char] = 0
            self.cps[char] = []

        # Build checkpoints at regular intervals
        for idx, char in enumerate(bw):
            char_count[char] += 1
            if (idx % cp_i_val) == 0:
                for key,item in char_count.items():
                    self.cps[key].append(item)

    def rank(self, bw, char, row):
        """
        Returns the number of occurrences of 'char' in the BWT up to and including a given row.

        Args:
        - bw (str): The Burrows-Wheeler Transform of a string.
        - char (str): Character to count occurrences of.
        - row (int): Row in the BWT up to which the count is computed.

        Returns:
        - int: Number of occurrences of 'char' up to and including 'row'.
        """
        if row < 0 or char not in self.cps:
            return 0

        idx, num_occ = row, 0
        # Calculate rank by walking left (up) and counting occurrences of 'char'
        while (idx % self.cp_i_val) != 0:
            if bw[idx] == char:
                num_occ += 1
            idx -= 1

        return self.cps[char][idx // self.cp_i_val] + num_occ

class FmIndex():
    """
    Implements an FM-index for a given text, enabling efficient substring searches. The FM-index uses
    the BWT of the text along with additional data structures like rank checkpoints and a downsampled
    suffix array for various queries.

    Attributes:
    - bwt (str): Burrows-Wheeler Transform of the input text.
    - sampled_sa (dict): Downsampled suffix array for resolving positions.
    - slen (int): Length of the BWT.
    - cps (FmCheckpoints): Object managing rank checkpoints.
    - first (dict): Concise representation of the first column in BWT.

    Methods:
    - count(c): Returns the number of occurrences of characters less than 'c'.
    - range(p): Finds the range of rows in the BWT that match a given prefix 'p'.
    - resolve(row): Converts a row in the BWT to its corresponding suffix array index.
    - has_sub_string(p): Checks if 'p' is a substring of the text.
    - has_suffix(p): Checks if 'p' is a suffix of the text.
    - occurrences(p): Finds all occurrences of 'p' in the text.
    """

    @staticmethod
    def downsamplesuffix_array(sa, n=4):
        """
        Creates a downsampled version of the suffix array. Retains only every nth entry to reduce space complexity.

        Args:
        - sa (list): The original suffix array.
        - n (int): Downsampling factor; retains every nth entry.

        Returns:
        - dict: A map from row indices in the BWT to their corresponding suffix array values.
        """
        sampled_sa = {}
        for i,sai in enumerate(sa):
            # We could use i % n instead of sa[i] % n, but we lose the
            # constant-time guarantee for resolutions
            if sai % n == 0:
                sampled_sa[i] = sai
        return sampled_sa

    def __init__(self, text, cp_i_val=4):
        """
        Initializes the FM-index for the given text. Adds a terminal character ('$') if not present,
        computes the BWT and initializes other required data structures.

        Args:
        - text (str): The text to index.
        - cp_i_val (int, optional): The interval between rank checkpoints and suffix array samples.
        """
        if text[-1] != '$':
            text += '$' # add dollar if not there already
        # Get BWT string and offset of $ within it
        sa = list(suffix_array(text))
        self.bwt = bwt_from_sa(text)
        # Get downsampled suffix array, taking every 1 out of 'cp_i_val'
        # elements w/r/t T
        self.sampled_sa = self.downsamplesuffix_array(sa, cp_i_val)
        self.slen = len(self.bwt)
        # Make rank checkpoints
        self.cps = FmCheckpoints(self.bwt, cp_i_val)
        # Calculate # occurrences of each character
        char_count = dict()
        for char in self.bwt:
            char_count[char] = char_count.get(char, 0) + 1
        # Calculate concise representation of first column
        self.first = {}
        idx = 0
        for char, count in sorted(char_count.items()):
            self.first[char] = idx
            idx += count

    def count(self, c):
        ''' Return number of occurrences of characters < c '''
        if c not in self.first:
            # (Unusual) case where c does not occur in text
            for char in sorted(self.first.keys()):
                if c < char:
                    return self.first[char]
            return self.first[char]
        else:
            return self.first[c]

    def range(self, p):
        """
        Computes the range of rows in the BWT that match a given prefix. Utilizes backward searching
        leveraging the FM-index properties.

        Args:
        - pr (str): The prefix to search for in the BWT.

        Returns:
        - tuple: A pair (l, r) indicating the range of rows matching the prefix.
        """
        l, r = 0, self.slen - 1 # closed (inclusive) interval
        for idx in range(len(p)-1, -1, -1): # from right to left
            l = self.cps.rank(self.bwt, p[idx], l-1) + self.count(p[idx])
            r = self.cps.rank(self.bwt, p[idx], r)   + self.count(p[idx]) - 1
            if r < l:
                break
        return l, r+1

    def resolve(self, row):
        """
        Converts a row in the BWT to its corresponding offset in the original text. This method
        is essential for locating the actual positions of patterns found in the text.

        Args:
        - row (int): The row index in the BWT.

        Returns:
        - int: The corresponding offset in the original text.
        """
        def step_left(row):
            """
            A helper function to step left in the BWT. It calculates the next row to move to
            during the resolution process.

            Args:
            - row (int): The current row index in the BWT.

            Returns:
            - int: The next row index to move to.
            """
            char = self.bwt[row]
            return self.cps.rank(self.bwt, char, row-1) + self.count(char)
        nsteps = 0
        while row not in self.sampled_sa:
            row = step_left(row)
            nsteps += 1
        return self.sampled_sa[row] + nsteps

    def has_sub_string(self, p):
        """
        Checks if a given pattern 'p' is a substring of the indexed text. This is achieved
        by checking if there is a non-empty range for the pattern in the BWT.

        Args:
        - p (str): The pattern to search for in the text.

        Returns:
        - bool: True if 'p' is a substring, False otherwise.
        """
        l, r = self.range(p)
        return r > l

    def has_suffix(self, s):
        """
        Checks if a given string 's' is a suffix of the indexed text. It first finds the range of 
        BWT rows that match the suffix 's' and then verifies if the resolved position of the first 
        occurrence (leftmost in the range) corresponds to the position where the suffix should start 
        in the original text.

        Args:
        - s (str): The suffix to search for in the text.

        Returns:
        - bool: True if 's' is a suffix of the indexed text, False otherwise.
        """
        l, r = self.range(s)  # Find the range of rows in the BWT that match the suffix 's'
        off = self.resolve(l)  # Resolve the first occurrence in the range to its position in the original text
        # Check if the suffix 's' starts at the correct position (end of the text)
        return r > l and off + len(s) == self.slen - 1

    def occurrences(self, p):
        """
        Finds all occurrences of a pattern 'p' in the indexed text. It returns a list of
        starting positions where the pattern occurs.

        Args:
        - p (str): The pattern to search for in the text.

        Returns:
        - list: A list of integers representing the starting positions of 'p' in the text.
        """
        l, r = self.range(p)
        return [ self.resolve(x) for x in range(l, r) ]
