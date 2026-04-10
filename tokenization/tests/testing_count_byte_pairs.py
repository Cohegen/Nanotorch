import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from tokenization import _count_byte_pairs

def testing_count_byte_pairs():
    """
    This function intends to test the frequency-weighted
    pair counting across multiple words

    """
    #Two word "hello" appears 3 times, "help" appears 1 time
    word_tokens = {
         "hello": ['h', 'e', 'l', 'l', 'o</w>'],
        "help": ['h', 'e', 'l', 'p</w>']
    }
    word_freq = Counter({"hello":3,"help":1})

    counts = _count_byte_pairs(word_tokens,word_freq)

     # ('h','e') appears in both words: 3 + 1 = 4
    assert counts[('h', 'e')] == 4, f"Expected 4, got {counts[('h', 'e')]}"

    # ('e','l') appears in both words: 3 + 1 = 4
    assert counts[('e', 'l')] == 4, f"Expected 4, got {counts[('e', 'l')]}"

    # ('l','l') appears only in "hello" (freq 3)
    assert counts[('l', 'l')] == 3, f"Expected 3, got {counts[('l', 'l')]}"

    # ('l','p</w>') appears only in "help" (freq 1)
    assert counts[('l', 'p</w>')] == 1, f"Expected 1, got {counts[('l', 'p</w>')]}"

    # Empty case
    empty_counts = _count_byte_pairs({}, Counter())
    assert len(empty_counts) == 0

    print("Byte pair counting works correctly!")

if __name__ == "__main__":
    testing_count_byte_pairs()