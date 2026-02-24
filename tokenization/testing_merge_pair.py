import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from tokenization import _merge_pair 

def testing_merge_pair():
    """
    This function is intended to test In-place merging of
    a specific pair across all word tokens lists.

    """
    # Set up word tokens
    word_tokens = {
        "hello": ['h', 'e', 'l', 'l', 'o</w>'],
        "help": ['h', 'e', 'l', 'p</w>']
    }

    # Merge ('h', 'e') → 'he'
    merged = _merge_pair(word_tokens, ('h', 'e'))
    assert merged == 'he', f"Expected 'he', got '{merged}'"
    assert word_tokens["hello"] == ['he', 'l', 'l', 'o</w>'], \
        f"Expected ['he', 'l', 'l', 'o</w>'], got {word_tokens['hello']}"
    assert word_tokens["help"] == ['he', 'l', 'p</w>'], \
        f"Expected ['he', 'l', 'p</w>'], got {word_tokens['help']}"

    # Now merge ('l', 'l') → 'll' (only affects "hello")
    merged2 = _merge_pair(word_tokens, ('l', 'l'))
    assert merged2 == 'll', f"Expected 'll', got '{merged2}'"
    assert word_tokens["hello"] == ['he', 'll', 'o</w>'], \
        f"Expected ['he', 'll', 'o</w>'], got {word_tokens['hello']}"
    # "help" unchanged (no 'l','l' pair)
    assert word_tokens["help"] == ['he', 'l', 'p</w>'], \
        f"help should be unchanged, got {word_tokens['help']}"

    # Edge case: pair not present
    word_tokens_empty = {"ab": ['a', 'b</w>']}
    _merge_pair(word_tokens_empty, ('x', 'y'))
    assert word_tokens_empty["ab"] == ['a', 'b</w>'], "No-match merge should leave tokens unchanged"

    print("Byte pair merging is goooooooood!!!")

if __name__ == "__main__":
    testing_merge_pair()