import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import CharTokenizer 

def testing_char_tokenizer():
    """
    This function intends to validate whether the
    CharTokenizer 
    """

    #testing basic functionality
    vocab = ['h','e','l','o',' ','w','r','d']
    tokenizer = CharTokenizer(vocab)

    #testing vocabulary setup
    assert tokenizer.vocab_size == 9 # 8 chars + UNK
    assert tokenizer.vocab[0] == '<UNK>'
    assert 'h' in tokenizer.char_to_id

    #testing encoding
    text = "hello"
    tokens = tokenizer.encode(text)
    expected = [1,2,3,3,4] #h,e,l,l,o (based on actual vocab order)
    assert tokens == expected, f"Expected {expected}, got {tokens}"

    #testing decoding
    decoded = tokenizer.decode(tokens)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"

    #testing unknown character handling
    tokens_with_unk = tokenizer.encode("hello!")
    assert tokens_with_unk[-1] == 0 # '!' should map to <unk>

    # Test vocabulary building
    corpus = ["hello world", "test text"]
    tokenizer.build_vocab(corpus)
    assert 't' in tokenizer.char_to_id
    assert 'x' in tokenizer.char_to_id

    print("Characters tokenizer works correctly!")

if __name__ == "__main__":
    testing_char_tokenizer()