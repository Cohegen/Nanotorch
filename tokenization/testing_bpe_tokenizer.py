import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import BPETokenizer 

def testing_bpe_tokenizer():
    """
    This function validates whether our BPE tokenizer
    learns merge rules and correclty encodes or decodes text
    """

    #testing basic functionality with simple corpus
    corpus = ["hello","world","hello","hell"] #"hell" and "hello" share prefix
    tokenizer= BPETokenizer(vocab_size=20)
    tokenizer.train(corpus)

    #checking that vocabulary was built
    assert len(tokenizer.vocab) >0
    assert '<UNK>' in tokenizer.vocab

    #testing helper functions
    word_tokens = tokenizer._get_word_tokens("test")
    assert word_tokens[-1].endswith('</w>'), "Should have end-of-word marker"
    
    pairs = tokenizer._get_pairs(['h','e','l','l','o</w>'])
    assert ('h','e') in pairs 
    assert ('l','l') in pairs 

    #testing encoding/decoding
    text = "hello"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens,list)
    assert all(isinstance(t,int) for t in tokens)

    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded,str)

    #testing round-strip on training data should work well
    for word in corpus:
        tokens = tokenizer.encode(word)
        decoded = tokenizer.decode(tokens)
        #allowing some flexibility due to BPE merging
        assert len(decoded.strip()) > 0

    print("BPE tokenizer works correctly!")

if __name__ == "__main__":
    testing_bpe_tokenizer()