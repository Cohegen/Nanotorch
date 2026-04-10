import os 
import sys
from typing import assert_type

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import BPETokenizer, CharTokenizer 
from tokenization import create_tokenizer,tokenize_dataset,analyze_tokenization

def testing_tokenization_utils():
    """
    This function validates whether our utility functions for
    tokenizer creation,dataset processing and analysis work
    """

    #testing tokenizer factory 
    corpus = ["hello world","test text","more examples"]

    char_tokenizer = create_tokenizer("char",corpus=corpus)
    assert isinstance(char_tokenizer,CharTokenizer)
    assert char_tokenizer.vocab_size > 0

    bpe_tokenizer = create_tokenizer('bpe',vocab_size=50,corpus=corpus)
    assert isinstance(bpe_tokenizer,BPETokenizer)

    #testing dataset tokenization

    texts = ["hello","world","test"]
    tokenized = tokenize_dataset(texts,char_tokenizer,max_length=10)
    assert len(tokenized) == len(texts)
    assert all(len(seq) <=10 for seq in tokenized)

    #testing analysis
    stats = analyze_tokenization(texts,char_tokenizer)
    assert 'vocab_size' in stats 
    assert 'avg_sequence_length' in stats 
    assert 'compression_ratio' in stats 
    assert stats['total_tokens'] > 0

    print("Tokenization utils work bueeeeeenoo!")

if __name__ =="__main__":
    testing_tokenization_utils()