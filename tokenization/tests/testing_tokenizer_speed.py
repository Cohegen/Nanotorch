import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import BPETokenizer,CharTokenizer,create_tokenizer
import time 


def analyze_tokenization_speed():
    """
    This function analyzes encoding-decoding speeds of various tokenizers.

    We will realize that Character Tokenization is Fastest since it has a simple dictionary lookup of O(n) complexity.
    BPE tokenization is slower since it requires merge rule learning and application.

    Large BPE vocab means fewer final tokens but more merge operations
    """
    print("General Analysis of Tokenization Speed...")
    print("="*70)

    #preparing test data (1000,text varying lengths)
    test_texts = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "machine learning is transforming artificial intelligence",
        "tokenization enables natural language processing in neural networks"
    ]*250 #1000 total texts

    #building tokenizer on training corpus
    corpus = test_texts[:100]
    tokenizers = [
        ("Character",create_tokenizer("char",corpus=corpus)),
        ("BPE-500",create_tokenizer("bpe",vocab_size=500,corpus=corpus)),
        ("BPE-2000",create_tokenizer("bpe",vocab_size=2000,corpus=corpus))

    ]
    print(f"{'Strategy':<12} {'Encode (ms)':<15} {'Decode (ms)':<15} {'Total Tokens':<15}")
    print("-" * 70)

    for name,tokenizer in tokenizers:
        #benchmarking encoding
        start = time.perf_counter()
        all_tokens = [tokenizer.encode(text) for text in test_texts]
        encode_time = (time.perf_counter() - start)*1000

        #benchmarking decoding
        start = time.perf_counter()
        decoded = [tokenizer.decode(tokens) for tokens in all_tokens]
        decode_time = (time.perf_counter()-start) * 1000

        total_tokens = sum(len(t) for t in all_tokens)

        print(f"{name:<12} {encode_time:<15.1f} {decode_time:<15.1f} {total_tokens:<15}")


if __name__ == "__main__":
    analyze_tokenization_speed()