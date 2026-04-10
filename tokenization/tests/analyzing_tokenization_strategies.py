import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import BPETokenizer,CharTokenizer,create_tokenizer,analyze_tokenization


def analyze_tokenization_strategies():
    """
    This function compares different tokenization strategies on
    various texts.
    """
    print("Analysis on different Tokenization Strategies...")
    print("="*60)

    #creating test corpus with different text types
    corpus = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog",
        "Machine Learning is transforming artificial intelligence",
        "Tokenization is fundemental to natural language processing",
        "Subword units balance vocabulary size and sequence length"

    ]

    #testing different strategies
    strategies = [
        ("Character",create_tokenizer("char",corpus=corpus)),
        ("BPE-100",create_tokenizer("bpe",vocab_size=100,corpus=corpus)),
        ("BPE-500",create_tokenizer("bpe",vocab_size=500,corpus=corpus))
    ]

    print(f"{'Strategy':<12} {'Vocab':<8} {'Avg Len':<8} {'Compression':<12} {'Coverage':<10}")
    print("-" * 60)

    for name, tokenizer in strategies:
        stats = analyze_tokenization(corpus, tokenizer)

        print(f"{name:<12} {stats['vocab_size']:<8} "
              f"{stats['avg_sequence_length']:<8.1f} "
              f"{stats['compression_ratio']:<12.2f} "
              f"{stats['unique_tokens']:<10}")


    print("\n KEY INSIGHTS:")
    print("   1. Character tokenization: Small vocab, long sequences, perfect coverage")
    print("   2. BPE: Larger vocab trades off with shorter sequences")
    print("   3. Higher compression ratio = more characters per token = efficiency")

    print("\n REAL-WORLD IMPLICATIONS:")
    print("   - GPT-3/4 uses ~50K BPE tokens for balance")
    print("   - Character models need more compute (longer sequences)")
    print("   - Embedding table size scales with vocabulary size")


    print("\n","="*60)

if __name__ == "__main__":
    analyze_tokenization_strategies()