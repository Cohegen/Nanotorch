import os 
import sys
import tracemalloc 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor.tensor import KB_TO_BYTES
from tokenization import BPETokenizer,CharTokenizer,create_tokenizer

def analyze_tokenization_memory():
    """
    This function helps us to know the actual
    amount of memory used by the Tokenizer.

    From this we come to realize that Char Tokenizer uses minimal memory because of its
    small vocab which roughly 100 tokens.
    BPE uses more memory with large vocab and merge rules which eat up storage.

    Memory scales with vocabulary size
    """
    print("Testing Tokenization Memory Usage...")
    print("="*70)

    #creating testing corpora of varying sizes
    corpus_small = ["hello world"]*100
    corpus_medium = ["the quick brown fox jumps over the lazy dog"]*1000
    corpus_large = ["machine learning processes natural language text"]*5000

    results = []
    for corpus_name,corpus in [
        ("Small(100)",corpus_small),
        ("Medium(1K)",corpus_medium),
        ("Large(5K)",corpus_large)]:
        #character tokenizer memory
        tracemalloc.start()
        char_tok = CharTokenizer()
        char_tok.build_vocab(corpus)
        char_current,char_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        #BPE tokenizer memory 
        tracemalloc.start()
        bpe_tok = BPETokenizer(vocab_size=1000)
        bpe_tok.train(corpus,vocab_size=1000)
        bpe_current,bpe_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            'corpus':corpus_name,
            'char_kb':char_peak /KB_TO_BYTES,
            'bpe_kb':bpe_peak /KB_TO_BYTES,
            'char_vocab': char_tok.vocab_size,
            'bpe_vocab':len(bpe_tok.vocab)
        }   
        )

        #displaying results 
        print(f"{'Corpus':<15} {'Char Mem (KB)':<15} {'BPE Mem (KB)':<15} {'Char Vocab':<12} {'BPE Vocab':<12}")
        print("-" * 70)

        for r in results:
            print(f"{r['corpus']:<15} {r['char_kb']:<15.1f} {r['bpe_kb']:<15.1f} "
              f"{r['char_vocab']:<12} {r['bpe_vocab']:<12}")

if __name__ == "__main__":
    analyze_tokenization_memory()
        