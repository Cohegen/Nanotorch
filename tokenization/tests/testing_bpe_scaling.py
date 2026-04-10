import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import KB_TO_BYTES 
import random
import string 
from tokenization import BPETokenizer,create_tokenizer

def analyze_bpe_scaling():
    """
    This functions intends to show us how BPE Training Time grows

    """
    print("Analyze BPE training scaling...")
    print("="*70)

    #generating random text helper
    def generate_random_text(length=100):
        return  ''.join(random.choices(string.ascii_lowercase + ' ', k=length))

    corpus_sizes = [100,500,1000,2500]

    print(f"{'Corpus Size':<15} {'Training Time (ms)':<20} {'Vocab Size':<15} {'Memory (KB)':<15}")
    print("-" * 70)

    for size in corpus_sizes:
        #generating corpus 
        corpus = [generate_random_text(length=15) for _ in range(size)]

        #measuring training time and memory 
        import tracemalloc
        import time 
        tracemalloc.start()

        start = time.perf_counter()
        tokenizer =BPETokenizer(vocab_size=500)
        tokenizer.train(corpus,vocab_size=500)
        train_time = (time.perf_counter() - start)*100

        memory_kb = tracemalloc.get_traced_memory()[1] /KB_TO_BYTES
        tracemalloc.stop()

        print(f"{size:<15} {train_time:<20.1f} {len(tokenizer.vocab):<15} {memory_kb:<15.1f}")

if __name__ == "__main__":
    analyze_bpe_scaling()