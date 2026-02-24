import os
import sys
import tokenize

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter 
from typing import Dict,List,Optional,Set,Tuple 
import numpy as np

#Constants for memory calculations
KB_TO_BYTES = 1024 # kilobytes to conversion 

"""
## Tokenization implementation
Here we will implement tokenization system step by step
and testing them later.

## Tokenization Class Architecture:
```
Tokenization System Structure:
┌─────────────────────────────────┐
│ Base Tokenizer Interface:       │
│ • encode(text) → token_ids      │
│ • decode(token_ids) → text      │
├─────────────────────────────────┤
│ CharTokenizer (Simple):         │
│ • vocab: list of characters     │
│ • char_to_id: lookup mapping    │
│ • id_to_char: reverse mapping   │
├─────────────────────────────────┤
│ BPETokenizer (Advanced):        │
│ • vocab: learned subwords       │
│ • merges: learned pair rules    │
│ • token_to_id/id_to_token maps  │
├─────────────────────────────────┤
│ Utility Functions:              │
│ • create_tokenizer()            │
│ • tokenize_dataset()            │
│ • analyze_tokenization()        │
└─────────────────────────────────┘
```

### Base Tokenizer Interface
All tokenizers need to provide two core operations: encoding text to numbers and
decoding the numbers back to text.

```
Tokenizer Interface:
     encode(text) -> [id1,id2,id3,...]
     decode([id1,id2,id3,...]) -> text
```
"""

class Tokenizer:
    """
    Base tokenizer class providing the interface for all tokenizers

    This defines the guidelines that all tokenizers must follow:
    - encode() : text -> list of token Ids
    - decode(): list of token IDs -> text
    """

    def encode(self,text:str) -> List[int]:
        """
        Convert text to a list of token IDs.
        
        subclasses will override this method

        """
        raise NotImplementedError(
            f"encode() not implemented in base Tokenizer class\n"
            f"  Called encode() on abstract base class {self.__class__.__name__}\n"
            f"   Tokenizer is an interface, it uses a concrete implementation like CharTokenizer or BPETokenizer\n"
            f"     Example: tokenizer = CharTokenizer(['a','b','c]);tokenizer.encode('abc')"

        )
    def decode(self,tokens:List[int]) -> str:
        """
        Converts list of token IDs back to text

        Subclasses will override this method
        It returns reconstructed text string
        """
        raise NotImplementedError(
            f"decode() not implemented in base Tokenizer class\n"
            f"     Called decode() on abstract base class {self.__class__.__name__}\n"
            f"       Tokenizer is an interface- use a concrete implementation like CharTokenizer or BPETokenizer\n"
            f"         Example:tokenizer=ChatTokenizer(['a','b','c']); tokenizer.decode([0,1,2])"

        )


class CharTokenizer(Tokenizer):
    """"
    This class intends to implement the character tokenization process.
    Explanation for CharacterTokenization is available in tokenization.md
    """

    def __init__(self,vocab:Optional[List[str]]=None):
        """
        Intializing character tokenizer
        """
        #storing vocabularly list
        if vocab is None:
            vocab = []

        ##Adding a special unkwown token
        self.vocab = ['<UNK>'] + vocab 
        self.vocab_size = len(self.vocab)

        #creating bidirectional mappings
        self.char_to_id = {char:idx for idx,char in enumerate(self.vocab)}
        self.id_to_char = {idx:char for idx,char in enumerate(self.vocab)}

        #storing unknwon token ID
        self.unk_id = 0

    def build_vocab(self,corpus:List[str])-> None:
        """
        This method builds vocabulary from corpus of text.
        """
        #collecting all unique characters 
        all_chars = set()
        for text in corpus:
            all_chars.update(text)

        #sorting for consistent ordering
        unique_chars = sorted(all_chars)

        #Rebuilding vocabulary with <UNK> token first
        self.vocab = ['<UNK>'] + unique_chars
        self.vocab_size = len(self.vocab)

        #rebuilding mappings
        self.char_to_id = {char:idx for idx,char in enumerate(self.vocab)}
        self.id_to_char = {idx:char for idx,char in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """
        This method encodeds text to list of character IDs.
        """
        tokens = []
        #loop to iterate through each character in text
        for char in text:
            tokens.append(self.char_to_id.get(char,self.unk_id))
        return tokens

    def decode(self,tokens:List[int]) -> str:
        """
        This method decodes list of tokens IDs back to text
        """
        chars = []
        for token_id in tokens:
            #using unknown for invalid IDs
            char = self.id_to_char.get(token_id,'<UNK>')
            chars.append(char)
        return ''.join(chars)


def _count_byte_pairs(word_tokens:Dict[str,List[str]],word_freq:Counter) -> Counter:
     """
     This function counts frquency of all adjacent token pairs
     across all words.

     Each pair's count is weighted by how often its containing word appears
     in the corpus, so frequent words contributes more to pair stastics
     """

     pair_counts = Counter()

     #iterating through each word and its frequency
     for word,freq in word_freq.items():
        tokens = word_tokens[word]
        #counting adjacent pairs 
        for i in range(len(tokens)-1):
            pair = (tokens[i],tokens[i+1])
            pair_counts[pair] += freq 

     return pair_counts

def _merge_pair(word_tokens:Dict[str,List[str]],pair:Tuple[str,str]) -> str:
    """
    This function Merges the most frequent pair in all word token lists.

    It scan through every word's tokens and replaces adjacent occurences
    of the pairs with a single concatenated token.
    It then modifies word_tokens in place and returns the new merged token string.

    """
    merged_token = pair[0] + pair[1]

    for word in word_tokens:
        tokens = word_tokens[word]
        new_tokens = []
        i =0
        while i < len(tokens):
            if (i<len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]):
                #merging frequent pair
                new_tokens.append(merged_token)
                i += 2 
            else:
                new_tokens.append(tokens[i])
                i += 1
        word_tokens[word] = new_tokens

    return merged_token


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer that leeanrs subword unit.

    This BPE implementation works by:
       1.Starting with character level vocabulary
       2.Finding most frequent character pairs 
        3.Merging frequent pairs into single tokens
       4.Repeating unit desired vocabulary size
    """

    def __init__(self,vocab_size:int=1000) :
        """
        Intializing BPE tokenizer
        """
        #storing target vocabulary size
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = [] # lists of (pair,new token) merges
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_word_tokens(self,word:str) -> List[str]:
        """
        This method converts word to list of 
        characters with end-of-word marker.
        """

        if not word:
            return []

        tokens = list(word)
        tokens[-1] += '</w>'
        return tokens

    def _get_pairs(self,word_tokens:List[str]) -> Set[Tuple[str,str]]:
        """
        This method gets all adjacent pairs form word tokens.
        """

        pairs = set()
        #iterating through adjacent tokens
        for i in range(len(word_tokens)-1):
            pairs.add((word_tokens[i],word_tokens[i+1]))
        return pairs 

    def train(self,corpus:List[str],vocab_size:int=None) -> None:
        """
        This method trains BPE on corpus to learn merge rules.

        This is the composition function; it initializes character vocabulary,
        the runs a greedy merge loop using _count_byte_pairs() to find the
        best pair and _merge_pair() to apply it.
        """
        if vocab_size:
            self.vocab_size = vocab_size

        #count word frequencies and intialize character vocabulary
        word_freq = Counter(corpus)
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        self.vocab = sorted(vocab)
        if '<UNK>' not in vocab:
            self.vocab = ['<UNK>'] + self.vocab 

        #Greedy merge loop follows count pairs -> merge best -> repeat procedure
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            pair_counts = _count_byte_pairs(word_tokens,word_freq)
            if not pair_counts:
                break 

            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = _merge_pair(word_tokens,best_pair)
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        self._build_mappings()

    def _build_mappings(self):
        """
        This methos builds token-to-ID and ID-to-token mappings.
        """
        self.token_to_id = {token:idx for idx,token in enumerate(self.vocab)}
        self.id_to_token = {idx:token for idx,token in enumerate(self.vocab)}

    def _apply_merges(self,tokens:List[str]) -> List[str]:
        """
        This method applies learned merge rules
        to token sequence.
        """
        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and tokens[i] == merge_pair[0] and tokens[i+1] == merge_pair[1]):
                    # applying merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens 


    def encode(self,text:str) -> List[int]:
        """
        This method encodes text using BPE
        """ 
        if not self.vocab:
            return []

        #simple word splitting 
        words = text.split()
        all_tokens = []

        for word in words:
            #getting character-level tokens
            word_tokens = self._get_word_tokens(word)

            #applying BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        #converting to IDs
        tokens_ids = []
        for token in all_tokens:
            tokens_ids.append(self.token_to_id.get(token,0))

        return tokens_ids

    def decode(self,tokens:List[int]) -> str:
        """
        This method decodes IDs back to text

        """
        if not self.id_to_token:
            return ""

        #converting IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id,'<UNK>')
            token_strings.append(token)

        #joining and cleaninig up
        text = ''.join(token_strings)

        #replacing end-of-word markers with spaces
        text = text.replace('</w>',' ')

        #cleaning up extra spaces 
        text = ''.join(text.split())

        return text 
  
def create_tokenizer(strategy:str="char",vocab_size:int =1000,corpus:List[str]=None)-> Tokenizer:
    """
    This is a factory function that creates and trains tokenizers.

    """
    #checking strategy
    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus,vocab_size)
    else:
        raise ValueError(
            f"Unknown tokenization strategy: '{strategy}'\n"
            f"   Strategy '{strategy}: ' is not recognized\n"
            f" NanoTorch supports 'char' (character-level) and 'bpe' (byte-pair encoding) strategies\n"
            f"    To fix this use: create_tokenizer('char',corpus=texts) or create_tokenizer('bpe',vocab_size=1000,corpus=texts)"
        )

    return tokenizer

def tokenize_dataset(texts:List[str],tokenizer:Tokenizer,max_length:int = None)-> List[List[int]]:
    """
    This function tokenizes a dataset with optional length limits
    """
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        #applying length limit
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized

def analyze_tokenization(texts:List[str],tokenizer:Tokenizer)->Dict[str,float]:
    """
    This function analyzes tokenization statistics
    """
    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    #calculating statistics
    tokenized_lengths = [len(tokenizer.encode(text))for text in texts]

    stats = {
        'vocab_size':tokenizer.vocab_size,
        'avg_sequence_length':np.mean(tokenized_lengths),
        'max_sequence_length':max(tokenized_lengths) if tokenized_lengths else 0,
        'total_tokens':len(all_tokens),
        'compression_ratio':total_chars / len(all_tokens) if all_tokens else 0,
        'unique_tokens':len(set(all_tokens))


    }

    return stats 


