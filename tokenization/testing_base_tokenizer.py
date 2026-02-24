from ast import Pass
import os
#from sre_parse import Tokenizer
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import Tokenizer 

def testing_base_tokenizer():
    """
    This function is intended to validate that our base tokenizer
    defines the correct interface for all implementations.
    """
    #testing that the base class defines the interface
    tokenizer = Tokenizer()

    #should raise NotImplementedError for both methods
    try:
        tokenizer.encode("test")
        assert False, "encode() should raise NotImplementedError"
    except NotImplementedError:
        pass

    try:
        tokenizer.decode([1,2,3])
        assert False,"decode() should raise NotImplementedError"
    except NotImplementedError:
        pass 

if __name__ == "__main__":
    testing_base_tokenizer()