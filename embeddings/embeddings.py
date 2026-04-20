import numpy as np
import math 
from typing import List,Optional,Tuple
import os 
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor.tensor import BYTES_PER_FLOAT32, KB_TO_BYTES, MB_TO_BYTES
from autograd.autograd import Function,enable_autograd
enable_autograd()
from Tensor import Tensor

#constants for memory calculations
BYTES_PER_FLOAT32 = 4 #standard float32 size in bytes
KB_TO_BYTES = 1024 #kilobytes to bytes conversion
MB_TO_BYTES= 1024 * 1024 #megabytes to bytes conversion


class EmbeddingBackward(Function):
    """
    This class computes gradients for embedding lookup operation\

    If Y = Embedding[indices] the dLoss/dEmbedding[i] =
    sum of all gradients where index ==i

    
    """

    def __init__(self,weight,indices):
        """
        Args:
            weight:Embedding weight matrix
            indices:indices used for lookup
        """

        super().__init__(weight)
        self.indices = indices

    def apply(self,grad_output):
        """
        Computes gradients for embedding lookup

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
              Tuple with single gradient for weight tensor
              
        **Mathematical Foundation:**
        - ∂(Embedding[indices])/∂Embedding = scatter gradients to selected rows
        - Multiple indices can point to same embedding → gradients accumulate

        """
        #extracting weight tensors from self.tensors
        weight, = self.saved_tensors
        grad_weight = None

        if isinstance(weight,Tensor) and weight.requires_grad:
            #intializing gradient with zeros
            grad_weight = np.zeros_like(weight.data)

            #scattering gradient back to embedding weights
            #np.add.at accumulates gradients for repeated indices
            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1,grad_output.shape[-1])

            np.add.at(grad_weight,indices_flat,grad_output_reshaped)

        return (grad_weight,)


from layers.layers import Layer, Parameter

class Embedding(Layer):
    """
    Leanable embedding layer that maps token indices to dense vectors.

    This is the fundemental building block for converting discrete tokens into
    continous representations that neural networks can process.
    """
    def __init__(self,vocab_size:int,embed_dim:int):
        """
        Initializing embedding layer with Xavier-uniform weights.

        Args: 
            vocab_size: size of vocabulary i.e number of unique tokens
            embed_dim:dimension of embedding vectors

        """
        super().__init__()
        # store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Xavier Glorot initialization for better gradient flow
        limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Parameter(
            np.random.uniform(
                -limit, limit, (vocab_size, embed_dim)
            )
        )

    def forward(self,indices:Tensor) -> Tensor:
        """
        Forward pass: lookup embedding for given indices.

        Args: 
             indices:Token indices of shape (batch_size,seq_len) or (seq_len,)

        Returns:
             Embedded vector of shape (*indices.shape,embed_dim)

        """
        # validating whether indices are within the range [0, vocab_size)
        if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
            min_idx = int(np.min(indices.data))
            max_idx = int(np.max(indices.data))
            raise ValueError(
                f"Embeddings index out of range for vocabulary size {self.vocab_size}\n"
                f"Found indices: min={min_idx},max={max_idx} (valid range: 0 to {self.vocab_size-1})\n"
                f"Token IDs must be within the vocabulary.IDs >= vocab_size reference non-existent tokens\n"
                f"Check your tokenizer output,or increase vocab_size to atleast {max_idx +1}"
            )

        # Performing embedding lookup using advanced indexing
        # This is equivalent to one-hot multiplication but much more efficient
        embedded = self.weight.data[indices.data.astype(int)]

        result = Tensor(embedded)

        # Attaching gradient function for backpropagation
        # EmbeddingBackward will handle sparse gradient accumulation
        if self.weight.requires_grad:
            result.requires_grad = True
            result._grad_fn = EmbeddingBackward(self.weight, indices)

        return result 

    def __call__(self,indices:Tensor)-> Tensor:
        """
        Allows the embedding to be called like a function.
        """
        return self.forward(indices)

    def parameters(self) -> List[Tensor]:
        """
        Returns trainable parameters.
        """
        return [self.weight]

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size},embed_dim={self.embed_dim})"

class PositionalEncoding:
    """
    Learnable positional encoding layer

    Adds trainable position-specific vector to token embeddings,
    allowing the model to learn positional patterns specific to the task
    """

    def __init__(self,max_seq_len:int,embed_dim:int):
        """
        Intializes learnable positional encoding

        Args: 
            max_seq_len:maximum sequence length to support
            embed_dim:Embedding dimension which must match token embeddings
        """

        #storing max_seq_len and embed_dim
        self.max_seq_len = max_seq_len 
        self.embed_dim = embed_dim 

        #intializing position mebedding matrix
        #smaller initialization than token embeddings sincw these are additive
        limit = math.sqrt(2.0/embed_dim)
        self.position_embeddings = Tensor(
            np.random.uniform(-limit,limit,(max_seq_len,embed_dim))
        )


    def forward(self,x:Tensor) -> Tensor:
        """
        Add postional encodings to input embeddings

        Args:
            x: Input embeddings of shape (batch_size,seq_len,embed_dim)

        Returns:
            Position-encoded embeddings of same shape
        """

        #validating whether input is 3D 
        if len(x.shape) == 2:
            raise ValueError(
                f"Expected 3D input (batch,seq,embed), got 2D: {x.shape}\n"
                f"  Missing batch dimension\n"
                f"  PositionalEncoding expects batched embeddings, not single sequences\n"
                f"Add batch dim: x.reshape(1, {x.shape[0],{x.shape[1]}})"

            )
        elif len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch,seq,embed), got {len(x.shape)}D: {x.shape}\n"
                f"  Input must have exactly 3 dimensions\n"
                f" PositionalEncoding expects shape (batch_size,sequence_length,embedding_dim)"

            )

        batch_size, seq_len,embed_dim = x.shape 

        #validating whether our input's sequence length exceeds max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeds maximum: {seq_len} > {self.max_seq_len}\n"
                f"   Input sequence has {seq_len} positions, but max_seq_len is {self.max_seq_len}\n"
                f"    Learned positional encoding have a fixed maximum length set at intialization\n"
                f"     Either truncate input to {self.max_seq_len} tokens, or create a new PositionalEncoding(max_seq_len={seq_len},...)"

            )

        #validating whether input embedding dimensions is equal to self.embed_dim
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f"   PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n "
                f"    Token embeddings and positional encodings must have the same dimension to be added together\n"
                f"    Ensure your Embedding layer uses embed_dim={self.embed_dim}, or create PositionalEncoding(embed_dim={embed_dim},...)"

            )
        
        #slicing positional embeddings for this sequence lengths using Tensor slicing
        pos_embeddings = self.position_embeddings[:seq_len] #(seq_len,embed_dim)

        #reshaping it to add dimension: (1,seq_len,embed_dim)
        pos_data = pos_embeddings.data[np.newaxis,:,:]
        pos_embeddings_batched = Tensor(pos_data)

        #Adding postional information
        result = x + pos_embeddings_batched

        return result 


    def __call__(self,x:Tensor) -> Tensor:
        """Allows the positional encoding to called like a function"""
        return self.forward(x)

    def parameters(self)->List[Tensor]:
        """Return trainable parameters"""
        return [self.position_embeddings]

    def __repr__(self):
            return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"

def _compute_sinusoidal_table(max_len:int,embed_dim:int)-> np.ndarray:
    """
    Computes the raw sinusoidal positional encoding table as a numpy array

    This helper function builds the (max_len,embed_dim) table of sin/cos values
    using the formula from the "Attention is All you need" paper.

     PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))
    """

    #creating position indices [0,1,2,...,max_len-1]
    position = np.arange(max_len,dtype=np.float32)[:,np.newaxis] #(max_len,1)

    #creating dimension indices for calculating frequencies
    div_term = np.exp(
        np.arange(0,embed_dim,2,dtype=np.float32)*
        -(math.log(10000.0)/ embed_dim) 
    ) #(embed_dim//2,)

    #intializing the positional encoding matrix
    pe = np.zeros((max_len,embed_dim),dtype=np.float32)

    #applying sine to even indices(0,2,4,6...)
    pe[:,0::2] = np.sin(position *div_term)

    #applying cosine to odd indices (1,3,5,7...)
    if embed_dim % 2 == 1:
        #handling odd embed_dim by only filling available positions
        pe[:,1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:,1::2] = np.cos(position * div_term)

    return pe

def create_sinusoidal_embeddings(max_seq_len:int,embed_dim:int)->Tensor:
    """
    This function creates sinusoidal positional encodings

    These fixed encodings use sine and cosine functions to create unique
    positional patterns that don't require training and can extraoloate
    to longer sequences than seen during training.
    """
    #calling _compute_sinusoidal_table
    pe = _compute_sinusoidal_table(max_seq_len,embed_dim)

    #wrapping pe in a Tensor
    return Tensor(pe)


class EmbeddingLayer:
    """
    Complete embedding system that combines token and positional embeddings

    """

    def __init__(
        self,vocab_size:int,
        embed_dim:int,
        max_seq_len:int=512,
        pos_encoding:str = 'learned',
        scale_embeddings:bool =False
    ) :
        """
        This __init__ method assembles the sub-compenents: a token `Embedding`
        for vocabulary lookup and one of three positional encoding strategies
        """

        ##storing configuration i.e (vocab_size,embed_dim,max_seq_len)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding
        self.scale_embeddings = scale_embeddings

        #Token embedding layer 
        self.token_embedding = Embedding(vocab_size,embed_dim)

        #positional encoding
        if pos_encoding == 'learned':
            self.pos_encoding = PositionalEncoding(max_seq_len,embed_dim)
        elif pos_encoding == 'sinusoidal':
            #create fixed sinusoidal encodings
            self.pos_encoding = create_sinusoidal_embeddings(max_seq_len,embed_dim)
        elif pos_encoding is None:
            self.pos_encoding = None 
        else:
            raise ValueError(
                f"Unknown positional encoding type: '{pos_encoding}'\n"
                f"   pos_encoding must be 'learnable', 'sinusoidal', or None\n"
                f"      'learnable'=trainable position embeddings (task-specific but max length)\n"
                f"       'sinusoidal'=mathematical sin/cos patterns (no parameters, can extrapolate)\n"
                f"     None= no positional encoding (order-agnostic model)\n"
                f"       Use: EmbeddingLayer(...,pos_encoding='learnable') or pos_encoding='sinusoidal'"
            )

    def __call__(self,tokens:Tensor) -> Tensor:
        """Allows the embedding layer to be called like a function"""
        return self.forward(tokens)

    def parameters(self)->List[Tensor]:
        """Returns all trainable parameters"""
        params = self.token_embedding.parameters()
        if self.pos_encoding_type == 'learned':
            params.extend(self.pos_encoding.parameters())
        return params 

    def __repr__(self):
        return (f"EmbeddingLayer(vocab_size={self.vocab_size}, "
                f"embed_dim={self.embed_dim}, "
                f"pos_encoding='{self.pos_encoding_type}')")

def emblayer_forward(self,tokens:Tensor)-> Tensor:
    """
    Forward pass through complete embedding system
    This method composes the full embedding pipeline i.e token lookup,
    optional scaling,positional encoding addition and batch dimension handling
    """
    #handling 1D input by adding batch dimension
    if len(tokens.shape)==1:
        tokens = tokens.reshape(1,-1)
        squeeze_batch = True 
    else:
        squeeze_batch = False 

    #Get token embeddings
    token_embeds = self.token_embedding.forward(tokens)  #(batch,seq,embed)

    #scale embeddings if user requests it 
    if self.scale_embeddings:
        scale_factor = math.sqrt(self.embed_dim)
        # apply scaling in-place to keep using the same variable
        token_embeds = token_embeds * scale_factor  # Tensor multiplication preserves gradients 

    #add positional encoding
    if self.pos_encoding_type == 'learned':
        #using learnable positional encoding 
        output = self.pos_encoding.forward(token_embeds)
    elif self.pos_encoding_type == 'sinusoidal':
        #using fixed sinusoidal encoding
        batch_size,seq_len,embed_dim = token_embeds.shape 
        pos_embeddings = self.pos_encoding[:seq_len]#slicing with Tensor slicing

        #reshaping to add batch dimension
        pos_data = pos_embeddings.data[np.newaxis,:,:]
        pos_embeddings_batched = Tensor(pos_data) 

        output = token_embeds + pos_embeddings_batched 

    else:
        #in scenario with no positional encoding 
        output = token_embeds

    #removing batch dimension if needed
    if squeeze_batch:
        #we use Tensor slicing
        output = output[0]

    return output 

#Attaching forward to EmbeddingLayer class
EmbeddingLayer.forward = emblayer_forward  