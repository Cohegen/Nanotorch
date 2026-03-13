import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#importing dependencies from other modules
import numpy as np
from activations.activations import GELU 
from autograd.autograd import Function
from embeddings.embeddings import EmbeddingLayer
from layers.layers import Layer, Linear 
from Tensor import Tensor
from attention.attention import MultiHeadAttention

#constants for memory calculations 
BYTES_PER_FLOAT32 = 4 #standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024 #megabytes to bytes conversion


def create_causal_maks(seq_len:int)-> Tensor:
    """
    A helper function that creates a causal(autoregressive) attention mask

    This mask ensures that position i can only attend to positions j where j<= i
    This essential for autoregressive language models like GPT.

    Args: 

       seq_len: Length of the sequence

    Returns: 
        Tensor of shape (1,seq_len,seq_len) with:
        - 1.0 for positions that CAN be attended to (lower triangle)
        - 0.0 for positions that CANNOT be attende to (upper triangle)

    Example:
        For seq_len=4, creates:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    #lower triangular matrix (1=can attended, 0= cannot attend)
    mask = np.tril(np.ones((seq_len,seq_len),dtype=np.float32))
    return Tensor(mask[np.newaxis,:,:]) #add batch dimension

    
class _LayerNormBackward(Function):
    """
    Gradient Computation for the full layer normalization operation

    Computes gradients for x, gamma and beta in one pass.
    output = gamma* ((x-mean)/ std) + beta

    The gradient for x uses the standard LayerNorm formula:
        dx = (gamma/std) * (grad - mean(grad) - normalized * mean(grad * normalized))

    """

    def __init__(self,x,gamma,beta,normalized_data,std_data):
        """
        Initializing with forward pass values needed for gradient computation
        """
        super().__init__(x,gamma,beta)
        self.normalized_data = normalized_data
        self.std_data = std_data

    def apply(self,grad_output):
        """
        Computes gradients for LayerNorm (x,gamma,beta)
        """
        x,gamma,beta = self.saved_tensors

        grad_x = grad_gamma = grad_beta = None
        normalized = self.normalized_data
        std_data = self.std_data

        #Gradient for beta: sum over all dims except last
        if isinstance(beta,Tensor) and beta.requires_grad:
            #sum over batch and sequence dimensions
            grad_beta = grad_output.copy()
            while grad_beta.ndim > 1:
                grad_beta = grad_beta.sum(axis=0)

        #Gradient for gamma: sum of (grad_output *normalized) over batch/seq dims
        if isinstance(gamma,Tensor) and gamma.requires_grad:
            grad_gamma = (grad_output * normalized).copy()
            while grad_gamma.ndim > 1:
                grad_gamma = grad_gamma.sum(axis=0)

        #Gradient for x: full LayerNorm backward formula
        if isinstance(x,Tensor) and x.requires_grad:
            #grad flowing through gamma: grad_output * gamma
            gamma_data = gamma.data if isinstance(gamma,Tensor) else gamma
            grad_norm = grad_output * gamma_data 

            mean_grad = np.mean(grad_norm,axis=1,keepdims=True)
            mean_grad_norm = np.mean(grad_norm * normalized,axis=-1,keepdims=True)
            grad_x = (1.0/std_data) *(grad_norm - mean_grad - normalized * mean_grad_norm)

        return (grad_x,grad_gamma,grad_beta)

class LayerNorm:
    """
    Layer Normalization for transformer blocks.

    It normalizes across the feature dimension (lst axis) for each sample independently,

    """

    def __init__(self,normalized_shape,eps=1e-5):
        """
        Intializing LayerNorm with learnable parameters.
        """
        self.normalized_shape = normalized_shape
        self.eps =eps 

        #Learnable parameters: scale and shift 
        self.gamma = Tensor(np.ones(normalized_shape)) #scale parameter 
        self.beta = Tensor(np.zeros(normalized_shape)) # shift parameter 

    
    def forward(self,x):
        """
        Applies layer normalization

        MATHEMATICAL FORMULA:
         y = (x - μ) / σ * γ + β
        where μ = mean(x), σ = sqrt(var(x) + ε)
        """

        #computing statistics across last dimension (features)
        mean_data = np.mean(x.data,axis=1,keepdims=True)

        #computing variance : E[(x - μ)²]
        diff = x.data - mean_data 
        variance = np.mean(diff *diff,axis=-1,keepdims=True)

        #Normalize: (x-mean) / sqrt(variance + eps)
        std_data = np.sqrt(variance + self.eps)
        normalized_data = diff/std_data 

        #Applying learnable transformation : gamma * normalized + beta 
        output_data = self.gamma.data * normalized_data + self.beta.data 
        output = Tensor(output_data)

        #Attaching gradient function for full LayerNorm backward
        if x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad:
            output.requires_grad = True 
            output._grad_fn = _LayerNormBackward(
              x,self.gamma,self.beta,normalized_data,std_data   
            )

        return output 

    def __call__(self,x):
        """Allows the layer norm to be called like a function"""
        return self.forward(x)

    def parameters(self):
        """Return learnable parameters"""
        return [self.gamma,self.beta]


class MLP:
    """
    Multi-Layer Perceptron (FFN) for transformer blocks.

    Standard pattern:Linear -> GELU -> Linear with expansion ration 4:1
    """

    def __init__(self,embed_dim,hidden_dim=None,dropout_prob=0.1):
        """
        Initialize MLP with two linear layer

        EXAMPLE:
        >>> mlp = MLP(512)  # Will create 512 -> 2048 -> 512 network
        >>> x = Tensor(np.random.randn(2, 10, 512))
        >>> output = mlp.forward(x)
        >>> assert output.shape == (2, 10, 512)
        """
        if hidden_dim is None:
            hidden_dim = 4* embed_dim # standard 4x expansion

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim 

        #two-layer feed-forward network 
        self.linear1 = Linear(embed_dim,hidden_dim)
        self.gelu = GELU() #use GELU activation from activations module
        self.linear2 = Linear(hidden_dim,embed_dim)

    def forward(self,x):
        """
        Forward pass through MLP

         COMPUTATION FLOW:
        x -> Linear -> GELU -> Linear -> output
        """
        #first linear layer with expansion
        hidden = self.linear1.forward(x)

        #GELU activation
        hidden = self.gelu.forward(hidden)

        #second Linear layer back to original size
        output = self.linear2.forward(hidden)

        return output

    def __call__(self,x):
        """Allows the MLP to be called like a function (forward pass)."""
        return self.forward(x)

    def parameters(self):
        """Returns all learnable parameters."""
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params

class TransformerBlock:
    """
    Complete Transformer Block with self-attention, MLP and residual connections
    """
    def __init__(self,embed_dim,num_heads,mlp_ratio=4,ff_dim=None,droout_prob=0.1):
        """
        Intializes a complete transformer block

         TRANSFORMER BLOCK ARCHITECTURE:
        x → LayerNorm → MultiHeadAttention → + (residual) →
            LayerNorm → MLP → + (residual) → output

        
         EXAMPLE:
        >>> block = TransformerBlock(embed_dim=512, num_heads=8)
        >>> x = Tensor(np.random.randn(2, 10, 512))  # (batch, seq, embed)
        >>> output = block.forward(x)
        >>> assert output.shape == (2, 10, 512)
        """
        self.embed_dim = embed_dim 
        self.num_heads = num_heads

        #Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim,num_heads)

        #layer normalization (pre-norm architecture)
        self.layer_norm1 = LayerNorm(embed_dim) #before attention
        self.layer_norm2 = LayerNorm(embed_dim) #before MLP

        #Feed-forward network
        #support both mlp_ration and explicit ff_dim for backward compatibility
        if ff_dim is not None:
            hidden_dim = ff_dim
        else:
            hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim,hidden_dim)


    def forward(self,x,mask=None):
        """
        Forward pass through transformer block

        COMPUTATION FLOW:
        x -> layer_norm1 -> +x -> layer_norm2 -> ml -> + -> output

        RESIDUAL CONNECTIONS:
        These are crucial for training deep networks- they allow gradients
        to flow directly through the network during backpropagation
        """
        #first sub-layer: Multi-head self-attention with residual connection
        #pre-norm: LayerNorm before attention
        normed1 = self.layer_norm1.forward(x)
        #self-attention:quiey,key,value are all the same (normed1)
        attention_out = self.attention.forward(normed1,mask)

        #residual connection
        x = x+ attention_out

        #second sub-layer :MLP with residual connection
        #pre-norm:LayerNorm before MLP
        normed2 = self.layer_norm2.forward(x)
        mlp_out = self.mlp.forward(normed2)

        #residual connection 
        output = x + mlp_out 

        return output 

    def __call__(self,x,mask=None):
        """
        Allows the transformer block to called like a function
        """
        return self.forward(x,mask)


    def parameters(self):
        """
        Return all learnable parameters
        """
        params = []
        params.extend(self.attention.parameters())
        params.extend(self.layer_norm1.parameters())
        params.extend(self.layer_norm2.parameters())
        params.extend(self.mlp.parameters())

        return params 

class GPT:
    """
    Compplete GPT(Generative Pre-Trained Transformer) Model

    """

    def __init__(self,vocab_size,embed_dim,num_layers,num_heads,max_seq_len=1024):
        """
        Intializes the Complete GPT model


        GPT ARCHITECTURE:
        tokens → embedding → + pos_embedding →
                transformer_blocks → layer_norm → lm_head → logits

        EXAMPLE:
        >>> model = GPT(vocab_size=1000, embed_dim=256, num_layers=6, num_heads=8)
        >>> tokens = Tensor(np.random.randint(0, 1000, (2, 10)))  # (batch, seq)
        >>> logits = model.forward(tokens)
        >>> assert logits.shape == (2, 10, 1000)  # (batch, seq, vocab)
        """
        self.vocab_size = vocab_size 
        self.embed_dim = embed_dim 
        self.num_layers = num_layers 
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        #Embedding layer 
        self.embedding_layer = EmbeddingLayer(vocab_size,embed_dim,max_seq_len)

        #stack of transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(embed_dim,num_heads)
            self.blocks.append(block)

        #final layer normalization
        self.layer_norm_final = LayerNorm(embed_dim)

        #language modelling heads (projects to vocabulary)
        self.lang_modelling = Linear(embed_dim,vocab_size,bias=False)

    def forward(self,tokens):
        """
        Forward pass through GPT model


        COMPUTATATION FLOW:
        tokens -> embed + pos_embed -> blocks-> layer_norm_final ->lang_modelling -> logits

       
        """
        batch_size,seq_len = tokens.shape 

        #passing tokens to embedding layer get token embeddings and positional embeddings
        x = self.embedding_layer.forward(tokens)

        #creating a causal mask for autoregresive generation
        mask = self._create_causal_mask(seq_len) 

        #passing through transformer blocks
        for block in self.blocks:
            x = block.forward(x,mask)

        #final layer normalization
        x = self.layer_norm_final.forward(x)

        #language moddelling head 
        logits = self.lang_modelling.forward(x)

        return logits 

    def __call__(self,tokens):
        """
        Allows GPT model to be called like a function
        """
        return self.forward(tokens)

    def _create_causal_mask(self,seq_len):
        """
        Create causal mask to prevent attending to future position
        """
        #upper trianglar matrix matrix filled with -inf
        mask = np.triu(np.ones((seq_len,seq_len))* -np.inf,k=1)
        return Tensor(mask)

    def _sample_next_token(self,logits,temperature=1.0):
        """
        Sample one token from vocabulary logits using temperature scaling

         EXAMPLE:
        >>> logits = np.array([[1.0, 2.0, 3.0]])  # Raw model output
        >>> token = model._sample_next_token(logits, temperature=1.0)
        >>> assert 0 <= token < 3  # Valid token index
        """
        #applying temperature scaling
        scaled_logits = logits /temperature

        #convert to probabilities (softmax with numerical stability)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits,axis=-1,keepdims=True))
        probs = exp_logits / np.sum(exp_logits,axis=-1,keepdims=True)

        #sample next token from probability distribution
        next_token = np.random.choice(self.vocab_size,p=probs[0])
        return next_token

    def generate(self,prompt_tokens,max_new_tokens=50,temperature=1.0):
        """
        Generate text autoregressively by repeatedly sampling next tokens

         EXAMPLE:
        >>> model = GPT(vocab_size=100, embed_dim=64, num_layers=2, num_heads=4)
        >>> prompt = Tensor([[1, 2, 3]])  # Some token sequence
        >>> generated = model.generate(prompt, max_new_tokens=5)
        >>> assert generated.shape[1] == 3 + 5  # original + new tokens

        """
        current_tokens = Tensor(prompt_tokens.data.copy())

        for _ in range(max_new_tokens):
            #getting logits for current sequence
            logits = self.forward(current_tokens)

            #get  logits for last positions(next token prediction)
            last_logits = logits.data[:,-1,:] #(batch_size,vocab_size)

            #sample next token using helper
            next_token_id = self._sample_next_token(last_logits,temperature)

            #Append to sequence
            next_token = np.array([[next_token_id]])
            current_tokens = Tensor(np.concatenate([current_tokens.data,next_token],axis=1))

        return current_tokens

    def parameters(self):
        """Returns all learnable parameters"""
        params = []
        params.extend(self.embedding_layer.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.layer_norm_final.parameters())
        params.extend(self.lang_modelling.parameters())

        return params 

