import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# importing dependencies from other modules
from activations.activations import GELU
from autograd.autograd import Function
from embeddings.embeddings import EmbeddingLayer
from layers.layers import Layer, Linear, Parameter, Sequential
from Tensor import Tensor
from attention.attention import MultiHeadAttention

def create_causal_mask(seq_len: int) -> Tensor:
    """
    A helper function that creates a causal(autoregressive) attention mask.
    1.0 for positions that CAN be attended to (lower triangle)
    -inf for positions that CANNOT be attended to (upper triangle)
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    # Standard attention mask: 0 for allow, -inf for block
    # But MultiHeadAttention in this repo might expect 1/0 or handle -inf.
    # Let's check MultiHeadAttention implementation.
    return Tensor(mask)

class _LayerNormBackward(Function):
    """
    Gradient Computation for the full layer normalization operation.
    """
    def __init__(self, x, gamma, beta, normalized_data, std_data):
        super().__init__(x, gamma, beta)
        self.normalized_data = normalized_data
        self.std_data = std_data

    def apply(self, grad_output):
        x, gamma, beta = self.saved_tensors
        normalized = self.normalized_data
        std_data = self.std_data
        
        axis = -1
        N = x.data.shape[axis]

        # grad_beta
        grad_beta = grad_output.copy()
        while grad_beta.ndim > 1:
            grad_beta = grad_beta.sum(axis=0)

        # grad_gamma
        grad_gamma = (grad_output * normalized).copy()
        while grad_gamma.ndim > 1:
            grad_gamma = grad_gamma.sum(axis=0)

        # grad_x
        if x.requires_grad:
            gamma_data = gamma.data
            dx_normalized = grad_output * gamma_data
            grad_x = (1.0 / (N * std_data)) * (
                N * dx_normalized - 
                np.sum(dx_normalized, axis=axis, keepdims=True) - 
                normalized * np.sum(dx_normalized * normalized, axis=axis, keepdims=True)
            )
        else:
            grad_x = None

        return grad_x, grad_gamma, grad_beta

class LayerNorm(Layer):
    """
    Layer Normalization for transformer blocks.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = Parameter(np.ones(normalized_shape))
        self.beta = Parameter(np.zeros(normalized_shape))

    def forward(self, x):
        axis = -1
        mean = np.mean(x.data, axis=axis, keepdims=True)
        var = np.var(x.data, axis=axis, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x.data - mean) / std
        
        output_data = self.gamma.data * x_norm + self.beta.data
        result = Tensor(output_data)
        
        if x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad:
            result.requires_grad = True
            result._grad_fn = _LayerNormBackward(x, self.gamma, self.beta, x_norm, std)
            
        return result

class MLP(Layer):
    """
    Multi-Layer Perceptron (FFN) for transformer blocks.
    """
    def __init__(self, embed_dim, hidden_dim=None, dropout_prob=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.gelu = GELU()
        self.linear2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

class TransformerBlock(Layer):
    """
    Complete Transformer Block with self-attention, MLP and residual connections.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, ff_dim=None, dropout_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm1 = LayerNorm(embed_dim)
        self.layer_norm2 = LayerNorm(embed_dim)
        
        if ff_dim is None:
            ff_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attention(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x

class GPT(Layer):
    """
    Complete GPT (Generative Pre-Trained Transformer) Model.
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)
        self.blocks = Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.layer_norm_final = LayerNorm(embed_dim)
        self.lang_modelling = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens):
        seq_len = tokens.shape[1]
        x = self.embedding_layer(tokens)
        mask = self._create_causal_mask(seq_len)
        x = self.blocks(x, mask)
        x = self.layer_norm_final(x)
        return self.lang_modelling(x)

    def _create_causal_mask(self, seq_len):
        # 0 for allow, -1e9 for block (MultiHeadAttention usually expects this for additive mask)
        mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        return Tensor(mask)

    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        current_tokens = prompt_tokens
        for _ in range(max_new_tokens):
            logits = self.forward(current_tokens)
            last_logits = logits.data[:, -1, :] / temperature
            
            # Softmax
            exp_logits = np.exp(last_logits - np.max(last_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            next_token_id = np.random.choice(probs.shape[-1], p=probs[0])
            next_token = Tensor(np.array([[next_token_id]]))
            current_tokens = Tensor(np.concatenate([current_tokens.data, next_token.data], axis=1))
        return current_tokens
