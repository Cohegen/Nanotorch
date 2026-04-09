import os
import sys 
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import inspect 
import math
from dataclasses import dataclass
from typing import Dict, List

from Tensor.tensor import Tensor 
from activations.activations import GELU, Softmax
from embeddings.embeddings import Embedding
from losses.losses import CrossEntropyLoss
from layers.layers import Layer, Linear, Dropout, Parameter, Sequential
from attention.attention import scaled_dot_product_attention
from transformers.transformers import LayerNorm
from optimizers.optimizers import AdamW

class CausalSelfAttention(Layer):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        # output projection
        self.c_proj = Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        # regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout
        
        # causal mask to ensure attention is only on the past
        self.register_buffer("bias", Tensor(np.tril(np.ones((1, 1, config.block_size, config.block_size)))))

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, value for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # manual attention implementation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        attn = Softmax().forward(att, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(Layer):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.embed_dim, 4 * config.embed_dim, bias=config.bias)
        self.gelu = GELU()
        self.c_proj = Linear(4 * config.embed_dim, config.embed_dim, bias=config.bias)
        self.dropout = Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(Layer):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 5000
    num_layers: int = 4
    embed_dim: int = 128
    dropout: float = 0.0
    num_heads: int = 4
    bias: bool = True 

class GPT(Layer):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config 

        self.transformer = Layer() # Container for transformer modules
        self.transformer.wte = Embedding(config.vocab_size, config.embed_dim)
        self.transformer.wpe = Embedding(config.block_size, config.embed_dim)
        self.transformer.drop = Dropout(config.dropout)
        self.transformer.h = Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.transformer.ln_f = LayerNorm(config.embed_dim)
        
        self.lm_head = Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def get_num_params(self, non_embedding=True):
        """
        Returns the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = Tensor(np.arange(0, t, dtype=np.int64)) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, embed_dim)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, embed_dim)
        x = self.transformer.drop(tok_emb + pos_emb)
        x = self.transformer.h(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # Flatten logits and targets for CrossEntropyLoss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = CrossEntropyLoss().forward(logits_flat, targets_flat)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None 

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (Tensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                # Simple top-k implementation with numpy
                v = np.partition(logits.data, -top_k, axis=-1)[:, -top_k:]
                min_v = np.min(v, axis=-1, keepdims=True)
                logits.data[logits.data < min_v] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = Softmax().forward(logits, dim=-1)
            
            # sample from the distribution
            idx_next_data = []
            for i in range(probs.shape[0]):
                p = probs.data[i]
                # normalize just in case
                p = p / np.sum(p)
                ix = np.random.choice(len(p), p=p)
                idx_next_data.append([ix])
            
            idx_next = Tensor(np.array(idx_next_data))
            # append sampled index to the running sequence and continue
            idx_data = np.concatenate((idx.data, idx_next.data), axis=1)
            idx = Tensor(idx_data)

        return idx
