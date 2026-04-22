import sys 
import os
from pathlib import Path
import numpy as np
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Tensor import Tensor
from activations.activations import GELU, Softmax, ReLU
from autograd.autograd import enable_autograd
from embeddings.embeddings import Embedding
from layers.layers import Linear, Dropout, Sequential, Layer
from losses.losses import CrossEntropyLoss
from optimizers.optimizers import AdamW
from transformers.transformers import LayerNorm

enable_autograd(quiet=True)

#parameters
batch_size = 16
block_size = 32
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
embed_dim = 96
num_heads = 3
num_layers = 3
head_dim = embed_dim // num_heads 
dropout = 0.2

import json
history = {"train": [], "val": [], "step": []}

#reading the dataset
with open('datasets/names.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#tokenization
stoi = {ch:i for i , ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s ]
decode = lambda l : ''.join([itos[i] for i in l])

#encoding the entire text dataset
data = Tensor(encode(text))
print(f"Dataset shape: {data.shape}")

#splitting data into training and validation sets
n = int(0.9*(data).shape[0])
train_data= data[:n]
val_data = data[n:]

#batch function
def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data_) - block_size, size=(batch_size,))
    x = np.stack([data_[i:i + block_size].data for i in ix])
    y = np.stack([data_[i + 1:i + block_size + 1].data for i in ix])
    
    return Tensor(x), Tensor(y)

class Head(Layer):
    """
    one head of self-attention
    """
    def __init__(self, head_dim):
        super().__init__()
        self.key = Linear(embed_dim, head_dim, bias=False)
        self.query = Linear(embed_dim, head_dim, bias=False)
        self.value = Linear(embed_dim, head_dim, bias=False)
        self.tril = np.tril(np.ones((block_size, block_size)))
        self.dropout = Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,head_dim)
        q = self.query(x) # (B,T,head_dim)

        # computing attention scores
        # (B,T,head_dim) @ (B,head_dim,T) -> (B,T,T)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        
        # masking
        mask = self.tril[:T, :T] == 0
        wei_data = wei.data.copy()
        wei_data[np.broadcast_to(mask, wei_data.shape)] = -float('inf')
        wei = Tensor(wei_data, requires_grad=wei.requires_grad)
        if hasattr(wei, '_grad_fn'):
             # We hacked the data, so we might have broken the graph if we didn't use Tensor ops
             # But since we want to mask, it's easier this way for now.
             pass

        softmax = Softmax()
        wei = softmax.forward(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B,T,head_dim)
        out = wei @ v # (B,T,T) @ (B,T,head_dim) -> (B,T,head_dim)
        return out 

    def parameters(self):
        return self.key.parameters() + self.query.parameters() + self.value.parameters()

class MultiHeadAttention(Layer):
    """
    Multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.heads = [Head(head_dim) for _ in range(num_heads)]
        self.proj = Linear(embed_dim, embed_dim)

    def train(self, mode=True):
        super().train(mode)
        for h in self.heads: h.train(mode)

    def eval(self):
        super().eval()
        for h in self.heads: h.eval()

    def forward(self, x):
        out_heads = [h(x) for h in self.heads]
        # Concatenate along the last dimension
        out_data = np.concatenate([h.data for h in out_heads], axis=-1)
        out = Tensor(out_data, requires_grad=any(h.requires_grad for h in out_heads))
        out = self.proj(out)
        return out

    def parameters(self):
        params = []
        for h in self.heads:
            params.extend(h.parameters())
        params.extend(self.proj.parameters())
        return params

class FeedForward(Layer):
    """
    A simple linear layer followed by non-linearity
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.net = Sequential(
            Linear(embed_dim, 4 * embed_dim),
            ReLU(),
            Linear(4 * embed_dim, embed_dim),
            Dropout(dropout)
        )

    def train(self, mode=True):
        super().train(mode)
        self.net.train(mode)

    def eval(self):
        super().eval()
        self.net.eval()

    def forward(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

class Block(Layer):
    """
    a single transformer block
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.sa = MultiHeadAttention(num_heads, head_dim)
        self.ffwd = FeedForward(embed_dim)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def train(self, mode=True):
        super().train(mode)
        self.sa.train(mode)
        self.ffwd.train(mode)

    def eval(self):
        super().eval()
        self.sa.eval()
        self.ffwd.eval()

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #residual connection
        x = x + self.ffwd(self.ln2(x)) #residual connection
        return x

    def parameters(self):
        return self.sa.parameters() + self.ffwd.parameters() + self.ln1.parameters() + self.ln2.parameters()

class NanoLanguageModel(Layer):
    def __init__(self, vocab_size, embed_dim, block_size, num_layers, num_heads):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size, embed_dim)
        self.position_embedding_table = Embedding(block_size, embed_dim)
        self.blocks = Sequential(*[Block(embed_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size)

    def train(self, mode=True):
        super().train(mode)
        self.blocks.train(mode)

    def eval(self):
        super().eval()
        self.blocks.eval()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(Tensor(np.arange(T))) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.reshape(B*T, C)
            targets_reshaped = targets.reshape(B*T)
            loss = CrossEntropyLoss().forward(logits_reshaped, targets_reshaped)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            softmax = Softmax()
            probs = softmax.forward(logits, dim=-1)
            
            # Sampling using numpy
            idx_next = []
            for i in range(probs.shape[0]):
                p = probs.data[i]
                # Ensure probabilities sum to 1
                p = p / np.sum(p)
                next_token = np.random.choice(len(p), p=p)
                idx_next.append(next_token)
            
            idx_next = Tensor(np.array(idx_next).reshape(-1, 1))
            idx_data = np.concatenate([idx.data, idx_next.data], axis=1)
            idx = Tensor(idx_data)
        return idx

    def parameters(self):
        params = self.token_embedding_table.parameters() + \
                 self.position_embedding_table.parameters() + \
                 self.blocks.parameters() + \
                 self.ln_f.parameters() + \
                 self.lm_head.parameters()
        return params

model = NanoLanguageModel(vocab_size, embed_dim, block_size, num_layers, num_heads)

@Tensor.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.data
        out[split] = losses.mean()
    model.train()
    return out

#creating optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        history["step"].append(iter)
        history["train"].append(float(losses['train']))
        history["val"].append(float(losses['val']))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with open('projects/Transformers/losses.json', 'w') as f:
    json.dump(history, f)

#Generate from model
context = Tensor(np.zeros((1, 1), dtype=np.int32))
print(decode(model.generate(context, max_new_tokens=100)[0].data.astype(int).tolist()))