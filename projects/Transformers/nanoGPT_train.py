import os
import sys
import time
import math
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Tensor.tensor import Tensor
from autograd.autograd import enable_autograd
from tokenization.tokenization import CharTokenizer
from dataloader.dataloader import TensorDataset, Dataloader
from projects.Transformers.nanoGPT_model import GPT, GPTConfig
from optimizers.optimizers import AdamW

# Enable autograd for training
enable_autograd(quiet=True)

# -----------------------------------------------------------------------------
# Configuration
batch_size = 16
block_size = 32 # context length
max_iters = 100
learning_rate = 1e-3
device_type = 'cpu'
eval_interval = 20
eval_iters = 10

# Model configuration
config = GPTConfig(
    block_size = block_size,
    vocab_size = None, # will be set from tokenizer
    num_layers = 4,
    num_heads = 4,
    embed_dim = 128,
    dropout = 0.0,
    bias = True,
)

# -----------------------------------------------------------------------------
# Data loading
data_path = os.path.join(ROOT, 'datasets', 'names.txt')
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenizer
tokenizer = CharTokenizer()
tokenizer.build_vocab([text])
vocab_size = tokenizer.vocab_size
config.vocab_size = vocab_size

print(f"Vocabulary size: {vocab_size}")

# Encode data
data = tokenizer.encode(text)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data_split) - block_size, (batch_size,))
    x = np.stack([data_split[i:i+block_size] for i in ix])
    y = np.stack([data_split[i+1:i+block_size+1] for i in ix])
    return Tensor(x), Tensor(y)

# -----------------------------------------------------------------------------
# Model initialization
model = GPT(config)

# Optimizer initialization
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device_type)

# -----------------------------------------------------------------------------
# Training loop
@Tensor.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.data
        out[split] = losses.mean()
    model.train()
    return out

print("Starting training...")
t0 = time.time()
for iter in range(max_iters):

    # evaluate the loss on train/val sets and write checkpoints
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # update parameters
    optimizer.step()

    if iter % 10 == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter}: loss {loss.data:.4f}, time {dt*1000:.2f}ms")

# -----------------------------------------------------------------------------
# Generation test
print("\nGenerating some names...")
model.eval()
context = Tensor(np.zeros((1, 1), dtype=np.int64)) # start with first token (usually <UNK> or newline)
generated = model.generate(context, max_new_tokens=20)
print(tokenizer.decode(generated.data[0].astype(int).tolist()))
