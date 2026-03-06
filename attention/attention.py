import os 
from re import T
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy 
import math 
import time 
from typing import Optional,Tuple,List 

#import dependecies from other modules
from Tensor import Tensor 
from layers.layers import Linear
from activations.activations import Softmax

#constant for attention computation
MASK_VALUE = -1e9 # large negative value used for attention masking since it becomes ~0 after softmax


def _compute_attention_scores(Q:Tensor,K:Tensor)-> Tensor:
    """
    A helper function that computes Attention scores

    Score Computation (Q@K^T):
      For each query position i and key position j:
      score[i,j] = sum(Q[i,d]x k[j,d]) for d in embedding_dims 

    """
    #transposing K i.e swapping the last two dims so (batch,seq,d) becomes (batch,d,seq)
    K_t = K.transpose(-2,-1)
    return Q.matmul(K_t)


def _scale_scores(scores:Tensor,d_model:int)->Tensor:
    """
    A helper function that scaling scores

    Raw dot products grow proportionally with dimension size. For d_model=512,
    scores would be ~500x larger than for d_model=1 which pushes softmax into extreme
    values where most weight falls on a single token.
    Dividing by sqrt(d_model) keeps scores in a stable range regardless
    of dimension.

    Scale attention scores by 1/sqrt(d_model)
    """
    #computing scale factor
    scale_factor = 1.0 / math.sqrt(d_model)
    return scores * scale_factor


def _apply_mask(scores:Tensor,mask:Tensor)-> Tensor:
  """
  Applies causal mask by setting masked positions  -infinity
  """

  #computing additive mask
  adder = (1.0 - mask.data)*MASK_VALUE
  return scores + Tensor(adder)


def scaled_dot_product_attention(Q:Tensor,K:Tensor,V:Tensor,mask:Optional[Tensor]=None)-> Tuple[Tensor,Tensor]:
  """
  A helper function with a complete dot-product attention.

  Args: 
       Q: Query tensor of shape (batch_szie,seq_len,d_model)
       K: Key tensor of shape (batch_size,seq_len,d_model)
       V: Value tensot of shape (batch_szie,seq_len,d_model)
       mask: Optional causal mask, True=allow,False=mask (batch_size,seq_len,seq_len)
  
  Returns:
      output:Attended values (batch_szie,seq_len,d_model)
      attention_weights: Attention matrix (batch_size,seq_len,seq_len)
  """
  #calling _compute_attention_scores for Q and K
  scores = _compute_attention_scores(Q,K)
  scores = _scale_scores(scores,Q.shape[-1])
  if mask is not None:
    scores = _apply_mask(scores,mask)
  softmax = Softmax()
  attention_weights = softmax(scores,dim=-1)
  output = attention_weights.matmul(V)
  return output, attention_weights 


class MultiHeadAttention:
  """
  This is Multi-Head attention mechanism class

  It runs multiple attention heads in parallel, each learning different relationships.
  """

  def __init__(self,embed_dim:int,num_heads:int):
    """
    Intializing multi-head attention
    """
    #validating whether embed_dim is divisible by num_heads
    if embed_dim % num_heads !=0:
      raise ValueError(
        f"Multiple-head attention dimension mismatch\n"
        f"    embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
        f"     Multi-head attention splits embed_dims equally maong heads, so embed_dim must be a multiple of num_heads\n"
        f"       Try: embed_dim={num_heads * (embed_dim // num_heads + 1)} ( next valid size) or num_heads={embed_dim // {embed_dim // num_heads}} (fewer heads)"

      )

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    #Linear projection for queries,keys,values
    self.q_proj = Linear(embed_dim,embed_dim)
    self.k_proj = Linear(embed_dim,embed_dim)
    self.v_proj = Linear(embed_dim,embed_dim)

    #output projection to mix information across heads
    self.out_proj = Linear(embed_dim,embed_dim)

  def _split_heads(self,x:Tensor,batch_size:int,seq_len:int) -> Tensor:
    """
    Reshapes to seperate attention heads for parallel processing
    """
    x = x.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
    return x.transpose(1,2)



  
  def _merge_heads(self,x:Tensor,batch_size:int,seq_len:int)-> Tensor:
    """
    This method merges attention heads back into single embedding dimension.
    After each head computes its own attention independently, we need to recombine
    them back into a single embedding. This is the reverse of splitting,whereby we 
    transpose the head and sequence dimension back,the reshape to merge (heads,head_dim)
    into a single embed_dim

    example: embed_dim=64,num_heads=8,head_dim=8:
         (2,8,10,8) -> transpose -> (2,10,8,8) -> reshape -> (2,10,64)
                                     batch seq heads dim       batch seq embed_dim 
     """
    x = x.transpose(1,2)
    return x.reshape(batch_size,seq_len,self.embed_dim)
    

  def forward(self,x:Tensor,mask:Optional[Tensor]=None)->Tensor:
    """
    Forward pass through multi-head attention

    Args:
       x:Input tensor (batch_size,seq_len,embed_dim)
       mask:Optional attention mask (batch_size,seq_len,seq_len)

      Returns:
         output:Attended representation (batch_szie,seq_len,seq_len)

    """
    #extracting dimension and validate 
    batch_size,seq_len,embed_dim = x.shape 
    if embed_dim != self.embed_dim:
      raise ValueError(
        f"Multi-HeadAttention input dimension mismatch\n"
        f"     Expected embed_dim={self.embed_dim}, got {embed_dim} from input shape {x.shape}"
        f"     The last dimension of input must match embed_dim from intialization(MultiHeadAttention({self.embed_dim},{self.num_heads}))\n"
        f"      Try: x.reshape({x.shape[0]},{x.shape[1]},{self.embed_dim}) or create new MultiHeadAttention({embed_dim},num_head) "
      )

    # projecting input to Q,K,V
    Q = self.q_proj.forward(x)
    K = self.k_proj.forward(x)
    V = self.v_proj.forward(x)

    # splitting into heads
    Q = self._split_heads(Q,batch_size,seq_len)
    K = self._split_heads(K,batch_size,seq_len)
    V = self._split_heads(V,batch_size,seq_len)

    # Applying attention (reshaping mask for head broadcasting)
    mask_reshaped = mask 
    if mask is not None and len(mask.shape) == 3:
      batch_size_mask,seq_len_mask,_ = mask.shape
      mask_data = mask.data.reshape(batch_size_mask,1,seq_len_mask,seq_len_mask)
      mask_reshaped = Tensor(mask_data)

    attended,_ = scaled_dot_product_attention(Q,K,V,mask=mask_reshaped)

    # Merging heads back together
    concat_output = self._merge_heads(attended,batch_size,seq_len)

    # apply output projection
    output = self.out_proj.forward(concat_output)

    return output 

  def __call__(self,x:Tensor,mask:Optional[Tensor]=None)-> Tensor:
    """
    Making MultiHeadAttention callable like attention(x)
    """
    return self.forward(x,mask)

  def parameters(self) ->List[Tensor]:
    """
    Returns all trainable parameters 
    """
    params: List[Tensor] = []
    params.extend(self.q_proj.parameters())
    params.extend(self.k_proj.parameters())
    params.extend(self.v_proj.parameters())
    params.extend(self.out_proj.parameters())
    return params 