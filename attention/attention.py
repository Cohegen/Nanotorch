import math
import os
import sys
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tensor import Tensor
from activations.activations import Softmax
from layers.layers import Linear

# constant for attention computation
MASK_VALUE = -1e9  # large negative value used for attention masking since it becomes ~0 after softmax
DEFAULT_FLASH_BLOCK_SIZE = 128
DEFAULT_SPARSE_WINDOW_SIZE = 128
DEFAULT_PAGE_SIZE = 128


def _to_4d_attention_tensors(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Convert 3D attention tensors into 4D head-aware form and report whether a head axis was added.
    """
    q_data = Q.data
    k_data = K.data
    v_data = V.data
    squeeze_head = False

    if q_data.ndim == 3:
        q_data = q_data[:, np.newaxis, :, :]
        k_data = k_data[:, np.newaxis, :, :]
        v_data = v_data[:, np.newaxis, :, :]
        squeeze_head = True

    if q_data.ndim != 4 or k_data.ndim != 4 or v_data.ndim != 4:
        raise ValueError("Attention variants expect 3D or 4D Q, K, and V tensors")

    return q_data, k_data, v_data, squeeze_head


def _restore_attention_output(output: np.ndarray, squeeze_head: bool) -> Tensor:
    """
    Restore attention output to the same rank convention as the input tensors.
    """
    if squeeze_head:
        output = output[:, 0]
    return Tensor(output.astype(np.float32))


def _numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable NumPy softmax helper.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.maximum(np.sum(exp_x, axis=axis, keepdims=True), 1e-9)


def _flash_attention_streaming(
    q_block: np.ndarray,
    k_data: np.ndarray,
    v_data: np.ndarray,
    normalized_mask: Optional[np.ndarray],
    key_block_size: int,
) -> np.ndarray:
    """
    Core streaming exact-attention kernel shared by flash-style and paged variants.
    """
    batch_size, num_heads, query_len, head_dim = q_block.shape
    key_len = k_data.shape[-2]
    scale = 1.0 / math.sqrt(head_dim)

    output = np.zeros((batch_size, num_heads, query_len, head_dim), dtype=np.float32)
    running_max = np.full((batch_size, num_heads, query_len), -np.inf, dtype=np.float32)
    running_sum = np.zeros((batch_size, num_heads, query_len), dtype=np.float32)

    for key_start in range(0, key_len, key_block_size):
        key_end = min(key_start + key_block_size, key_len)
        k_block = k_data[:, :, key_start:key_end, :]
        v_block = v_data[:, :, key_start:key_end, :]

        scores = np.matmul(q_block, np.swapaxes(k_block, -2, -1)) * scale

        if normalized_mask is not None:
            mask_block = normalized_mask[:, :, :, key_start:key_end]
            scores = np.where(mask_block, scores, MASK_VALUE)

        block_max = np.max(scores, axis=-1)
        new_running_max = np.maximum(running_max, block_max)

        exp_scores = np.exp(scores - new_running_max[..., np.newaxis])
        max_rescale = np.exp(running_max - new_running_max)

        output *= max_rescale[..., np.newaxis]
        running_sum *= max_rescale

        output += np.matmul(exp_scores, v_block)
        running_sum += np.sum(exp_scores, axis=-1)
        running_max = new_running_max

    return output / np.maximum(running_sum[..., np.newaxis], 1e-9)


def _normalize_mask_for_attention_variant(
    mask: Optional[Tensor],
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
) -> Optional[np.ndarray]:
    """
    Normalize and broadcast masks for 4D attention kernels.
    """
    if mask is None:
        return None

    normalized_mask = _normalize_attention_mask(mask, batch_size, query_len, key_len)
    if normalized_mask.shape[1] == 1 and num_heads != 1:
        normalized_mask = np.broadcast_to(normalized_mask, (batch_size, num_heads, query_len, key_len))
    return normalized_mask


def _normalize_attention_mask(mask: Tensor, batch_size: int, query_len: int, key_len: int) -> np.ndarray:
    """
    Normalize masks to a broadcastable boolean array of shape (B, H_or_1, Tq, Tk).

    Supported inputs:
    - (Tq, Tk)
    - (B, Tq, Tk)
    - (B, H, Tq, Tk)
    """
    mask_data = mask.data if isinstance(mask, Tensor) else np.asarray(mask)

    if mask_data.ndim == 2:
        if mask_data.shape != (query_len, key_len):
            raise ValueError(
                f"Mask shape mismatch: expected {(query_len, key_len)}, got {mask_data.shape}"
            )
        mask_data = np.broadcast_to(mask_data.reshape(1, 1, query_len, key_len), (batch_size, 1, query_len, key_len))
    elif mask_data.ndim == 3:
        if mask_data.shape[0] != batch_size or mask_data.shape[1:] != (query_len, key_len):
            raise ValueError(
                f"Mask shape mismatch: expected {(batch_size, query_len, key_len)}, got {mask_data.shape}"
            )
        mask_data = mask_data.reshape(batch_size, 1, query_len, key_len)
    elif mask_data.ndim == 4:
        if mask_data.shape[0] != batch_size or mask_data.shape[-2:] != (query_len, key_len):
            raise ValueError(
                f"Mask shape mismatch: expected batch {batch_size} and trailing dims {(query_len, key_len)}, got {mask_data.shape}"
            )
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D mask, got {mask_data.ndim}D")

    return mask_data.astype(bool)


def _compute_attention_scores(Q: Tensor, K: Tensor) -> Tensor:
    """
    Compute raw dot-product attention scores.

    Supports both 3D tensors (B, T, D) and 4D tensors (B, H, T, D).
    """
    K_t = K.transpose(-2, -1)
    return Q.matmul(K_t)


def _scale_scores(scores: Tensor, d_model: int) -> Tensor:
    """
    Scale attention scores by 1/sqrt(d_model).
    """
    scale_factor = 1.0 / math.sqrt(d_model)
    return scores * scale_factor


def _apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """
    Apply a causal or padding mask by adding a large negative number to masked positions.
    """
    batch_size = scores.shape[0]
    query_len = scores.shape[-2]
    key_len = scores.shape[-1]
    normalized_mask = _normalize_attention_mask(mask, batch_size, query_len, key_len)
    adder = (~normalized_mask).astype(np.float32) * MASK_VALUE
    if scores.ndim == 3 and adder.shape[1] == 1:
        adder = adder[:, 0]
    return scores + Tensor(adder)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Standard scaled dot-product attention.
    """
    scores = _compute_attention_scores(Q, K)
    scores = _scale_scores(scores, Q.shape[-1])
    if mask is not None:
        scores = _apply_mask(scores, mask)
    softmax = Softmax()
    attention_weights = softmax(scores, dim=-1)
    output = attention_weights.matmul(V)
    return output, attention_weights


def flash_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    block_size: int = DEFAULT_FLASH_BLOCK_SIZE,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    FlashAttention-style exact attention using tiled, numerically stable softmax accumulation.

    This implementation is CPU/NumPy oriented and optimized for lower peak memory rather than
    full-kernel GPU fusion. It returns attention output and `None` for weights to avoid
    materializing the full attention matrix.
    """
    q_data, k_data, v_data, squeeze_head = _to_4d_attention_tensors(Q, K, V)
    batch_size, num_heads, query_len, head_dim = q_data.shape
    key_len = k_data.shape[-2]
    normalized_mask = _normalize_mask_for_attention_variant(mask, batch_size, num_heads, query_len, key_len)
    output = _flash_attention_streaming(q_data, k_data, v_data, normalized_mask, block_size)
    return _restore_attention_output(output, squeeze_head), None


def flash_attention_v2(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    query_block_size: int = 64,
    key_block_size: int = DEFAULT_FLASH_BLOCK_SIZE,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Educational FlashAttention-2 style variant using both query and key tiling.
    """
    q_data, k_data, v_data, squeeze_head = _to_4d_attention_tensors(Q, K, V)
    batch_size, num_heads, query_len, _ = q_data.shape
    key_len = k_data.shape[-2]
    normalized_mask = _normalize_mask_for_attention_variant(mask, batch_size, num_heads, query_len, key_len)

    output = np.zeros_like(q_data, dtype=np.float32)
    for query_start in range(0, query_len, query_block_size):
        query_end = min(query_start + query_block_size, query_len)
        q_block = q_data[:, :, query_start:query_end, :]
        mask_block = None if normalized_mask is None else normalized_mask[:, :, query_start:query_end, :]
        output[:, :, query_start:query_end, :] = _flash_attention_streaming(
            q_block,
            k_data,
            v_data,
            mask_block,
            key_block_size,
        )

    return _restore_attention_output(output, squeeze_head), None


def flash_attention_v3(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    query_block_size: int = 64,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Educational FlashAttention-3 style variant that combines query blocking with page-aligned KV streaming.
    """
    return flash_attention_v2(
        Q,
        K,
        V,
        mask=mask,
        query_block_size=query_block_size,
        key_block_size=page_size,
    )


def paged_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Paged attention processes K/V memory in fixed-size pages, similar to paged KV caches during decoding.
    """
    return flash_attention(Q, K, V, mask=mask, block_size=page_size)


def sparse_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    window_size: int = DEFAULT_SPARSE_WINDOW_SIZE,
    global_indices: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Local-window sparse attention with optional global tokens.
    """
    q_data, k_data, v_data, squeeze_head = _to_4d_attention_tensors(Q, K, V)
    batch_size, num_heads, query_len, head_dim = q_data.shape
    key_len = k_data.shape[-2]

    allowed = np.zeros((1, 1, query_len, key_len), dtype=bool)
    global_indices = [] if global_indices is None else list(global_indices)

    for i in range(query_len):
        start = max(0, i - window_size)
        end = min(key_len, i + window_size + 1)
        allowed[:, :, i, start:end] = True

    for index in global_indices:
        if 0 <= index < key_len:
            allowed[:, :, :, index] = True

    allowed = np.broadcast_to(allowed, (batch_size, num_heads, query_len, key_len))
    normalized_mask = _normalize_mask_for_attention_variant(mask, batch_size, num_heads, query_len, key_len)
    if normalized_mask is not None:
        allowed = allowed & normalized_mask

    scores = np.matmul(q_data, np.swapaxes(k_data, -2, -1)) * (1.0 / math.sqrt(head_dim))
    scores = np.where(allowed, scores, MASK_VALUE)
    attention_weights = _numpy_softmax(scores, axis=-1)
    output = np.matmul(attention_weights, v_data)

    if squeeze_head:
        return _restore_attention_output(output, True), Tensor(attention_weights[:, 0].astype(np.float32))
    return _restore_attention_output(output, False), Tensor(attention_weights.astype(np.float32))


def linear_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Associative linear attention using the ELU(x)+1 positive feature map.
    """
    if mask is not None:
        raise NotImplementedError("Masked linear attention is not implemented in this educational CPU path")

    q_data, k_data, v_data, squeeze_head = _to_4d_attention_tensors(Q, K, V)

    phi_q = np.where(q_data > 0, q_data, np.exp(q_data) - 1.0) + 1.0
    phi_k = np.where(k_data > 0, k_data, np.exp(k_data) - 1.0) + 1.0

    kv_summary = np.matmul(np.swapaxes(phi_k, -2, -1), v_data)
    k_sum = np.sum(phi_k, axis=-2)

    numerator = np.matmul(phi_q, kv_summary)
    denominator = np.sum(phi_q * k_sum[:, :, np.newaxis, :], axis=-1, keepdims=True)
    output = numerator / np.maximum(denominator, eps)

    return _restore_attention_output(output, squeeze_head), None


def _repeat_kv_heads(x: Tensor, num_query_heads: int) -> Tensor:
    """
    Repeat key/value heads so grouped-query and multi-query attention can share them.
    """
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
    repeat_factor = num_query_heads // num_kv_heads
    repeated = np.repeat(x.data, repeat_factor, axis=1)
    return Tensor(repeated)


def _resolve_attention_backend(
    backend: Union[str, Callable[..., Tuple[Tensor, Optional[Tensor]]]]
) -> Callable[..., Tuple[Tensor, Optional[Tensor]]]:
    if callable(backend):
        return backend
    if backend == "standard":
        return scaled_dot_product_attention
    if backend == "flash":
        return flash_attention
    raise ValueError(f"Unknown attention backend '{backend}'")


class MultiHeadAttention:
    """
    Standard multi-head attention with pluggable attention backend.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_backend: Union[str, Callable[..., Tuple[Tensor, Optional[Tensor]]]] = "standard",
        block_size: int = DEFAULT_FLASH_BLOCK_SIZE,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Multiple-head attention dimension mismatch\n"
                f"    embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
                f"     Multi-head attention splits embed_dims equally maong heads, so embed_dim must be a multiple of num_heads\n"
                f"       Try: embed_dim={num_heads * (embed_dim // num_heads + 1)} ( next valid size) or num_heads={embed_dim // {embed_dim // num_heads}} (fewer heads)"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_backend = attention_backend
        self.attention_fn = _resolve_attention_backend(attention_backend)
        self.block_size = block_size

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, x: Tensor, batch_size: int, seq_len: int, num_heads: Optional[int] = None) -> Tensor:
        """
        Reshape (B, T, H*D) into (B, H, T, D).
        """
        current_heads = num_heads if num_heads is not None else self.num_heads
        current_head_dim = x.shape[-1] // current_heads
        x = x.reshape(batch_size, seq_len, current_heads, current_head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Reshape (B, H, T, D) back into (B, T, H*D).
        """
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def _project_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj.forward(x), batch_size, seq_len, self.num_heads)
        k = self._split_heads(self.k_proj.forward(x), batch_size, seq_len, self.num_heads)
        v = self._split_heads(self.v_proj.forward(x), batch_size, seq_len, self.num_heads)
        return q, k, v

    def _run_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        if self.attention_backend == "flash":
            attended, _ = self.attention_fn(q, k, v, mask=mask, block_size=self.block_size)
        else:
            attended, _ = self.attention_fn(q, k, v, mask=mask)
        return attended

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Multi-HeadAttention input dimension mismatch\n"
                f"     Expected embed_dim={self.embed_dim}, got {embed_dim} from input shape {x.shape}"
                f"     The last dimension of input must match embed_dim from intialization(MultiHeadAttention({self.embed_dim},{self.num_heads}))\n"
                f"      Try: x.reshape({x.shape[0]},{x.shape[1]},{self.embed_dim}) or create new MultiHeadAttention({embed_dim},num_head) "
            )

        q, k, v = self._project_qkv(x)
        attended = self._run_attention(q, k, v, mask)
        concat_output = self._merge_heads(attended, batch_size, seq_len)
        return self.out_proj.forward(concat_output)

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.forward(x, mask)

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params


class FlashMultiHeadAttention(MultiHeadAttention):
    """
    Multi-head attention that uses a tiled FlashAttention-style backend.
    """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int = DEFAULT_FLASH_BLOCK_SIZE):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_backend="flash",
            block_size=block_size,
        )


class FlashMultiHeadAttentionV2(MultiHeadAttention):
    """
    Multi-head attention using the educational FlashAttention-2 style backend.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        query_block_size: int = 64,
        key_block_size: int = DEFAULT_FLASH_BLOCK_SIZE,
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.query_block_size = query_block_size
        self.key_block_size = key_block_size

    def _run_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        attended, _ = flash_attention_v2(
            q,
            k,
            v,
            mask=mask,
            query_block_size=self.query_block_size,
            key_block_size=self.key_block_size,
        )
        return attended


class FlashMultiHeadAttentionV3(MultiHeadAttention):
    """
    Multi-head attention using the educational FlashAttention-3 style backend.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        query_block_size: int = 64,
        page_size: int = DEFAULT_PAGE_SIZE,
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.query_block_size = query_block_size
        self.page_size = page_size

    def _run_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        attended, _ = flash_attention_v3(
            q,
            k,
            v,
            mask=mask,
            query_block_size=self.query_block_size,
            page_size=self.page_size,
        )
        return attended


class GroupedQueryAttention(MultiHeadAttention):
    """
    Grouped-query attention: many query heads share a smaller number of key/value heads.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        attention_backend: Union[str, Callable[..., Tuple[Tensor, Optional[Tensor]]]] = "standard",
        block_size: int = DEFAULT_FLASH_BLOCK_SIZE,
    ):
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
        super().__init__(embed_dim, num_heads, attention_backend=attention_backend, block_size=block_size)
        self.num_kv_heads = num_kv_heads
        self.k_proj = Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = Linear(embed_dim, num_kv_heads * self.head_dim)

    def _project_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj.forward(x), batch_size, seq_len, self.num_heads)
        k = self._split_heads(self.k_proj.forward(x), batch_size, seq_len, self.num_kv_heads)
        v = self._split_heads(self.v_proj.forward(x), batch_size, seq_len, self.num_kv_heads)
        k = _repeat_kv_heads(k, self.num_heads)
        v = _repeat_kv_heads(v, self.num_heads)
        return q, k, v


class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi-query attention: all query heads share a single key head and a single value head.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_backend: Union[str, Callable[..., Tuple[Tensor, Optional[Tensor]]]] = "standard",
        block_size: int = DEFAULT_FLASH_BLOCK_SIZE,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=1,
            attention_backend=attention_backend,
            block_size=block_size,
        )


class MultiLatentAttention(MultiHeadAttention):
    """
    Multi-latent attention: compress input to a smaller latent representation before building K/V.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int,
        num_kv_heads: Optional[int] = None,
        attention_backend: Union[str, Callable[..., Tuple[Tensor, Optional[Tensor]]]] = "standard",
        block_size: int = DEFAULT_FLASH_BLOCK_SIZE,
    ):
        super().__init__(embed_dim, num_heads, attention_backend=attention_backend, block_size=block_size)
        self.latent_dim = latent_dim
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by num_kv_heads={self.num_kv_heads}"
            )

        self.kv_down_proj = Linear(embed_dim, latent_dim)
        self.k_proj = Linear(latent_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(latent_dim, self.num_kv_heads * self.head_dim)

    def _project_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj.forward(x), batch_size, seq_len, self.num_heads)
        latent = self.kv_down_proj.forward(x)
        k = self._split_heads(self.k_proj.forward(latent), batch_size, seq_len, self.num_kv_heads)
        v = self._split_heads(self.v_proj.forward(latent), batch_size, seq_len, self.num_kv_heads)
        if self.num_kv_heads != self.num_heads:
            k = _repeat_kv_heads(k, self.num_heads)
            v = _repeat_kv_heads(v, self.num_heads)
        return q, k, v

    def parameters(self) -> List[Tensor]:
        params = super().parameters()
        params.extend(self.kv_down_proj.parameters())
        return params


class SparseMultiHeadAttention(MultiHeadAttention):
    """
    Multi-head attention with local-window sparse connectivity and optional global tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = DEFAULT_SPARSE_WINDOW_SIZE,
        global_indices: Optional[List[int]] = None,
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.window_size = window_size
        self.global_indices = [] if global_indices is None else list(global_indices)

    def _run_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        attended, _ = sparse_attention(
            q,
            k,
            v,
            mask=mask,
            window_size=self.window_size,
            global_indices=self.global_indices,
        )
        return attended


class LinearMultiHeadAttention(MultiHeadAttention):
    """
    Multi-head attention with associative linear attention instead of explicit score matrices.
    """

    def __init__(self, embed_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.eps = eps

    def _run_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        attended, _ = linear_attention(q, k, v, mask=mask, eps=self.eps)
        return attended


class PagedMultiHeadAttention(MultiHeadAttention):
    """
    Multi-head attention that streams KV memory in fixed pages.
    """

    def __init__(self, embed_dim: int, num_heads: int, page_size: int = DEFAULT_PAGE_SIZE):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.page_size = page_size

    def _run_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        attended, _ = paged_attention(q, k, v, mask=mask, page_size=self.page_size)
        return attended
