"""Gemma 4 的全局与滑动窗口 attention 参考实现。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.layers.linear import BF16LinearFactory, LinearFactory
from hpc101_infer.layers.norm import RMSNorm
from hpc101_infer.layers.rotary import RotaryEmbedding
from hpc101_infer.models.config import RotaryConfig
from hpc101_infer.runtime.kv_cache import KVCache


def make_causal_mask(
    sequence_length: int,
    layer_type: str,
    sliding_window: int,
    device: torch.device,
) -> torch.Tensor:
    """构造 ``[1, 1, query_length, key_length]`` 的加性 causal mask。"""
    query = torch.arange(sequence_length, device=device).unsqueeze(1)
    key = torch.arange(sequence_length, device=device).unsqueeze(0)
    allowed = key <= query
    if layer_type == "sliding_attention":
        allowed &= key > query - sliding_window
    mask = torch.zeros(
        (sequence_length, sequence_length), device=device, dtype=torch.float32
    )
    return mask.masked_fill(~allowed, float("-inf")).unsqueeze(0).unsqueeze(0)


def make_attention_mask(
    positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    key_length: int,
    layer_type: str,
    dtype: torch.dtype,
    sliding_window: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """同时屏蔽未来 token、padding token 和滑动窗口外的历史 token。"""
    key_positions = torch.arange(key_length, device=positions.device)
    query_valid = positions < sequence_lengths[:, None]
    allowed = key_positions[None, None, :] <= positions[:, :, None]
    allowed &= key_positions[None, None, :] < sequence_lengths[:, None, None]
    if layer_type == "sliding_attention":
        allowed &= key_positions[None, None, :] > (
            positions[:, :, None] - sliding_window
        )
    allowed &= query_valid[:, :, None]
    # 全 padding 的 query 若整行都是 -inf，softmax 会产生 NaN。这里临时
    # 放行一个 key，随后再通过 query_valid 将该 query 的输出清零。
    safe_allowed = allowed.clone()
    safe_allowed[..., 0] |= ~query_valid
    mask = torch.zeros(allowed.shape, device=positions.device, dtype=dtype)
    mask.masked_fill_(~safe_allowed, float("-inf"))
    return mask.unsqueeze(1), query_valid


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """将较少的 KV heads 逻辑扩展到 query heads，实现 grouped-query attention。"""
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(
        batch, heads, repeats, seq_len, head_dim
    )
    return expanded.reshape(batch, heads * repeats, seq_len, head_dim)


def _decode_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    layer_type: str,
    sliding_window: int,
    num_kv_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Efficient GQA attention for decode (seq_len=1) without repeat_kv.

    Avoids materializing the expanded K/V tensors by grouping Q heads
    and computing per-KV-head attention with matmul.
    """
    batch, num_kv_heads, kv_len, head_dim = key.shape
    num_qo_heads = query.shape[1]
    device = key.device

    key_positions = torch.arange(kv_len, device=device)
    pos = positions[:, 0]
    allowed = (key_positions[None, :] <= pos[:, None]) & (
        key_positions[None, :] < sequence_lengths[:, None]
    )
    if layer_type == "sliding_attention":
        allowed &= key_positions[None, :] > (pos[:, None] - sliding_window)
    query_valid = (pos < sequence_lengths)[:, None]

    safe_allowed = allowed.clone()
    safe_allowed[~query_valid[:, 0], 0] = True

    q_grouped = query.view(batch, num_kv_heads, num_kv_groups, head_dim)
    scores = torch.matmul(q_grouped, key.transpose(-2, -1))
    scores.masked_fill_(~safe_allowed[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    output = output.reshape(batch, num_qo_heads, 1, head_dim).to(query.dtype)
    return output, query_valid


class AttentionLayer(nn.Module):
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rotary_config: RotaryConfig,
        rms_norm_eps: float,
        max_position_embeddings: int,
        attention_k_eq_v: bool = False,
        linear_factory: LinearFactory | None = None,
        module_prefix: str = "self_attn",
    ):
        super().__init__()
        factory = linear_factory or BF16LinearFactory()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_qo_heads // self.num_kv_heads
        self.q_proj = factory.create(
            f"{module_prefix}.q_proj", hidden_size, num_qo_heads * self.head_dim, False
        )
        self.k_proj = factory.create(
            f"{module_prefix}.k_proj",
            hidden_size,
            self.num_kv_heads * self.head_dim,
            False,
        )
        self.v_proj = (
            None
            if attention_k_eq_v
            else factory.create(
                f"{module_prefix}.v_proj",
                hidden_size,
                self.num_kv_heads * self.head_dim,
                False,
            )
        )
        self.o_proj = factory.create(
            f"{module_prefix}.o_proj",
            num_qo_heads * self.head_dim,
            hidden_size,
            False,
        )
        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, rms_norm_eps, with_scale=False)

        self.rotary = RotaryEmbedding(
            rotary_config,
            head_dim,
            max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        max_seq_len: int,
        layer_id: int,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        # Projection 后先保持 [batch, sequence, heads, head_dim]，应用 RoPE
        # 后再转为 attention 计算使用的 [batch, heads, sequence, head_dim]。
        query = self.q_proj(hidden_states).view(batch, seq_len, -1, self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        value = (
            key
            if self.v_proj is None
            else self.v_proj(hidden_states).view(
                batch, seq_len, self.num_kv_heads, self.head_dim
            )
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary(position_ids, query, key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = self.v_norm(value).transpose(1, 2)
        if kv_cache is not None:
            kv_cache.write(layer_id, position_ids, key, value)
            cached = kv_cache.view(layer_id, max_seq_len)
            key, value = cached.key, cached.value
            if seq_len == 1:
                output, query_valid = _decode_gqa_attention(
                    query, key, value, position_ids, sequence_lengths,
                    "full_attention", 0, self.num_kv_groups,
                )
                output = output.transpose(1, 2).reshape(batch, seq_len, -1)
                output = output * query_valid.unsqueeze(-1).to(output.dtype)
                return self.o_proj(output)
        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)
        if seq_len == 1:
            mask, query_valid = make_attention_mask(
                positions=position_ids,
                sequence_lengths=sequence_lengths,
                key_length=key.shape[2],
                layer_type="full_attention",
                dtype=query.dtype,
            )
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0, scale=1.0,
            )
        elif seq_len > 1:
            if key.shape[2] > seq_len:
                # Chunked scoring: keys include previous chunks, need position-based mask
                mask, query_valid = make_attention_mask(
                    positions=position_ids,
                    sequence_lengths=sequence_lengths,
                    key_length=key.shape[2],
                    layer_type="full_attention",
                    dtype=query.dtype,
                )
                output = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, dropout_p=0.0, scale=1.0,
                )
            else:
                # Prefill: standard causal mask is correct
                output = F.scaled_dot_product_attention(
                    query, key, value, is_causal=True, dropout_p=0.0, scale=1.0,
                )
                query_valid = torch.ones(
                    batch, seq_len, dtype=torch.bool, device=query.device
                )
        else:
            mask, query_valid = make_attention_mask(
                positions=position_ids,
                sequence_lengths=sequence_lengths,
                key_length=key.shape[2],
                layer_type="full_attention",
                dtype=query.dtype,
            )
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0, scale=1.0,
            )
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        output = output * query_valid.unsqueeze(-1)
        return self.o_proj(output)


class SlidingAttentionLayer(nn.Module):
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rotary_config: RotaryConfig,
        rms_norm_eps: float,
        max_position_embeddings: int,
        sliding_window: int,
        linear_factory: LinearFactory | None = None,
        module_prefix: str = "self_attn",
    ):
        super().__init__()
        factory = linear_factory or BF16LinearFactory()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_qo_heads // self.num_kv_heads
        self.sliding_window = sliding_window
        self.q_proj = factory.create(
            f"{module_prefix}.q_proj", hidden_size, num_qo_heads * self.head_dim, False
        )
        self.k_proj = factory.create(
            f"{module_prefix}.k_proj",
            hidden_size,
            self.num_kv_heads * self.head_dim,
            False,
        )
        self.v_proj = factory.create(
            f"{module_prefix}.v_proj",
            hidden_size,
            self.num_kv_heads * self.head_dim,
            False,
        )
        self.o_proj = factory.create(
            f"{module_prefix}.o_proj",
            num_qo_heads * self.head_dim,
            hidden_size,
            False,
        )
        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, rms_norm_eps, with_scale=False)

        self.rotary = RotaryEmbedding(
            rotary_config,
            head_dim,
            max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        max_seq_len: int,
        layer_id: int,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        # Projection 后先保持 [batch, sequence, heads, head_dim]，应用 RoPE
        # 后再转为 attention 计算使用的 [batch, heads, sequence, head_dim]。
        query = self.q_proj(hidden_states).view(batch, seq_len, -1, self.head_dim)
        key = self.k_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        value = (
            key
            if self.v_proj is None
            else self.v_proj(hidden_states).view(
                batch, seq_len, self.num_kv_heads, self.head_dim
            )
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary(position_ids, query, key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = self.v_norm(value).transpose(1, 2)
        if kv_cache is not None:
            kv_cache.write(layer_id, position_ids, key, value)
            cached = kv_cache.view(layer_id, max_seq_len)
            key, value = cached.key, cached.value
            if seq_len == 1:
                output, query_valid = _decode_gqa_attention(
                    query, key, value, position_ids, sequence_lengths,
                    "sliding_attention", self.sliding_window, self.num_kv_groups,
                )
                output = output.transpose(1, 2).reshape(batch, seq_len, -1)
                output = output * query_valid.unsqueeze(-1).to(output.dtype)
                return self.o_proj(output)
        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)
        mask, query_valid = make_attention_mask(
            positions=position_ids,
            sequence_lengths=sequence_lengths,
            key_length=key.shape[2],
            layer_type="sliding_attention",
            dtype=query.dtype,
            sliding_window=self.sliding_window,
        )
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, scale=1.0,
        )
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        output = output * query_valid.unsqueeze(-1)
        return self.o_proj(output)
