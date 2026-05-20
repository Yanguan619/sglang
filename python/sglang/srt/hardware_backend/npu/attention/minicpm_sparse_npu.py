from __future__ import annotations

import math
from typing import Optional

import torch


def get_block_table_v2_ref(
    topk_idx: torch.Tensor,
    page_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_k_sparse_bs_tensor: torch.Tensor,
    sparse_topk: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Reference implementation of get_block_table_v2 for NPU.

    Args:
        topk_idx: [num_sparse_tokens, sparse_topk], block-level indices from compressed attention.
        page_table: [bs, max_blocks], block-level page table (physical block IDs).
        token_to_bs: [num_sparse_tokens], maps each sparse token to its batch index.
        token_pos_in_bs: [num_sparse_tokens], position of token in its batch.
        seqlen_k_sparse_bs_tensor: [num_sparse_bs], KV seq length per sparse batch.
        sparse_topk: number of sparse blocks per query token.
        block_size: tokens per block.

    Returns:
        [num_sparse_tokens, sparse_topk * block_size], sparse block table.
    """
    num_tokens = topk_idx.shape[0]
    device = topk_idx.device
    num_topk_tokens = sparse_topk * block_size

    batch_pt = page_table[token_to_bs.long()]

    block_idx = topk_idx
    offsets = torch.arange(block_size, device=device).view(1, 1, block_size)
    logical_pos = (
        block_idx.unsqueeze(-1) * block_size + offsets
    ).reshape(num_tokens, num_topk_tokens).long()

    max_len = page_table.shape[1]
    clamped_pos = logical_pos.clamp(0, max_len - 1)
    output = torch.gather(batch_pt, 1, clamped_pos).int()

    valid_blocks = topk_idx >= 0
    valid_mask = (
        valid_blocks.unsqueeze(-1)
        .expand(-1, -1, block_size)
        .reshape(num_tokens, num_topk_tokens)
    )
    output[~valid_mask] = 0

    return output


def infllmv2_attn_stage1_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_v: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    causal: bool = True,
) -> torch.Tensor:
    total_q, num_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    groups = num_heads // kv_heads
    total_k = k.shape[0]
    device = q.device
    scale = 1.0 / math.sqrt(head_dim)

    max_seqlen_k_val = max_seqlen_k if max_seqlen_k is not None else total_k

    q_r = (
        q.view(total_q, kv_heads, groups, head_dim).permute(1, 0, 2, 3).contiguous()
    )
    k_r = k.permute(1, 2, 0).contiguous()

    score_padded = torch.full(
        (kv_heads, total_q, max_seqlen_k_val),
        float("-inf"),
        dtype=q.dtype,
        device=device,
    )

    if cu_seqlens_k is None:
        score = torch.einsum("hqgd,hdk->hqgk", q_r, k_r) * scale
        score = score.sum(dim=2)
        k_len = min(total_k, max_seqlen_k_val)
        score_padded[:, :, :k_len] = score[:, :, :k_len]
        return score_padded

    batch_size = cu_seqlens_k.shape[0] - 1
    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item() if b + 1 < cu_seqlens_k.shape[0] else total_k

        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len <= 0 or k_len <= 0:
            continue

        q_seq = q_r[:, q_start:q_end, :, :].contiguous()
        k_seq = k_r[:, :, k_start:k_end].contiguous()

        seq_score = torch.einsum("hqgd,hdk->hqgk", q_seq, k_seq) * scale

        if causal:
            for q_pos in range(q_len):
                seq_score[:, q_pos, :, q_pos + 1 :] = float("-inf")

        seq_score_max = seq_score.amax(dim=-1, keepdim=True)
        seq_score = (seq_score - seq_score_max).exp()
        seq_score = seq_score / seq_score.sum(dim=-1, keepdim=True)
        seq_score = seq_score.sum(dim=2)

        k_len_pad = min(k_len, max_seqlen_k_val)
        score_padded[:, q_start:q_end, :k_len_pad] = seq_score[:, :, :k_len_pad]

    return score_padded


def max_pooling_1d_varlen_npu(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cache_lens: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_context_len: int,
    local_blocks: int = 2,
    init_blocks: int = 1,
    block_size: int = 64,
    stride: int = 16,
    total_q: int = -1,
) -> torch.Tensor:
    kv_heads = score.shape[0]
    total_q_len = score.shape[1]
    total_k = score.shape[2]
    device = score.device
    dtype = score.dtype

    num_blocks = max_context_len // block_size

    block_score = torch.full(
        (kv_heads, total_q_len, num_blocks),
        float("-inf"),
        dtype=dtype,
        device=device,
    )

    for pos in range(total_k):
        token_pos = pos * stride
        block_idx = token_pos // block_size
        if block_idx < num_blocks:
            block_score[:, :, block_idx] = torch.maximum(
                block_score[:, :, block_idx], score[:, :, pos]
            )

    for b in range(min(init_blocks, num_blocks)):
        block_score[:, :, b] = 1.0

    if local_blocks > 0:
        q_entries = cu_seqlens_q.shape[0] - 1
        for entry_idx in range(q_entries):
            q_start = cu_seqlens_q[entry_idx].item()
            q_end = cu_seqlens_q[entry_idx + 1].item()
            if q_end <= q_start:
                continue
            cache_len = cache_lens[entry_idx].item() if cache_lens is not None else 0
            for q_idx in range(q_start, q_end):
                seq_q_pos = q_idx - q_start
                abs_q_pos = cache_len + seq_q_pos
                query_block = abs_q_pos // block_size
                for lb in range(
                    max(0, query_block - local_blocks + 1),
                    min(num_blocks, query_block + local_blocks),
                ):
                    block_score[:, q_idx, lb] = 1.0

    return block_score


def get_block_table_v3_ref(
    topk_idx: torch.Tensor,
    page_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    cache_seqlens: torch.Tensor,
    _extra: torch.Tensor,
    sparse_topk: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Reference implementation of get_block_table_v3 for NPU (decode version).

    Args:
        topk_idx: [2*bs, sparse_topk], block-level indices per head-group entry.
        page_table: [bs, max_blocks], block-level page table.
        token_to_bs: [2*bs], maps each head-group entry to its batch index.
        cache_seqlens: [bs], cache length per batch.
        _extra: ignored (second copy of cache_seqlens from the C extension interface).
        sparse_topk: number of sparse blocks per head-group entry.
        block_size: tokens per block.

    Returns:
        [2*bs, sparse_topk * block_size], sparse block table.
    """
    num_entries = topk_idx.shape[0]
    device = topk_idx.device
    num_topk_tokens = sparse_topk * block_size

    batch_pt = page_table[token_to_bs.long()]

    block_idx = topk_idx
    offsets = torch.arange(block_size, device=device).view(1, 1, block_size)
    logical_pos = (
        block_idx.unsqueeze(-1) * block_size + offsets
    ).reshape(num_entries, num_topk_tokens).long()

    max_len = page_table.shape[1]
    clamped_pos = logical_pos.clamp(0, max_len - 1)
    output = torch.gather(batch_pt, 1, clamped_pos).int()

    valid_blocks = topk_idx >= 0
    valid_mask = (
        valid_blocks.unsqueeze(-1)
        .expand(-1, -1, block_size)
        .reshape(num_entries, num_topk_tokens)
    )
    output[~valid_mask] = 0

    return output
