# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable sparse attention for the DFly2 scratch context.

FlashMLA's sparse prefill kernel exists only for SM90 and the SM100 family.
Capability 12 (DGX Spark / GB10, RTX 50, RTX PRO Blackwell) has no FlashMLA, so
the draft's attention over its small BF16 scratch context runs here instead.
The context is at most `index_topk + sliding_window + block_size` rows per
request, so a gather plus two batched matmuls stays small (not yet measured).
"""

import torch

# Upper bound for the float32 score tensor of one chunk, in elements.
_MAX_SCORE_ELEMENTS = 1 << 26


def dfly_sparse_attention_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    sink: torch.Tensor | None,
    width: int,
) -> torch.Tensor:
    """Sparse MQA with K = V and a per-head sink, like `flash_mla_sparse_fwd`.

    Args:
        q: `[batch * width, heads, dim]` queries; the `width` rows of one
            request are consecutive and share that request's indices.
        kv: `[rows, dim]` flattened scratch context of all requests.
        indices: `[batch, keys]` global rows of `kv`; `-1` is unused.
        scale: softmax scale applied to `q @ k`.
        sink: `[>= heads]` float32 sink logits, or `None`.
        width: query rows per request.

    Returns:
        `[batch * width, heads, dim]` in `q.dtype`. Rows whose request has no
        valid key are zero.

    """
    tokens, heads, dim = q.shape
    batch, keys = indices.shape
    assert tokens == batch * width, (tokens, batch, width)
    assert kv.dim() == 2 and kv.shape[1] == dim, (kv.shape, dim)
    out = q.new_zeros(tokens, heads, dim)
    if batch == 0 or keys == 0:
        return out
    sink_h = None if sink is None else sink[:heads].float()
    per_request = max(1, width * heads * keys)
    chunk = max(1, min(batch, _MAX_SCORE_ELEMENTS // per_request))
    qb = q.view(batch, width, heads, dim)
    ob = out.view(batch, width, heads, dim)
    for start in range(0, batch, chunk):
        end = min(batch, start + chunk)
        idx = indices[start:end].long()
        valid = idx >= 0
        rows = kv[idx.clamp(min=0)].float()  # [b, keys, dim]
        # Empty slots read row 0; zero them so a NaN there cannot leak in.
        rows = rows.masked_fill(~valid[..., None], 0.0)
        scores = torch.einsum("bwhd,bkd->bwhk", qb[start:end].float(), rows)
        scores = scores * scale
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        top = scores.amax(dim=-1)  # [b, w, h]
        if sink_h is not None:
            top = torch.maximum(top, sink_h)
        top = torch.nan_to_num(top, neginf=0.0)
        weights = torch.exp(scores - top[..., None])
        denom = weights.sum(dim=-1)
        if sink_h is not None:
            denom = denom + torch.exp(sink_h - top)
        result = torch.einsum("bwhk,bkd->bwhd", weights, rows)
        result = result / denom.clamp_min(torch.finfo(torch.float32).tiny)[..., None]
        alive = valid.any(dim=-1)[:, None, None, None]
        ob[start:end] = torch.where(alive, result, 0.0).to(out.dtype)
    return out
