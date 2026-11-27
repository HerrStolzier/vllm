# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The portable DFly2 sparse attention matches a per-token FP32 reference."""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.ops import dfly_torch
from vllm.models.deepseek_v41.nvidia.ops.dfly_torch import dfly_sparse_attention_torch


def _reference(q, kv, indices, scale, sink, width):
    out = torch.zeros(q.shape, dtype=torch.float32)
    for t in range(q.shape[0]):
        idx = indices[t // width].long()
        idx = idx[idx >= 0]
        if idx.numel() == 0:
            continue
        rows = kv[idx].float()
        s = (q[t].float() @ rows.T) * scale
        m = s.max(dim=1).values
        if sink is not None:
            m = torch.maximum(m, sink[: q.shape[1]])
        p = torch.exp(s - m[:, None])
        denom = p.sum(dim=1)
        if sink is not None:
            denom = denom + torch.exp(sink[: q.shape[1]] - m)
        out[t] = (p @ rows) / denom[:, None]
    return out


@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("with_sink", [True, False])
@pytest.mark.parametrize("small_chunks", [False, True])
def test_matches_reference(width, with_sink, small_chunks, monkeypatch):
    if small_chunks:
        monkeypatch.setattr(dfly_torch, "_MAX_SCORE_ELEMENTS", 1)
    g = torch.Generator().manual_seed(width * 10 + with_sink)
    batch, heads, dim, per_request, keys = 5, 4, 32, 40, 24
    q = torch.randn(batch * width, heads, dim, generator=g)
    kv = torch.randn(batch * per_request, dim, generator=g)
    indices = torch.full((batch, keys), -1, dtype=torch.int32)
    for b in range(batch):
        n = [keys, 7, 1, 0, 13][b]
        rows = torch.randperm(per_request, generator=g)[:n] + b * per_request
        indices[b, :n] = rows.to(torch.int32)
    # Padded sink as in the model: longer than the local head count.
    sink = torch.randn(8, generator=g) if with_sink else None
    scale = dim**-0.5

    got = dfly_sparse_attention_torch(q, kv, indices, scale, sink, width)
    want = _reference(q, kv, indices, scale, sink, width)

    assert got.shape == q.shape and got.dtype == q.dtype
    torch.testing.assert_close(got.float(), want, rtol=1e-5, atol=1e-5)
    # Request 3 has no valid key: its rows must be exactly zero.
    assert torch.count_nonzero(got[3 * width : 4 * width]) == 0


def test_bf16_inputs_return_bf16():
    q = torch.randn(6, 2, 16).bfloat16()
    kv = torch.randn(12, 16).bfloat16()
    indices = torch.tensor([[0, 3, 5, -1]], dtype=torch.int32)
    got = dfly_sparse_attention_torch(q, kv, indices, 0.25, None, 6)
    want = _reference(q, kv, indices, 0.25, None, 6)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)


def test_empty_slots_never_read_their_row():
    """`-1` slots are empty, as in `flash_mla_sparse_fwd`: a row no valid index
    names must not change the result, even if it holds NaN."""
    g = torch.Generator().manual_seed(3)
    heads, dim, width = 4, 32, 6
    q = torch.randn(2 * width, heads, dim, generator=g)
    kv = torch.randn(80, dim, generator=g)
    kv[0] = float("nan")
    indices = torch.tensor(
        [[2, 5, 9, -1, -1, -1], [41, 47, -1, 60, -1, -1]], dtype=torch.int32
    )
    sink = torch.randn(8, generator=g)

    got = dfly_sparse_attention_torch(q, kv, indices, dim**-0.5, sink, width)
    want = _reference(q, kv, indices, dim**-0.5, sink, width)

    torch.testing.assert_close(got.float(), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("width", [1, 6])
def test_matches_flash_mla_sparse_fwd(width):
    """The fallback reproduces the call `DFlyContext.attend` makes on SM90/SM100."""
    import vllm.v1.attention.ops.flashmla as fm
    from vllm.platforms import current_platform

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    batch, heads, dim, query_width, context = 4, 16, 512, 6, 640
    stride = context + query_width
    keys = 768 if width > 1 else context
    q = torch.randn(batch * width, heads, dim, generator=g, device=device).bfloat16()
    kv = torch.randn(batch * stride, dim, generator=g, device=device).bfloat16()
    indices = torch.full((batch, keys), -1, dtype=torch.int32, device=device)
    for b, n in enumerate([keys, 100, 1, 0]):
        cols = torch.randperm(min(keys, stride), generator=g, device=device)[:n]
        indices[b, cols] = (b * stride + cols).to(torch.int32)
    min_heads = 8 if current_platform.is_device_capability_family(90) else 64
    padded = max(min_heads, 1 << (heads - 1).bit_length())
    sink = torch.full((padded,), -float("inf"), device=device)
    sink[:heads] = torch.randn(heads, generator=g, device=device)

    flat = indices[:, None, :].expand(-1, width, -1).reshape(q.shape[0], 1, -1)
    want, _, _ = fm.flash_mla_sparse_fwd(
        torch.nn.functional.pad(q, (0, 0, 0, padded - heads)).contiguous(),
        kv.view(-1, 1, dim),
        flat.contiguous(),
        dim**-0.5,
        dim,
        sink,
    )
    want = torch.where((flat >= 0).any(-1).unsqueeze(-1), want[:, :heads], 0)
    got = dfly_sparse_attention_torch(q, kv, indices, dim**-0.5, sink, width)
    assert got.dtype == want.dtype
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-3)
