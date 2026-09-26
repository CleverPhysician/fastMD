"""Fused segmented attention aggregation.

Computes:

    out[s, d] = sum_i softmax(logits[i, d] over rows with segment[i] == s)
                      * value[i, d]

This fuses the existing ``Dimwise_softmax(logits, segment)`` plus
``weighted_segment_sum(alpha, value, segment)`` chain and avoids materializing
the row-wise attention weights.
"""
from __future__ import annotations

import os
import torch
import triton
import triton.language as tl
from torch.autograd import Function

_NUM_WARPS = int(os.getenv("MATRIS_FUSED_SEGMENT_ATTENTION_NUM_WARPS", "4"))
_PAIRED_BWD_BM = int(os.getenv("MATRIS_FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BM", "8"))
_PAIRED_BWD_BD = int(os.getenv("MATRIS_FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BD", "128"))


@triton.jit
def _csr_segment_attention_fwd(
    logits,
    value,
    perm,
    off,
    out,
    D: tl.constexpr,
    LOG_S0: tl.constexpr,
    LOG_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    BLK: tl.constexpr,
    IDENTITY_PERM: tl.constexpr,
):
    s = tl.program_id(0)
    start = tl.load(off + s)
    end = tl.load(off + s + 1)
    cols = tl.arange(0, BLK)
    mask = cols < D
    dtype = logits.dtype.element_ty

    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    acc = tl.zeros((BLK,), dtype)

    for j in range(start, end):
        if IDENTITY_PERM:
            row = j
        else:
            row = tl.load(perm + j)
        logit = tl.load(
            logits + row * LOG_S0 + cols * LOG_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        m_new = tl.maximum(m, logit)
        old_scale = tl.exp(m - m_new)
        new_scale = tl.exp(logit - m_new)
        denom = denom * old_scale + new_scale
        acc = acc * old_scale + new_scale * val
        m = m_new

    result = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(out + s * D + cols, result, mask=mask)


@triton.jit
def _csr_segment_attention_bwd(
    logits,
    value,
    out,
    gout,
    perm,
    off,
    grad_logits,
    grad_value,
    D: tl.constexpr,
    LOG_S0: tl.constexpr,
    LOG_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    BLK: tl.constexpr,
    IDENTITY_PERM: tl.constexpr,
):
    s = tl.program_id(0)
    start = tl.load(off + s)
    end = tl.load(off + s + 1)
    cols = tl.arange(0, BLK)
    mask = cols < D
    dtype = logits.dtype.element_ty

    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    for j in range(start, end):
        if IDENTITY_PERM:
            row = j
        else:
            row = tl.load(perm + j)
        logit = tl.load(
            logits + row * LOG_S0 + cols * LOG_S1,
            mask=mask,
            other=-float("inf"),
        )
        m_new = tl.maximum(m, logit)
        denom = denom * tl.exp(m - m_new) + tl.exp(logit - m_new)
        m = m_new

    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    go = tl.load(gout + s * D + cols, mask=mask, other=0.0).to(dtype)
    out_s = tl.load(out + s * D + cols, mask=mask, other=0.0).to(dtype)

    for j in range(start, end):
        if IDENTITY_PERM:
            row = j
        else:
            row = tl.load(perm + j)
        logit = tl.load(
            logits + row * LOG_S0 + cols * LOG_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        alpha = tl.exp(logit - m) * inv_denom
        offs = row * D + cols
        tl.store(grad_value + offs, alpha * go, mask=mask)
        tl.store(grad_logits + offs, alpha * go * (val - out_s), mask=mask)


@triton.jit
def _paired_csr_segment_attention_fwd(
    logits_a,
    logits_b,
    value,
    perm_a,
    off_a,
    perm_b,
    off_b,
    out_a,
    out_b,
    max_a,
    denom_a,
    max_b,
    denom_b,
    D: tl.constexpr,
    LOGA_S0: tl.constexpr,
    LOGA_S1: tl.constexpr,
    LOGB_S0: tl.constexpr,
    LOGB_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    BLK: tl.constexpr,
    IDENTITY_A: tl.constexpr,
    IDENTITY_B: tl.constexpr,
):
    s = tl.program_id(0)
    cols = tl.arange(0, BLK)
    mask = cols < D
    dtype = logits_a.dtype.element_ty

    start = tl.load(off_a + s)
    end = tl.load(off_a + s + 1)
    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    acc = tl.zeros((BLK,), dtype)
    for j in range(start, end):
        if IDENTITY_A:
            row = j
        else:
            row = tl.load(perm_a + j)
        logit = tl.load(
            logits_a + row * LOGA_S0 + cols * LOGA_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        m_new = tl.maximum(m, logit)
        old_scale = tl.exp(m - m_new)
        new_scale = tl.exp(logit - m_new)
        denom = denom * old_scale + new_scale
        acc = acc * old_scale + new_scale * val
        m = m_new
    result = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(out_a + s * D + cols, result, mask=mask)
    tl.store(max_a + s * D + cols, m, mask=mask)
    tl.store(denom_a + s * D + cols, denom, mask=mask)

    start = tl.load(off_b + s)
    end = tl.load(off_b + s + 1)
    m = tl.full((BLK,), -float("inf"), dtype)
    denom = tl.zeros((BLK,), dtype)
    acc = tl.zeros((BLK,), dtype)
    for j in range(start, end):
        if IDENTITY_B:
            row = j
        else:
            row = tl.load(perm_b + j)
        logit = tl.load(
            logits_b + row * LOGB_S0 + cols * LOGB_S1,
            mask=mask,
            other=-float("inf"),
        )
        val = tl.load(
            value + row * VAL_S0 + cols * VAL_S1,
            mask=mask,
            other=0.0,
        ).to(dtype)
        m_new = tl.maximum(m, logit)
        old_scale = tl.exp(m - m_new)
        new_scale = tl.exp(logit - m_new)
        denom = denom * old_scale + new_scale
        acc = acc * old_scale + new_scale * val
        m = m_new
    result = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(out_b + s * D + cols, result, mask=mask)
    tl.store(max_b + s * D + cols, m, mask=mask)
    tl.store(denom_b + s * D + cols, denom, mask=mask)


@triton.jit
def _paired_segment_attention_bwd_rows(
    logits_a,
    logits_b,
    value,
    gout_a,
    gout_b,
    out_a,
    out_b,
    max_a,
    denom_a,
    max_b,
    denom_b,
    segment_a,
    segment_b,
    grad_logits_a,
    grad_logits_b,
    grad_value,
    N: tl.constexpr,
    D: tl.constexpr,
    LOGA_S0: tl.constexpr,
    LOGA_S1: tl.constexpr,
    LOGB_S0: tl.constexpr,
    LOGB_S1: tl.constexpr,
    VAL_S0: tl.constexpr,
    VAL_S1: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    mask = (rows[:, None] < N) & (cols[None, :] < D)
    seg_a = tl.load(segment_a + rows, mask=rows < N, other=0)
    seg_b = tl.load(segment_b + rows, mask=rows < N, other=0)

    logit_a = tl.load(
        logits_a + rows[:, None] * LOGA_S0 + cols[None, :] * LOGA_S1,
        mask=mask,
        other=-float("inf"),
    )
    logit_b = tl.load(
        logits_b + rows[:, None] * LOGB_S0 + cols[None, :] * LOGB_S1,
        mask=mask,
        other=-float("inf"),
    )
    val = tl.load(
        value + rows[:, None] * VAL_S0 + cols[None, :] * VAL_S1,
        mask=mask,
        other=0.0,
    )

    stat_a = seg_a[:, None] * D + cols[None, :]
    stat_b = seg_b[:, None] * D + cols[None, :]
    ma = tl.load(max_a + stat_a, mask=mask, other=-float("inf"))
    da = tl.load(denom_a + stat_a, mask=mask, other=0.0)
    mb = tl.load(max_b + stat_b, mask=mask, other=-float("inf"))
    db = tl.load(denom_b + stat_b, mask=mask, other=0.0)

    alpha_a = tl.exp(logit_a - ma) * tl.where(da > 0.0, 1.0 / da, 0.0)
    alpha_b = tl.exp(logit_b - mb) * tl.where(db > 0.0, 1.0 / db, 0.0)
    go_a = tl.load(gout_a + stat_a, mask=mask, other=0.0)
    go_b = tl.load(gout_b + stat_b, mask=mask, other=0.0)
    out_a_row = tl.load(out_a + stat_a, mask=mask, other=0.0)
    out_b_row = tl.load(out_b + stat_b, mask=mask, other=0.0)

    offs = rows[:, None] * D + cols[None, :]
    tl.store(grad_logits_a + offs, alpha_a * go_a * (val - out_a_row), mask=mask)
    tl.store(grad_logits_b + offs, alpha_b * go_b * (val - out_b_row), mask=mask)
    tl.store(grad_value + offs, alpha_a * go_a + alpha_b * go_b, mask=mask)


class _SegmentAttention(Function):
    @staticmethod
    def forward(ctx, logits, value, segment, num_segment, perm, off, identity_perm):
        if logits.shape != value.shape:
            raise ValueError(
                f"logits and value must have the same shape, got {logits.shape} and {value.shape}"
            )
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape {logits.shape}")
        logits = logits.contiguous() if logits.stride(1) != 1 else logits
        value = value.contiguous() if value.stride(1) != 1 else value
        segment = segment.to(torch.int64)
        n_rows, dim = logits.shape
        num_segment = int(num_segment)
        log_s0, log_s1 = logits.stride()
        val_s0, val_s1 = value.stride()
        blk = triton.next_power_of_2(dim)

        identity_perm = bool(identity_perm)
        if identity_perm and off is None:
            raise ValueError("identity CSR metadata requires offsets")
        if not identity_perm and (perm is None or off is None):
            perm, off = segment_attention_csr_metadata(segment, num_segment)
        if identity_perm:
            perm = segment
        out = torch.empty((num_segment, dim), device=logits.device, dtype=logits.dtype)
        _csr_segment_attention_fwd[(num_segment,)](
            logits,
            value,
            perm,
            off,
            out,
            D=dim,
            LOG_S0=log_s0,
            LOG_S1=log_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            BLK=blk,
            IDENTITY_PERM=identity_perm,
            num_warps=_NUM_WARPS,
        )
        ctx.save_for_backward(logits, value, perm, off, out)
        ctx.identity_perm = identity_perm
        return out

    @staticmethod
    def backward(ctx, gout):
        logits, value, perm, off, out = ctx.saved_tensors
        identity_perm = ctx.identity_perm
        gout = gout.contiguous()
        n_rows, dim = logits.shape
        log_s0, log_s1 = logits.stride()
        val_s0, val_s1 = value.stride()
        blk = triton.next_power_of_2(dim)
        grad_logits = torch.empty((n_rows, dim), device=logits.device, dtype=logits.dtype)
        grad_value = torch.empty((n_rows, dim), device=value.device, dtype=value.dtype)
        _csr_segment_attention_bwd[(out.shape[0],)](
            logits,
            value,
            out,
            gout,
            perm,
            off,
            grad_logits,
            grad_value,
            D=dim,
            LOG_S0=log_s0,
            LOG_S1=log_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            BLK=blk,
            IDENTITY_PERM=identity_perm,
            num_warps=_NUM_WARPS,
        )
        return grad_logits, grad_value, None, None, None, None, None


class _PairedSegmentAttention(Function):
    @staticmethod
    def forward(
        ctx,
        logits_a,
        logits_b,
        value,
        segment_a,
        segment_b,
        num_segment,
        perm_a,
        off_a,
        perm_b,
        off_b,
        identity_a,
        identity_b,
    ):
        if logits_a.shape != logits_b.shape or logits_a.shape != value.shape:
            raise ValueError(
                "logits_a, logits_b, and value must have the same shape, got "
                f"{logits_a.shape}, {logits_b.shape}, {value.shape}"
            )
        if logits_a.dim() != 2:
            raise ValueError(f"logits must be 2D, got {logits_a.shape}")
        logits_a = logits_a.contiguous() if logits_a.stride(1) != 1 else logits_a
        logits_b = logits_b.contiguous() if logits_b.stride(1) != 1 else logits_b
        value = value.contiguous() if value.stride(1) != 1 else value
        segment_a = segment_a.to(torch.int64).contiguous()
        segment_b = segment_b.to(torch.int64).contiguous()
        n_rows, dim = logits_a.shape
        num_segment = int(num_segment)
        loga_s0, loga_s1 = logits_a.stride()
        logb_s0, logb_s1 = logits_b.stride()
        val_s0, val_s1 = value.stride()
        blk = triton.next_power_of_2(dim)

        identity_a = bool(identity_a)
        identity_b = bool(identity_b)
        if identity_a and off_a is None:
            raise ValueError("identity CSR metadata for segment_a requires offsets")
        if identity_b and off_b is None:
            raise ValueError("identity CSR metadata for segment_b requires offsets")
        if not identity_a and (perm_a is None or off_a is None):
            perm_a, off_a = segment_attention_csr_metadata(segment_a, num_segment)
        if not identity_b and (perm_b is None or off_b is None):
            perm_b, off_b = segment_attention_csr_metadata(segment_b, num_segment)
        if identity_a:
            perm_a = segment_a
        if identity_b:
            perm_b = segment_b

        out_a = torch.empty((num_segment, dim), device=logits_a.device, dtype=logits_a.dtype)
        out_b = torch.empty_like(out_a)
        max_a = torch.empty_like(out_a)
        denom_a = torch.empty_like(out_a)
        max_b = torch.empty_like(out_a)
        denom_b = torch.empty_like(out_a)
        _paired_csr_segment_attention_fwd[(num_segment,)](
            logits_a,
            logits_b,
            value,
            perm_a,
            off_a,
            perm_b,
            off_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
            D=dim,
            LOGA_S0=loga_s0,
            LOGA_S1=loga_s1,
            LOGB_S0=logb_s0,
            LOGB_S1=logb_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            BLK=blk,
            IDENTITY_A=identity_a,
            IDENTITY_B=identity_b,
            num_warps=_NUM_WARPS,
        )
        ctx.save_for_backward(
            logits_a,
            logits_b,
            value,
            segment_a,
            segment_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
        )
        return out_a, out_b

    @staticmethod
    def backward(ctx, gout_a, gout_b):
        (
            logits_a,
            logits_b,
            value,
            segment_a,
            segment_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
        ) = ctx.saved_tensors
        gout_a = gout_a.contiguous()
        gout_b = gout_b.contiguous()
        n_rows, dim = logits_a.shape
        loga_s0, loga_s1 = logits_a.stride()
        logb_s0, logb_s1 = logits_b.stride()
        val_s0, val_s1 = value.stride()
        bd = min(triton.next_power_of_2(dim), _PAIRED_BWD_BD)
        grad_logits_a = torch.empty((n_rows, dim), device=logits_a.device, dtype=logits_a.dtype)
        grad_logits_b = torch.empty_like(grad_logits_a)
        grad_value = torch.empty((n_rows, dim), device=value.device, dtype=value.dtype)
        _paired_segment_attention_bwd_rows[
            (triton.cdiv(n_rows, _PAIRED_BWD_BM), triton.cdiv(dim, bd))
        ](
            logits_a,
            logits_b,
            value,
            gout_a,
            gout_b,
            out_a,
            out_b,
            max_a,
            denom_a,
            max_b,
            denom_b,
            segment_a,
            segment_b,
            grad_logits_a,
            grad_logits_b,
            grad_value,
            N=n_rows,
            D=dim,
            LOGA_S0=loga_s0,
            LOGA_S1=loga_s1,
            LOGB_S0=logb_s0,
            LOGB_S1=logb_s1,
            VAL_S0=val_s0,
            VAL_S1=val_s1,
            BM=_PAIRED_BWD_BM,
            BD=bd,
            num_warps=4,
        )
        return (
            grad_logits_a,
            grad_logits_b,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def segment_attention_csr_metadata(
    segment: torch.Tensor,
    num_segment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    segment = segment.to(torch.int64)
    num_segment = int(num_segment)
    perm = torch.argsort(segment)
    sorted_segment = segment[perm]
    off = torch.searchsorted(
        sorted_segment,
        torch.arange(num_segment + 1, device=segment.device, dtype=segment.dtype),
    ).int()
    return perm, off


def _eager_segment_attention(
    logits: torch.Tensor,
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
) -> torch.Tensor:
    dim = logits.shape[1]
    segment_expanded = segment.unsqueeze(1).expand(-1, dim)
    max_per_segment = logits.new_full((num_segment, dim), float("-inf"))
    max_per_segment = max_per_segment.scatter_reduce(
        0,
        segment_expanded,
        logits,
        reduce="amax",
        include_self=False,
    )
    centered = logits - max_per_segment[segment]
    weight = centered.exp()
    denom = logits.new_zeros((num_segment, dim)).scatter_reduce(
        0,
        segment_expanded,
        weight,
        reduce="sum",
        include_self=False,
    )
    alpha = weight / denom[segment]
    out = logits.new_zeros((num_segment, dim))
    return out.index_add_(0, segment, alpha * value)


def segment_attention(
    logits: torch.Tensor,
    value: torch.Tensor,
    segment: torch.Tensor,
    num_segment: int,
    perm: torch.Tensor | None = None,
    off: torch.Tensor | None = None,
    identity_perm: bool = False,
) -> torch.Tensor:
    if logits.is_cuda and value.is_cuda:
        return _SegmentAttention.apply(
            logits,
            value,
            segment,
            int(num_segment),
            perm,
            off,
            bool(identity_perm),
        )
    return _eager_segment_attention(logits, value, segment, int(num_segment))


def paired_segment_attention(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    value: torch.Tensor,
    segment_a: torch.Tensor,
    segment_b: torch.Tensor,
    num_segment: int,
    perm_a: torch.Tensor | None = None,
    off_a: torch.Tensor | None = None,
    perm_b: torch.Tensor | None = None,
    off_b: torch.Tensor | None = None,
    identity_a: bool = False,
    identity_b: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits_a.is_cuda and logits_b.is_cuda and value.is_cuda:
        return _PairedSegmentAttention.apply(
            logits_a,
            logits_b,
            value,
            segment_a,
            segment_b,
            int(num_segment),
            perm_a,
            off_a,
            perm_b,
            off_b,
            bool(identity_a),
            bool(identity_b),
        )
    return (
        _eager_segment_attention(logits_a, value, segment_a, int(num_segment)),
        _eager_segment_attention(logits_b, value, segment_b, int(num_segment)),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for n_rows, dim, num_segment in [(26174, 128, 1165), (152446, 128, 13087), (257, 32, 31)]:
        segment = torch.randint(0, num_segment, (n_rows,), device=dev)
        logits = (torch.randn(n_rows, dim, device=dev) * 3).requires_grad_(True)
        value = torch.randn(n_rows, dim, device=dev, requires_grad=True)
        logits_tri = logits.detach().clone().requires_grad_(True)
        value_tri = value.detach().clone().requires_grad_(True)
        ref = _eager_segment_attention(logits, value, segment, num_segment)
        tri = segment_attention(logits_tri, value_tri, segment, num_segment)
        fwd_err = (ref - tri).abs().max().item()
        grad = torch.randn_like(ref)
        ref.backward(grad)
        tri.backward(grad)
        bwd_err = max(
            (logits.grad - logits_tri.grad).abs().max().item(),
            (value.grad - value_tri.grad).abs().max().item(),
        )
        print(
            f"N={n_rows} D={dim} S={num_segment}: "
            f"fwd max|d|={fwd_err:.2e}  bwd max|d|={bwd_err:.2e}"
        )
