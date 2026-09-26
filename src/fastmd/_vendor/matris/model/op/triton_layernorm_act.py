"""Fusion C: Triton fused LayerNorm + activation (SiLU/Sigmoid).

Replaces nn.LayerNorm(x) -> SiLU/Sigmoid (two kernels + 2 HBM round-trips of the
[rows,128] tensor) with one fused kernel (1 read, 1 write). act='silu' for the
GatedMLP core, 'sigmoid' for the gate, 'none' for plain LN (edge_init_norm).

Validated standalone (fwd allclose vs nn.LayerNorm+act, FP64 gradcheck) before
model integration. D=128 fixed (one row = one program).
"""
import torch, triton
import triton.language as tl
from torch.autograd import Function

ACT_NONE, ACT_SILU, ACT_SIGMOID = 0, 1, 2


@triton.jit
def _ln_act_fwd(x, w, b, out, mean, rstd, D: tl.constexpr, BLK: tl.constexpr, EPS: tl.constexpr, ACT: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLK); m = c < D
    xr = tl.load(x + row * D + c, mask=m, other=0.0).to(tl.float32)
    mu = tl.sum(xr, axis=0) / D
    xc = tl.where(m, xr - mu, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rs = 1.0 / tl.sqrt(var + EPS)
    xhat = xc * rs
    y = xhat * tl.load(w + c, mask=m, other=0.0) + tl.load(b + c, mask=m, other=0.0)
    if ACT == 1:
        o = y * tl.sigmoid(y)
    elif ACT == 2:
        o = tl.sigmoid(y)
    else:
        o = y
    tl.store(out + row * D + c, o, mask=m)
    tl.store(mean + row, mu)
    tl.store(rstd + row, rs)


@triton.jit
def _ln_act_bwd(x, w, b, dout, mean, rstd, dx, dw_part, db_part,
               D: tl.constexpr, BLK: tl.constexpr, ACT: tl.constexpr, NEED_WB: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLK); m = c < D
    xr = tl.load(x + row * D + c, mask=m, other=0.0).to(tl.float32)
    mu = tl.load(mean + row); rs = tl.load(rstd + row)
    xhat = (xr - mu) * rs
    wv = tl.load(w + c, mask=m, other=0.0)
    y = xhat * wv + tl.load(b + c, mask=m, other=0.0)
    go = tl.load(dout + row * D + c, mask=m, other=0.0).to(tl.float32)
    # d(out)/d(y)
    sig = tl.sigmoid(y)
    if ACT == 1:
        dy = go * (sig * (1.0 + y * (1.0 - sig)))
    elif ACT == 2:
        dy = go * (sig * (1.0 - sig))
    else:
        dy = go
    # weight/bias partial grads (per-row; reduced across rows by caller) — only when training
    if NEED_WB:
        tl.store(dw_part + row * D + c, dy * xhat, mask=m)
        tl.store(db_part + row * D + c, dy, mask=m)
    # LayerNorm input grad
    dxhat = dy * wv
    mean_dxhat = tl.sum(tl.where(m, dxhat, 0.0), axis=0) / D
    mean_dxhat_xhat = tl.sum(tl.where(m, dxhat * xhat, 0.0), axis=0) / D
    dxr = (dxhat - mean_dxhat - xhat * mean_dxhat_xhat) * rs
    tl.store(dx + row * D + c, dxr, mask=m)


class _LNAct(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, act):
        x = x.contiguous()
        N, D = x.shape
        BLK = triton.next_power_of_2(D)
        out = torch.empty_like(x)
        mean = torch.empty(N, device=x.device, dtype=torch.float32)
        rstd = torch.empty(N, device=x.device, dtype=torch.float32)
        _ln_act_fwd[(N,)](x, weight.contiguous(), bias.contiguous(), out, mean, rstd,
                          D=D, BLK=BLK, EPS=eps, ACT=act)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.act = act
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        N, D = x.shape
        BLK = triton.next_power_of_2(D)
        dx = torch.empty_like(x)
        need_wb = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        if need_wb:
            dw_part = torch.empty_like(x); db_part = torch.empty_like(x)
            _ln_act_bwd[(N,)](x, weight, bias, dout.contiguous(), mean, rstd, dx, dw_part, db_part,
                              D=D, BLK=BLK, ACT=ctx.act, NEED_WB=True)
            return dx, dw_part.sum(0), db_part.sum(0), None, None
        # inference (forces): only dx needed -> skip weight/bias grad work
        _ln_act_bwd[(N,)](x, weight, bias, dout.contiguous(), mean, rstd, dx, dx, dx,
                          D=D, BLK=BLK, ACT=ctx.act, NEED_WB=False)
        return dx, None, None, None, None


def fused_ln_act(x, weight, bias, eps=1e-5, act=ACT_SILU):
    if x.is_cuda:
        return _LNAct.apply(x, weight, bias, eps, act)
    import torch.nn.functional as F
    y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return {ACT_NONE: y, ACT_SILU: F.silu(y), ACT_SIGMOID: torch.sigmoid(y)}[act]


if __name__ == "__main__":
    import torch.nn.functional as F
    dev = "cuda"; torch.manual_seed(0)
    for act, fn in [(ACT_SILU, F.silu), (ACT_SIGMOID, torch.sigmoid), (ACT_NONE, lambda z: z)]:
        for (N, D) in [(26174, 128), (152446, 128), (1165, 128)]:
            x = torch.randn(N, D, device=dev); w = torch.randn(D, device=dev); b = torch.randn(D, device=dev)
            x1 = x.clone().requires_grad_(True); w1 = w.clone().requires_grad_(True); b1 = b.clone().requires_grad_(True)
            x2 = x.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True); b2 = b.clone().requires_grad_(True)
            ref = fn(F.layer_norm(x1, (D,), w1, b1, 1e-5))
            tri = fused_ln_act(x2, w2, b2, 1e-5, act)
            fe = (ref - tri).abs().max().item()
            g = torch.randn_like(ref); ref.backward(g); tri.backward(g)
            de = max((x1.grad-x2.grad).abs().max().item(), (w1.grad-w2.grad).abs().max().item(), (b1.grad-b2.grad).abs().max().item())
            print(f"act={act} N={N}: fwd={fe:.2e} grad={de:.2e}")
    # FP64 gradcheck
    Ns, Ds = 40, 16
    xe = torch.randn(Ns, Ds, device=dev, dtype=torch.float64, requires_grad=True)
    we = torch.randn(Ds, device=dev, dtype=torch.float64, requires_grad=True)
    be = torch.randn(Ds, device=dev, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda a,c,d: _LNAct.apply(a,c,d,1e-5,ACT_SILU), (xe,we,be), eps=1e-6, atol=1e-4)
    print("FP64 gradcheck (silu):", ok)
