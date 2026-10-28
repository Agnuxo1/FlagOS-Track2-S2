"""Micro-benchmark of the FlagGems ops on MiniCPM5-2B hot shapes.

Compares FlagGems v5.3.5 kernels against candidate kernels (if importable) and a
PyTorch fp32 reference. Correctness first, then call-level CUDA-event timing (input clones included),
3 seeds. Writes results/micro/<tag>.csv.
"""
import argparse
import csv
import importlib
import os
import statistics
import sys

import torch

HIDDEN = 2048
INTER = 6144
TOKENS = [64, 256, 1024, 4096, 16384]


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # microseconds


def ref_add_rms(x, r, w, eps):
    c = x.float() + r.float()
    y = c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype), c.to(x.dtype)


def ref_silu_mul(x):
    g, u = x.float().chunk(2, -1)
    return (torch.nn.functional.silu(g) * u).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--candidate", default="")  # module path exposing add_rms_norm(x,r,w,eps), silu_and_mul(x)
    a = ap.parse_args()
    import flag_gems

    cand = importlib.import_module(a.candidate) if a.candidate else None
    rows = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        for T in TOKENS:
            x = torch.randn(T, HIDDEN, device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
            eps = 1e-6
            yr, cr = ref_add_rms(x, r, w, eps)
            impls = {"flaggems": lambda xx, rr: flag_gems.fused_add_rms_norm(xx, rr, [HIDDEN], w, eps)}
            if cand is not None:
                impls["candidate"] = lambda xx, rr: cand.add_rms_norm(xx, rr, w, eps)
            for name, f in impls.items():
                xx, rr = x.clone(), r.clone()
                out = f(xx, rr)
                y, c = out if isinstance(out, tuple) else (xx, rr)
                torch.testing.assert_close(y, yr, rtol=1.6e-2, atol=2e-2)
                torch.testing.assert_close(c, cr, rtol=1.6e-2, atol=2e-2)
                err = max((y.float() - yr.float()).abs().max().item(), (c.float() - cr.float()).abs().max().item())
                t = bench(lambda: f(x.clone(), r.clone()))
                rows.append(dict(op="fused_add_rms_norm", impl=name, tokens=T, seed=seed, us=t, max_abs_err=err))
            g = torch.randn(T, 2 * INTER, device="cuda", dtype=torch.bfloat16)
            sr = ref_silu_mul(g)
            impls = {"flaggems": lambda gg: flag_gems.silu_and_mul(*gg.chunk(2, -1))}
            if cand is not None:
                impls["candidate"] = lambda gg: cand.silu_and_mul(gg)
            for name, f in impls.items():
                y = f(g)
                torch.testing.assert_close(y, sr, rtol=1.6e-2, atol=2e-2)
                err = (y.float() - sr.float()).abs().max().item()
                t = bench(lambda: f(g))
                rows.append(dict(op="silu_and_mul", impl=name, tokens=T, seed=seed, us=t, max_abs_err=err))
    os.makedirs("results/micro", exist_ok=True)
    path = f"results/micro/{a.tag}.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    agg = {}
    for r_ in rows:
        agg.setdefault((r_["op"], r_["impl"], r_["tokens"]), []).append(r_["us"])
    for k, v in sorted(agg.items()):
        print(k, f"{statistics.mean(v):.1f}us ±{statistics.pstdev(v):.1f}")
    print("wrote", path, file=sys.stderr)


if __name__ == "__main__":
    main()
