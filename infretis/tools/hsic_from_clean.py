#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

SHIFT_BLOCK_ROWS = 256


# -----------------------------
# Basic helpers
# -----------------------------
def center_gram_hkh(K: np.ndarray) -> np.ndarray:
    # This is exactly HKH, using K - rowmean(K) - colmean(K) + grandmean(K).
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    grand_mean = float(K.mean())
    return K - row_mean - col_mean + grand_mean


def hsic_from_centered(Kc: np.ndarray, Lc: np.ndarray, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(np.sum(Kc * Lc) / denom)


def hsic_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    if n < 3:
        return float("nan")
    Kc = center_gram_hkh(K)
    Lc = center_gram_hkh(L)
    # Empirical HSIC: (1/(n-1)^2) * tr(KHLH) == (1/(n-1)^2) * sum(Kc * Lc) for symmetric Grams.
    return hsic_from_centered(Kc, Lc, float((n - 1) ** 2))


def resolve_shift_range(n, shift_min, shift_max, shift_policy):
    if shift_max is None:
        shift_max_eff = n - 1
    else:
        shift_max_eff = min(int(shift_max), n - 1)
    shift_min_eff = max(1, int(shift_min))

    if shift_policy == "band" and int(shift_min) == 1 and shift_max is None:
        lo = max(1, int(np.ceil(0.1 * n)))
        hi = min(n - 1, int(np.floor(0.9 * n)))
        if lo <= hi:
            shift_min_eff, shift_max_eff = lo, hi

    return shift_min_eff, shift_max_eff


def hsic_shift_null_centered(
    Kc: np.ndarray,
    Lc: np.ndarray,
    denom: float,
    shift_min=1,
    shift_max=None,
    max_shifts=80,
    seed=0,
    block_rows: int = SHIFT_BLOCK_ROWS,
    shift_policy: str = "band",
):
    n = Kc.shape[0]
    if n < 3:
        return np.array([], float)
    shift_min, shift_max = resolve_shift_range(n, shift_min, shift_max, shift_policy)
    if shift_min > shift_max:
        return np.array([], float)

    shifts = np.arange(shift_min, shift_max + 1)
    if max_shifts is not None and len(shifts) > max_shifts:
        rng = np.random.default_rng(seed)
        shifts = rng.choice(shifts, size=max_shifts, replace=False)
        shifts.sort()

    base = np.arange(n)
    col_base = np.arange(n)
    br = max(1, int(block_rows))
    null = np.empty(len(shifts), float)
    for t, s in enumerate(shifts):
        col = (col_base + s) % n
        tot = 0.0
        for i0 in range(0, n, br):
            i1 = min(n, i0 + br)
            rows = base[i0:i1]
            rows_s = (rows + s) % n
            Lblk = Lc[rows_s][:, col]
            tot += float(np.sum(Kc[i0:i1, :] * Lblk))
        null[t] = tot / denom
    return null


def hsic_shift_null(
    K: np.ndarray,
    L: np.ndarray,
    shift_min=1,
    shift_max=None,
    max_shifts=80,
    seed=0,
    block_rows: int = SHIFT_BLOCK_ROWS,
    shift_policy: str = "band",
):
    if K.shape[0] < 3:
        return np.array([], float)
    n = K.shape[0]
    denom = float((n - 1) ** 2)
    Kc = center_gram_hkh(K)
    Lc = center_gram_hkh(L)
    return hsic_shift_null_centered(
        Kc,
        Lc,
        denom,
        shift_min=shift_min,
        shift_max=shift_max,
        max_shifts=max_shifts,
        seed=seed,
        block_rows=block_rows,
        shift_policy=shift_policy,
    )


def hsic_with_null(
    K: np.ndarray,
    L: np.ndarray,
    shift_min=1,
    shift_max=None,
    max_shifts=80,
    seed=0,
    block_rows: int = SHIFT_BLOCK_ROWS,
    shift_policy: str = "band",
):
    n = K.shape[0]
    if n < 3:
        return float("nan"), float("nan"), float("nan"), np.array([], float)
    denom = float((n - 1) ** 2)
    Kc = center_gram_hkh(K)
    Lc = center_gram_hkh(L)
    obs = hsic_from_centered(Kc, Lc, denom)
    null = hsic_shift_null_centered(
        Kc,
        Lc,
        denom,
        shift_min=shift_min,
        shift_max=shift_max,
        max_shifts=max_shifts,
        seed=seed,
        block_rows=block_rows,
        shift_policy=shift_policy,
    )
    if null.size == 0 or np.isnan(obs):
        return obs, float("nan"), float("nan"), null
    q95 = float(np.quantile(null, 0.95))
    p = float((np.sum(null >= obs) + 1) / (null.size + 1))
    return obs, q95, p, null


def median_sigma_vec(X: np.ndarray, nsample=20000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    n = len(X)
    if n < 2:
        return 1.0
    m = min(nsample, n * (n - 1) // 2)
    if m < 1:
        return 1.0
    need = m
    chunks = []
    while need > 0:
        draw = max(int(need * 1.5), 256)
        I = rng.integers(0, n, size=draw)
        J = rng.integers(0, n, size=draw)
        keep = I != J
        if not np.any(keep):
            continue
        diff = X[I[keep]] - X[J[keep]]
        ds = np.linalg.norm(diff, axis=1)
        take = min(need, ds.size)
        chunks.append(ds[:take])
        need -= take
    ds_all = np.concatenate(chunks) if chunks else np.array([], float)
    s = float(np.median(ds_all)) if ds_all.size else 1.0
    return s if s > 1e-12 else 1.0


def sigma_grid(sig0: float, powers):
    return [float(sig0) * (2.0 ** int(p)) for p in powers]


def finite_median(vals):
    arr = np.asarray(vals, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


# -----------------------------
# Block kernel + Gram
# -----------------------------
def block_kernel_rbf(A: np.ndarray, B: np.ndarray, sigma: float) -> float:
    """Mean_{a in A, b in B} exp(-||a-b||^2/(2σ^2)). A,B: (m,d),(n,d)."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.size == 0 or B.size == 0:
        return float("nan")
    An = np.sum(A * A, axis=1)
    Bn = np.sum(B * B, axis=1)
    dist2 = An[:, None] + Bn[None, :] - 2.0 * (A @ B.T)
    np.clip(dist2, 0.0, None, out=dist2)
    return float(np.exp(-dist2 / (2.0 * sigma * sigma)).mean())


def full_gram_multi_sigma(blocks, sigmas, dtype=np.float64):
    sigma_list = [float(s) for s in sigmas]
    n_blocks = len(blocks)
    out = {s: np.zeros((n_blocks, n_blocks), dtype=dtype) for s in sigma_list}
    if n_blocks == 0 or not sigma_list:
        return out

    inv_scales = {s: 1.0 / (2.0 * s * s) for s in sigma_list}
    first = np.asarray(blocks[0], float)
    same_shape = first.ndim == 2 and all(np.asarray(b).shape == first.shape for b in blocks)

    if same_shape:
        m, _d = first.shape
        if m == 0:
            for s in sigma_list:
                out[s].fill(np.nan)
            return out

        X = np.stack([np.asarray(b, float) for b in blocks], axis=0)  # (N,m,d)
        norms = np.sum(X * X, axis=2)  # (N,m)
        for a in range(m):
            Xa = X[:, a, :]
            na = norms[:, a]
            for b in range(m):
                Xb = X[:, b, :]
                nb = norms[:, b]
                dist2 = na[:, None] + nb[None, :] - 2.0 * (Xa @ Xb.T)
                np.clip(dist2, 0.0, None, out=dist2)
                for s in sigma_list:
                    v = np.exp(-dist2 * inv_scales[s])
                    if out[s].dtype != v.dtype:
                        v = v.astype(out[s].dtype, copy=False)
                    out[s] += v
        scale = 1.0 / float(m * m)
        for s in sigma_list:
            out[s] *= scale
            out[s] = 0.5 * (out[s] + out[s].T)
        return out

    for i in range(n_blocks):
        Ai = np.asarray(blocks[i], float)
        for j in range(i, n_blocks):
            Bj = np.asarray(blocks[j], float)
            if Ai.size == 0 or Bj.size == 0:
                vals = {s: float("nan") for s in sigma_list}
            else:
                An = np.sum(Ai * Ai, axis=1)
                Bn = np.sum(Bj * Bj, axis=1)
                dist2 = An[:, None] + Bn[None, :] - 2.0 * (Ai @ Bj.T)
                np.clip(dist2, 0.0, None, out=dist2)
                vals = {s: float(np.exp(-dist2 * inv_scales[s]).mean()) for s in sigma_list}
            for s in sigma_list:
                out[s][i, j] = out[s][j, i] = vals[s]

    for s in sigma_list:
        out[s] = 0.5 * (out[s] + out[s].T)
    return out


def full_gram(blocks, sigma: float, dtype=np.float64) -> np.ndarray:
    sigma = float(sigma)
    return full_gram_multi_sigma(blocks, [sigma], dtype=dtype)[sigma]


def hsic_with_null_from_centered(
    Kc: np.ndarray,
    Lc: np.ndarray,
    denom: float,
    shift_min=1,
    shift_max=None,
    max_shifts=80,
    seed=0,
    block_rows: int = SHIFT_BLOCK_ROWS,
    shift_policy: str = "band",
):
    obs = hsic_from_centered(Kc, Lc, denom)
    null = hsic_shift_null_centered(
        Kc,
        Lc,
        denom,
        shift_min=shift_min,
        shift_max=shift_max,
        max_shifts=max_shifts,
        seed=seed,
        block_rows=block_rows,
        shift_policy=shift_policy,
    )
    if null.size == 0 or np.isnan(obs):
        return obs, float("nan"), float("nan"), null
    q95 = float(np.quantile(null, 0.95))
    p = float((np.sum(null >= obs) + 1) / (null.size + 1))
    return obs, q95, p, null


def hsic_self_from_centered(Kc: np.ndarray, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(np.sum(Kc * Kc) / denom)


def nhsic_from_centered(Kc: np.ndarray, Lc: np.ndarray, denom: float) -> float:
    hs = hsic_from_centered(Kc, Lc, denom)
    hk = hsic_self_from_centered(Kc, denom)
    hl = hsic_self_from_centered(Lc, denom)
    if not (np.isfinite(hs) and np.isfinite(hk) and np.isfinite(hl)):
        return float("nan")
    if hk <= 0 or hl <= 0:
        return float("nan")
    return float(hs / np.sqrt(hk * hl))


def nhsic_shift_null_centered(
    Kc: np.ndarray,
    Lc: np.ndarray,
    denom: float,
    shift_min=1,
    shift_max=None,
    max_shifts=80,
    seed=0,
    block_rows: int = SHIFT_BLOCK_ROWS,
    shift_policy: str = "band",
):
    n = Kc.shape[0]
    if n < 3:
        return np.array([], float)
    shift_min, shift_max = resolve_shift_range(n, shift_min, shift_max, shift_policy)
    if shift_min > shift_max:
        return np.array([], float)

    hk = hsic_self_from_centered(Kc, denom)
    if not np.isfinite(hk) or hk <= 0:
        return np.array([], float)

    shifts = np.arange(shift_min, shift_max + 1)
    if max_shifts is not None and len(shifts) > max_shifts:
        rng = np.random.default_rng(seed)
        shifts = rng.choice(shifts, size=max_shifts, replace=False)
        shifts.sort()

    base = np.arange(n)
    col_base = np.arange(n)
    br = max(1, int(block_rows))
    null = np.empty(len(shifts), float)
    for t, s in enumerate(shifts):
        col = (col_base + s) % n
        tot = 0.0
        tot_ll = 0.0
        for i0 in range(0, n, br):
            i1 = min(n, i0 + br)
            rows = base[i0:i1]
            rows_s = (rows + s) % n
            Lblk = Lc[rows_s][:, col]
            Kblk = Kc[i0:i1, :]
            tot += float(np.sum(Kblk * Lblk))
            tot_ll += float(np.sum(Lblk * Lblk))

        hs = tot / denom
        hl = tot_ll / denom
        if hl <= 0 or not np.isfinite(hl):
            null[t] = float("nan")
        else:
            null[t] = float(hs / np.sqrt(hk * hl))
    return null


def nhsic_with_null_from_centered(
    Kc: np.ndarray,
    Lc: np.ndarray,
    denom: float,
    shift_min=1,
    shift_max=None,
    max_shifts=80,
    seed=0,
    block_rows: int = SHIFT_BLOCK_ROWS,
    shift_policy: str = "band",
):
    obs = nhsic_from_centered(Kc, Lc, denom)
    null = nhsic_shift_null_centered(
        Kc,
        Lc,
        denom,
        shift_min=shift_min,
        shift_max=shift_max,
        max_shifts=max_shifts,
        seed=seed,
        block_rows=block_rows,
        shift_policy=shift_policy,
    )
    valid = null[np.isfinite(null)]
    if valid.size == 0 or np.isnan(obs):
        return obs, float("nan"), float("nan"), null
    q95 = float(np.quantile(valid, 0.95))
    p = float((np.sum(valid >= obs) + 1) / (valid.size + 1))
    return obs, q95, p, null


def parse_k_grid(k_grid: str, k_eff: int):
    if str(k_grid).strip().lower() == "all":
        return list(range(1, int(k_eff) + 1))
    out = []
    for tok in str(k_grid).split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            k = int(t)
        except ValueError as exc:
            raise SystemExit(f"invalid --k-grid token: {t!r}") from exc
        if 1 <= k <= int(k_eff):
            out.append(k)
    return sorted(set(out))


def gram_offdiag_stats(G: np.ndarray):
    n = G.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = G[mask]
    return {
        "mu_off": float(off.mean()),
        "var_off": float(off.var()),
        "q05_off": float(np.quantile(off, 0.05)),
        "q50_off": float(np.quantile(off, 0.50)),
        "q95_off": float(np.quantile(off, 0.95)),
    }


def centered_energy_ratio(G: np.ndarray):
    Gc = center_gram_hkh(G)
    num = np.linalg.norm(Gc, ord="fro")
    den = np.linalg.norm(G, ord="fro")
    return float(num / den) if den > 0 else float("nan"), Gc


def effective_rank(Gc: np.ndarray, eps=1e-12):
    A = 0.5 * (Gc + Gc.T)
    w = np.linalg.eigvalsh(A)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= eps:
        return 1.0
    p = w / s
    p = p[p > eps]
    hval = -float(np.sum(p * np.log(p)))
    return float(np.exp(hval))


def gram_degeneracy_flags(
    mu_off: float,
    erank: float,
    e_ratio: float,
    mu_min: float,
    mu_max: float,
    erank_min: float,
    e_ratio_min: float,
):
    if not (np.isfinite(mu_off) and np.isfinite(erank) and np.isfinite(e_ratio)):
        return ["nonfinite"]
    flags = []
    if mu_off <= mu_min:
        flags.append("too_sparse")
    if mu_off >= mu_max:
        flags.append("too_dense")
    if erank < erank_min:
        flags.append("low_erank")
    if e_ratio < e_ratio_min:
        flags.append("low_energy")
    return flags


def choose_window_starts(
    N: int,
    window_size: int,
    kmax: int,
    stride: int,
    max_windows: int,
    mode: str,
    seed: int,
):
    if N < 3:
        return [], 0, 0

    n = min(int(window_size), N - 1)
    k_eff = min(int(kmax), N - n)
    if n < 3 or k_eff < 1:
        return [], 0, 0

    max_start = N - (n + k_eff)
    starts = np.arange(0, max_start + 1, max(1, int(stride)), dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)

    cap = max(1, int(max_windows))
    if starts.size > cap:
        if mode == "start":
            starts = starts[:cap]
        elif mode == "tail":
            starts = starts[-cap:]
        elif mode == "grid":
            idx = np.linspace(0, starts.size - 1, num=cap)
            idx = np.unique(np.round(idx).astype(int))
            starts = starts[idx]
        elif mode == "random":
            rng = np.random.default_rng(seed)
            starts = np.sort(rng.choice(starts, size=cap, replace=False))
        else:
            raise ValueError(f"Unknown window mode: {mode}")

    return starts.tolist(), int(n), int(k_eff)


def indep_lag_from_curve(rows, alpha=0.05):
    """rows: list of (k, hsic_med, null95_med, p_med)."""
    for (k, hs, q95, p) in rows:
        if np.isfinite(p) and np.isfinite(q95) and np.isfinite(hs):
            if (p >= alpha) and (hs <= q95):
                return k
    return None


def dependence_frequency(rows, alpha=0.05):
    good = [r for r in rows if np.isfinite(r[1]) and np.isfinite(r[2]) and np.isfinite(r[3])]
    if not good:
        return float("nan")
    dep = sum((hs > q95) and (p < alpha) for (_, hs, q95, p) in good)
    return float(dep / len(good))


# -----------------------------
# IO + grouping
# -----------------------------
def load_group_pkl(pkl_path: Path):
    with pkl_path.open("rb") as fh:
        payload = pickle.load(fh)
    blocks = payload["blocks"]
    meta = payload["meta"]
    group_id = payload.get("group_id", pkl_path.stem.replace("blocks_", ""))
    whitened = bool(payload.get("whitened", False))
    return group_id, blocks, meta, whitened


def split_sequences_by_pin(blocks, meta):
    by_pin = defaultdict(list)
    for X, m in zip(blocks, meta):
        pin = int(m.get("pin"))
        cstep = int(m.get("cstep"))
        by_pin[pin].append((cstep, X, m))
    out = {}
    for pin, rows in by_pin.items():
        rows.sort(key=lambda t: t[0])
        out[pin] = rows
    return out


def summarize_curves(agg_by_sigma_k, sigmas, metric_prefix: str):
    out = {}
    y = metric_prefix
    y_q95 = f"{metric_prefix}_null95"
    y_p = f"{metric_prefix}_p"
    for sigma in sigmas:
        rows = []
        keys = sorted(k for (s, k) in agg_by_sigma_k.keys() if s == sigma)
        for k in keys:
            cell = agg_by_sigma_k[(sigma, k)]
            hs_m = finite_median(cell[y])
            q95_m = finite_median(cell[y_q95])
            p_m = finite_median(cell[y_p])
            rows.append((k, hs_m, q95_m, p_m))
        out[sigma] = rows
    return out


def run_self_test(args):
    rng = np.random.default_rng(args.seed)
    N = 260
    d = 3

    iid = [rng.normal(size=(6, d)) for _ in range(N)]
    sticky = []
    state = np.zeros(d, float)
    for _ in range(N):
        state = 0.97 * state + rng.normal(scale=0.2, size=d)
        sticky.append((state + 0.1 * rng.normal(size=(6, d))).astype(float))

    def one_curve(blocks, label):
        starts, n, k_eff = choose_window_starts(
            N=len(blocks),
            window_size=min(args.window_size, 200),
            kmax=min(args.kmax, 8),
            stride=max(1, min(args.stride, 100)),
            max_windows=min(args.max_windows, 3),
            mode="start",
            seed=args.seed,
        )
        if not starts:
            raise RuntimeError(f"self-test failed: no windows for {label}")

        s0 = starts[0]
        sigma0 = median_sigma_vec(np.vstack(blocks[s0 : s0 + n]), seed=args.seed + 123)
        sigma = sigma_grid(sigma0, [0])[0]
        gram_dtype = np.float32 if args.gram_dtype == "float32" else np.float64
        tested_set = set(range(1, k_eff + 1))

        agg = defaultdict(
            lambda: {
                "hsic": [],
                "hsic_null95": [],
                "hsic_p": [],
                "nhsic": [],
                "nhsic_null95": [],
                "nhsic_p": [],
            }
        )
        for ws in starts:
            sub = blocks[ws : ws + n + k_eff]
            G = full_gram_multi_sigma(sub, [sigma], dtype=gram_dtype)[sigma]
            K = G[:n, :n]
            Kc = center_gram_hkh(K)
            denom = float((n - 1) ** 2)
            for k in range(1, k_eff + 1):
                L = G[k : k + n, k : k + n]
                Lc = center_gram_hkh(L)
                if k in tested_set:
                    hs, hs_q95, hs_p, _ = hsic_with_null_from_centered(
                        Kc,
                        Lc,
                        denom,
                        shift_min=args.shift_min,
                        shift_max=args.shift_max,
                        max_shifts=min(args.max_shifts, 40),
                        seed=args.seed + 9000 + ws + k,
                        shift_policy="full",
                    )
                    nhs, nhs_q95, nhs_p, _ = nhsic_with_null_from_centered(
                        Kc,
                        Lc,
                        denom,
                        shift_min=args.shift_min,
                        shift_max=args.shift_max,
                        max_shifts=min(args.max_shifts, 40),
                        seed=args.seed + 9000 + ws + k,
                        shift_policy="full",
                    )
                else:
                    hs = hsic_from_centered(Kc, Lc, denom)
                    nhs = nhsic_from_centered(Kc, Lc, denom)
                    hs_q95 = float("nan")
                    hs_p = float("nan")
                    nhs_q95 = float("nan")
                    nhs_p = float("nan")
                agg[k]["hsic"].append(hs)
                agg[k]["hsic_null95"].append(hs_q95)
                agg[k]["hsic_p"].append(hs_p)
                agg[k]["nhsic"].append(nhs)
                agg[k]["nhsic_null95"].append(nhs_q95)
                agg[k]["nhsic_p"].append(nhs_p)

        hsic_rows = []
        nhsic_rows = []
        for k in sorted(agg.keys()):
            hsic_rows.append(
                (
                    k,
                    finite_median(agg[k]["hsic"]),
                    finite_median(agg[k]["hsic_null95"]),
                    finite_median(agg[k]["hsic_p"]),
                )
            )
            nhsic_rows.append(
                (
                    k,
                    finite_median(agg[k]["nhsic"]),
                    finite_median(agg[k]["nhsic_null95"]),
                    finite_median(agg[k]["nhsic_p"]),
                )
            )
        return hsic_rows, nhsic_rows

    iid_hsic_rows, iid_nhsic_rows = one_curve(iid, "iid")
    sticky_hsic_rows, sticky_nhsic_rows = one_curve(sticky, "sticky")

    iid_hsic_k1 = next((r for r in iid_hsic_rows if r[0] == 1), None)
    sticky_hsic_k1 = next((r for r in sticky_hsic_rows if r[0] == 1), None)
    iid_nhsic_k1 = next((r for r in iid_nhsic_rows if r[0] == 1), None)
    sticky_nhsic_k1 = next((r for r in sticky_nhsic_rows if r[0] == 1), None)
    if (
        iid_hsic_k1 is None
        or sticky_hsic_k1 is None
        or iid_nhsic_k1 is None
        or sticky_nhsic_k1 is None
    ):
        raise RuntimeError("self-test failed: missing k=1 rows")

    print(
        "[self-test] iid k=1: "
        f"hsic={iid_hsic_k1[1]:.6g}, hsic_p={iid_hsic_k1[3]:.6g}, "
        f"nhsic={iid_nhsic_k1[1]:.6g}, nhsic_p={iid_nhsic_k1[3]:.6g}"
    )
    print(
        "[self-test] sticky k=1: "
        f"hsic={sticky_hsic_k1[1]:.6g}, hsic_p={sticky_hsic_k1[3]:.6g}, "
        f"nhsic={sticky_nhsic_k1[1]:.6g}, nhsic_p={sticky_nhsic_k1[3]:.6g}"
    )

    if not (
        sticky_hsic_k1[1] > iid_hsic_k1[1]
        and sticky_hsic_k1[3] < iid_hsic_k1[3]
        and sticky_nhsic_k1[1] > iid_nhsic_k1[1]
        and sticky_nhsic_k1[3] < iid_nhsic_k1[3]
    ):
        raise RuntimeError("self-test failed: sticky dependence was not stronger than iid at k=1")

    print("[self-test] OK")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="cleaned_blocks_long")
    ap.add_argument("--outdir", default="hsic_results")
    ap.add_argument("--pattern", default="blocks_*.pkl")
    ap.add_argument("--by-pin", action="store_true", help="compute HSIC per pin sequence")
    ap.add_argument("--shuffle", action="store_true", help="shuffle blocks within each pin (negative control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=60)
    ap.add_argument("--min-n", type=int, default=80)
    ap.add_argument("--sigma-pows", default="-2,-1,0,1,2")
    ap.add_argument("--max-shifts", type=int, default=80)
    ap.add_argument("--shift-min", type=int, default=1)
    ap.add_argument("--shift-max", type=int, default=None)
    ap.add_argument("--shift-policy", choices=["band", "full"], default="band")
    ap.add_argument("--max-blocks", type=int, default=None, help="cap blocks per pin by taking the last max-blocks")
    ap.add_argument("--window-size", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=20)
    ap.add_argument("--window-mode", choices=["start", "tail", "grid", "random"], default="grid")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k-grid", default="all")
    ap.add_argument("--bonferroni", action="store_true")
    ap.add_argument("--gram-dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--summary-metric", choices=["hsic", "nhsic"], default="nhsic")
    ap.add_argument("--degenerate-mode", choices=["warn", "skip"], default="warn")
    ap.add_argument("--degenerate-mu-off-min", type=float, default=0.02)
    ap.add_argument("--degenerate-mu-off-max", type=float, default=0.98)
    ap.add_argument("--degenerate-erank-min", type=float, default=5.0)
    ap.add_argument("--degenerate-e-ratio-min", type=float, default=0.05)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.stride is None:
        args.stride = args.window_size
    if args.window_size < 3:
        raise SystemExit("--window-size must be >= 3")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.max_windows < 1:
        raise SystemExit("--max-windows must be >= 1")
    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("--alpha must be in (0, 1]")
    if args.degenerate_mu_off_min >= args.degenerate_mu_off_max:
        raise SystemExit("--degenerate-mu-off-min must be < --degenerate-mu-off-max")
    if args.self_test:
        run_self_test(args)
        return

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sigma_pows = [int(x.strip()) for x in args.sigma_pows.split(",") if x.strip()]
    gram_dtype = np.float32 if args.gram_dtype == "float32" else np.float64
    rng = np.random.default_rng(args.seed)

    pkls = sorted(indir.glob(args.pattern))
    if not pkls:
        raise SystemExit(f"No files matching {args.pattern} in {indir}")

    summary_rows = []
    curves_tsv = outdir / "hsic_curves.tsv"
    with curves_tsv.open("w", encoding="utf-8") as fcur:
        fcur.write(
            "group_id\tpin\twindow_start\tsigma\tk\tn\thsic\thsic_null95\thsic_p\tnhsic\tnhsic_null95\tnhsic_p\n"
        )

        for pkl in pkls:
            group_id, blocks, meta, _whitened = load_group_pkl(pkl)
            gdir = outdir / group_id
            gdir.mkdir(parents=True, exist_ok=True)

            seqs = split_sequences_by_pin(blocks, meta) if args.by_pin else {-1: list(zip([m["cstep"] for m in meta], blocks, meta))}

            prepared = {}
            for pin, rows in seqs.items():
                rows = list(rows)
                if args.max_blocks is not None and len(rows) > args.max_blocks:
                    rows = rows[-args.max_blocks:]

                csteps = [r[0] for r in rows]
                xblocks = [r[1] for r in rows]
                mvals = [r[2] for r in rows]

                if args.shuffle and len(xblocks) >= 2:
                    perm = rng.permutation(len(xblocks))
                    xblocks = [xblocks[i] for i in perm]
                    csteps = [csteps[i] for i in perm]
                    mvals = [mvals[i] for i in perm]

                prepared[pin] = (csteps, xblocks, mvals)

            sigma_ref = None
            sigma_ref_meta = None
            for pin, (_csteps, xblocks, _mvals) in prepared.items():
                starts, nwin, _k_eff = choose_window_starts(
                    N=len(xblocks),
                    window_size=args.window_size,
                    kmax=args.kmax,
                    stride=args.stride,
                    max_windows=args.max_windows,
                    mode=args.window_mode,
                    seed=args.seed + 17 + int(pin),
                )
                if not starts or nwin < args.min_n:
                    continue
                s0 = starts[0]
                pooled = np.vstack(xblocks[s0 : s0 + nwin])
                sigma_ref = median_sigma_vec(pooled, seed=args.seed + 1234 + int(pin))
                sigma_ref_meta = (pin, s0, nwin)
                break

            if sigma_ref is None:
                summary_rows.append(
                    [
                        group_id,
                        "nan",
                        args.window_size,
                        "nan",
                        "",
                        "nan",
                        "nan",
                        0,
                        0,
                        0,
                        "",
                    ]
                )
                continue

            sigmas = sigma_grid(sigma_ref, sigma_pows)
            n_sigmas_total = len(sigmas)

            diag_path = gdir / "sigma_diagnostics_first_window.tsv"
            allowed_sigmas = []
            deg_entries = []
            with diag_path.open("w", encoding="utf-8") as fd:
                fd.write(
                    "group_id\tpin\twindow_start\tn\tsigma0\tsigma\tmu_off\tvar_off\tq05_off\tq50_off\tq95_off\tE_ratio\terank\tdegeneracy_flags\n"
                )
                diag_pin, diag_s0, diag_n = sigma_ref_meta
                diag_blocks = prepared[diag_pin][1][diag_s0 : diag_s0 + diag_n]
                Gd_all = full_gram_multi_sigma(diag_blocks, sigmas, dtype=gram_dtype)
                for sigma in sigmas:
                    Gd = Gd_all[sigma]
                    stats = gram_offdiag_stats(Gd)
                    e_ratio, Gdc = centered_energy_ratio(Gd)
                    er = effective_rank(Gdc)
                    flags = gram_degeneracy_flags(
                        stats["mu_off"],
                        er,
                        e_ratio,
                        args.degenerate_mu_off_min,
                        args.degenerate_mu_off_max,
                        args.degenerate_erank_min,
                        args.degenerate_e_ratio_min,
                    )
                    flag_txt = ",".join(flags)
                    if flags:
                        deg_entries.append(f"{sigma:.8g}:{'|'.join(flags)}")
                    else:
                        allowed_sigmas.append(sigma)
                    fd.write(
                        f"{group_id}\t{diag_pin}\t{diag_s0}\t{diag_n}\t{sigma_ref:.8g}\t{sigma:.8g}\t"
                        f"{stats['mu_off']:.6g}\t{stats['var_off']:.6g}\t{stats['q05_off']:.6g}\t"
                        f"{stats['q50_off']:.6g}\t{stats['q95_off']:.6g}\t{e_ratio:.6g}\t{er:.6g}\t{flag_txt}\n"
                    )

            degeneracy_any = 1 if deg_entries else 0
            degeneracy_sigmas = ", ".join(deg_entries)

            if allowed_sigmas:
                active_sigmas = allowed_sigmas
            elif args.degenerate_mode == "warn":
                active_sigmas = list(sigmas)
                print(
                    f"[warn] {group_id}: all sigmas flagged as degenerate; "
                    "continuing with all sigmas because --degenerate-mode=warn"
                )
            else:
                summary_rows.append(
                    [
                        group_id,
                        "nan",
                        args.window_size,
                        "nan",
                        "",
                        "nan",
                        f"{sigma_ref:.8g}",
                        n_sigmas_total,
                        0,
                        degeneracy_any,
                        degeneracy_sigmas,
                    ]
                )
                continue

            n_sigmas_allowed = len(active_sigmas)
            sigma_used = min(active_sigmas, key=lambda s: abs(np.log(s / sigma_ref)))

            agg_by_sigma_k = defaultdict(
                lambda: {
                    "hsic": [],
                    "hsic_null95": [],
                    "hsic_p": [],
                    "nhsic": [],
                    "nhsic_null95": [],
                    "nhsic_p": [],
                }
            )
            n_windows_total = 0

            for pin, (_csteps, xblocks, _mvals) in prepared.items():
                N = len(xblocks)
                starts, nwin, k_eff = choose_window_starts(
                    N=N,
                    window_size=args.window_size,
                    kmax=args.kmax,
                    stride=args.stride,
                    max_windows=args.max_windows,
                    mode=args.window_mode,
                    seed=args.seed + 177 + int(pin),
                )
                if not starts or nwin < args.min_n:
                    continue
                tested_set = set(parse_k_grid(args.k_grid, k_eff))

                for s in starts:
                    n_windows_total += 1
                    slc = xblocks[s : s + nwin + k_eff]
                    Gs = full_gram_multi_sigma(slc, active_sigmas, dtype=gram_dtype)
                    for sigma in active_sigmas:
                        G = Gs[sigma]
                        K = G[:nwin, :nwin]
                        Kc = center_gram_hkh(K)
                        denom = float((nwin - 1) ** 2)
                        for k in range(1, k_eff + 1):
                            L = G[k : k + nwin, k : k + nwin]
                            Lc = center_gram_hkh(L)
                            hs = hsic_from_centered(Kc, Lc, denom)
                            nhs = nhsic_from_centered(Kc, Lc, denom)
                            if k in tested_set:
                                hs, hs_q95, hs_p, _ = hsic_with_null_from_centered(
                                    Kc,
                                    Lc,
                                    denom,
                                    shift_min=args.shift_min,
                                    shift_max=args.shift_max,
                                    max_shifts=args.max_shifts,
                                    seed=args.seed + 999 + s + k + int(pin),
                                    shift_policy=args.shift_policy,
                                )
                                nhs, nhs_q95, nhs_p, _ = nhsic_with_null_from_centered(
                                    Kc,
                                    Lc,
                                    denom,
                                    shift_min=args.shift_min,
                                    shift_max=args.shift_max,
                                    max_shifts=args.max_shifts,
                                    seed=args.seed + 999 + s + k + int(pin),
                                    shift_policy=args.shift_policy,
                                )
                            else:
                                hs_q95 = float("nan")
                                hs_p = float("nan")
                                nhs_q95 = float("nan")
                                nhs_p = float("nan")
                            fcur.write(
                                f"{group_id}\t{pin}\t{s}\t{sigma:.8g}\t{k}\t{nwin}\t"
                                f"{hs:.10g}\t{hs_q95:.10g}\t{hs_p:.10g}\t"
                                f"{nhs:.10g}\t{nhs_q95:.10g}\t{nhs_p:.10g}\n"
                            )
                            cell = agg_by_sigma_k[(sigma, k)]
                            cell["hsic"].append(hs)
                            cell["hsic_null95"].append(hs_q95)
                            cell["hsic_p"].append(hs_p)
                            cell["nhsic"].append(nhs)
                            cell["nhsic_null95"].append(nhs_q95)
                            cell["nhsic_p"].append(nhs_p)

            rows_by_sigma_hsic = summarize_curves(agg_by_sigma_k, active_sigmas, metric_prefix="hsic")
            rows_by_sigma_nhsic = summarize_curves(agg_by_sigma_k, active_sigmas, metric_prefix="nhsic")

            if args.summary_metric == "hsic":
                default_rows = rows_by_sigma_hsic.get(sigma_used, [])
            else:
                default_rows = rows_by_sigma_nhsic.get(sigma_used, [])
            tested_lags = sum(np.isfinite(r[3]) for r in default_rows)
            alpha_eff = args.alpha
            if args.bonferroni and tested_lags > 0:
                alpha_eff = args.alpha / tested_lags
            kstar = indep_lag_from_curve(default_rows, alpha=alpha_eff)
            dep_freq = dependence_frequency(
                default_rows,
                alpha=(alpha_eff if args.bonferroni else args.alpha),
            )

            summary_rows.append(
                [
                    group_id,
                    f"{sigma_used:.8g}",
                    args.window_size,
                    n_windows_total,
                    ("" if kstar is None else str(kstar)),
                    ("" if not np.isfinite(dep_freq) else f"{dep_freq:.6g}"),
                    f"{sigma_ref:.8g}",
                    n_sigmas_total,
                    n_sigmas_allowed,
                    degeneracy_any,
                    degeneracy_sigmas,
                ]
            )

    summ = outdir / "hsic_summary.tsv"
    with summ.open("w", encoding="utf-8") as fs:
        fs.write(
            "group_id\tsigma_used\twindow_size\tn_windows\tkstar_p05\tdependence_frequency\t"
            "sigma_ref\tn_sigmas_total\tn_sigmas_allowed\tdegeneracy_any\tdegeneracy_sigmas\n"
        )
        for r in summary_rows:
            fs.write("\t".join(map(str, r)) + "\n")

    print(
        f"[OK] wrote: {curves_tsv}, {summ}, "
        f"and per-group sigma diagnostics under {outdir}"
    )


if __name__ == "__main__":
    main()
