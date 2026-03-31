#!/usr/bin/env python3
"""Phase 2a benchmark analysis.

Usage:
    python benchmark/analyze_benchmark.py \
        [--results RESULTS_DIR] \
        [--out OUT_DIR] \
        [--interfaces I0 I1 ... IN]

Produces:
    pcross_by_paths.png
    pcross_by_effort.png
    pcross_by_wallclock.png
    local_pcross_by_paths.png
    controller_diagnostics.png   (SD arms only)
    max_op_convergence.png
    summary printed to stdout
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INTERFACES = [-0.99, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 1.0]
SD_ARMS = {"arm_sd_phase1", "arm_sd_empirical"}
ARM_ORDER = [
    "arm_static_n1",
    "arm_static_n2",
    "arm_static_n4",
    "arm_static_n8",
    "arm_sd_phase1",
    "arm_sd_empirical",
]
ARM_LABELS = {
    "arm_static_n1": "static n=1",
    "arm_static_n2": "static n=2",
    "arm_static_n4": "static n=4",
    "arm_static_n8": "static n=8",
    "arm_sd_phase1": "SD phase1",
    "arm_sd_empirical": "SD empirical",
}
ARM_COLORS = {
    "arm_static_n1": "#888888",
    "arm_static_n2": "#555555",
    "arm_static_n4": "#222222",
    "arm_static_n8": "#aaaaaa",
    "arm_sd_phase1": "#2196F3",
    "arm_sd_empirical": "#E91E63",
}
ARM_STYLES = {
    "arm_static_n1": "--",
    "arm_static_n2": "--",
    "arm_static_n4": "-.",
    "arm_static_n8": ":",
    "arm_sd_phase1": "-",
    "arm_sd_empirical": "-",
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_infretis_data(path):
    """Parse infretis_data.txt.

    Returns a list of dicts with keys:
        path_len (int), max_op (float), weights (list[float | None])
    Skips the 3-line header.
    """
    rows = []
    with open(path) as fh:
        # Skip header lines (lines starting with '#' or first 3 non-empty lines)
        lines = [l for l in fh if l.strip()]
    header_count = 0
    for line in lines:
        if line.startswith("#"):
            header_count += 1
            continue
        parts = line.split()
        if not parts:
            continue
        # Columns: xxx  len  max_OP  ens_000 ... ens_007
        try:
            path_len = int(parts[1])
            max_op = float(parts[2])
        except (ValueError, IndexError):
            # Still in header territory
            header_count += 1
            if header_count > 10:
                raise RuntimeError(f"Too many skipped header lines in {path}")
            continue
        weights = []
        for w in parts[3:]:
            if w == "----" or w == "---":
                weights.append(None)
            else:
                try:
                    weights.append(float(w))
                except ValueError:
                    weights.append(None)
        rows.append({"path_len": path_len, "max_op": max_op, "weights": weights})
    return rows


def _parse_timing(path):
    """Return (t_start, t_end) floats from timing.txt, or (None, None)."""
    try:
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        return float(lines[0]), float(lines[1])
    except Exception:
        return None, None


def _parse_epoch_summary(path):
    """Parse epoch_summary.tsv.  Returns list of dicts (one per row)."""
    from infretis.tools._io import read_epoch_summary_rows
    try:
        rows, _ = read_epoch_summary_rows(Path(path))
        return rows
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# Convergence computation
# ---------------------------------------------------------------------------

def _running_pcross(rows, interfaces):
    """Compute running point-match crossing probability.

    Parameters
    ----------
    rows : list[dict]  from _parse_infretis_data
    interfaces : list[float]

    Returns
    -------
    pcross_ab : np.ndarray shape (N,)  running overall P(A→B)
    local_p   : np.ndarray shape (N, n_ifaces-1)  per-interface P(λ_{i+1}|λ_i)
    cum_len   : np.ndarray shape (N,)  cumulative MD steps
    max_ops   : np.ndarray shape (N,)  raw max_op per path
    """
    n_ifaces = len(interfaces)
    n_ens = n_ifaces - 1  # ensembles 0..(n_ifaces-2); column index 0-based
    N = len(rows)
    if N == 0:
        empty = np.empty(0)
        return empty, np.empty((0, n_ens)), empty, empty

    pcross_ab = np.full(N, np.nan)
    local_p = np.full((N, n_ens), np.nan)
    cum_len = np.zeros(N)
    max_ops = np.zeros(N)

    # Running accumulators: for each ensemble j, accumulate
    #   w_sum[j]     = sum of weights so far
    #   w_cross[j]   = sum of weights where max_op > ifaces[j+1]
    w_sum = np.zeros(n_ens)
    w_cross = np.zeros(n_ens)

    for i, row in enumerate(rows):
        cum_len[i] = (cum_len[i - 1] if i > 0 else 0) + row["path_len"]
        max_ops[i] = row["max_op"]

        w = row["weights"]
        for j in range(n_ens):
            wj = w[j] if j < len(w) and w[j] is not None else 0.0
            if wj > 0.0:
                w_sum[j] += wj
                if row["max_op"] > interfaces[j + 1]:
                    w_cross[j] += wj

        # Local crossing probabilities
        p_local = np.full(n_ens, np.nan)
        for j in range(n_ens):
            if w_sum[j] > 0:
                p_local[j] = w_cross[j] / w_sum[j]
        local_p[i] = p_local

        # Overall P(A→B) = product of valid local probs (skip ensembles without data)
        valid = p_local[~np.isnan(p_local)]
        if len(valid) == n_ens and np.all(valid >= 0):
            pcross_ab[i] = float(np.prod(valid))
        # else leave as nan until all ensembles have data

    return pcross_ab, local_p, cum_len, max_ops


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(results_dir):
    """Return dict: arm_name → {seed_idx → run_dir}."""
    runs = {}
    if not os.path.isdir(results_dir):
        return runs
    for arm in sorted(os.listdir(results_dir)):
        arm_dir = os.path.join(results_dir, arm)
        if not os.path.isdir(arm_dir):
            continue
        seeds = {}
        for entry in sorted(os.listdir(arm_dir)):
            if entry.startswith("seed"):
                try:
                    sidx = int(entry[4:])
                except ValueError:
                    continue
                seed_dir = os.path.join(arm_dir, entry)
                data_file = os.path.join(seed_dir, "infretis_data.txt")
                if os.path.isfile(data_file):
                    seeds[sidx] = seed_dir
        if seeds:
            runs[arm] = seeds
    return runs


# ---------------------------------------------------------------------------
# Per-run data loading
# ---------------------------------------------------------------------------

def load_run(run_dir, interfaces):
    """Load all data for one run directory.

    Returns dict with:
        pcross_ab, local_p, cum_len, max_ops, wall_total, epoch_rows
    """
    data_path = os.path.join(run_dir, "infretis_data.txt")
    rows = _parse_infretis_data(data_path)
    pcross_ab, local_p, cum_len, max_ops = _running_pcross(rows, interfaces)

    t_start, t_end = _parse_timing(os.path.join(run_dir, "timing.txt"))
    wall_total = (t_end - t_start) if (t_start is not None and t_end is not None) else None

    epoch_rows = _parse_epoch_summary(os.path.join(run_dir, "epoch_summary.tsv"))

    return {
        "pcross_ab": pcross_ab,
        "local_p": local_p,
        "cum_len": cum_len,
        "max_ops": max_ops,
        "wall_total": wall_total,
        "epoch_rows": epoch_rows,
        "n_paths": len(rows),
    }


# ---------------------------------------------------------------------------
# Aggregation (median + IQR over seeds)
# ---------------------------------------------------------------------------

def _interp_to_grid(xs, ys, grid):
    """Interpolate (xs, ys) onto grid.  xs must be monotone non-decreasing."""
    if len(xs) == 0:
        return np.full(len(grid), np.nan)
    # Forward-fill nans in ys for interpolation
    y_filled = np.array(ys, dtype=float)
    last_valid = np.nan
    for i in range(len(y_filled)):
        if not math.isnan(y_filled[i]):
            last_valid = y_filled[i]
        elif not math.isnan(last_valid):
            y_filled[i] = last_valid
    return np.interp(grid, xs, y_filled, left=np.nan, right=y_filled[-1])


def aggregate_arm(arm_runs, interfaces, grid_size=500):
    """Aggregate seed runs for one arm.

    arm_runs : dict seed_idx → run_dict

    Returns dict with keys for paths/effort/wallclock × median/p25/p75.
    """
    seed_data = [load_run(d, interfaces) for d in arm_runs.values()]

    max_paths = max(r["n_paths"] for r in seed_data)
    max_effort = max(r["cum_len"][-1] for r in seed_data if r["n_paths"] > 0)

    path_grid = np.linspace(1, max_paths, grid_size)
    effort_grid = np.linspace(0, max_effort, grid_size)

    def _collect(key, x_key):
        """key: 'pcross_ab' or 'max_ops'; x_key: 'cum_len' or path indices."""
        arrays = []
        for r in seed_data:
            if r["n_paths"] == 0:
                continue
            if x_key == "paths":
                xs = np.arange(1, r["n_paths"] + 1, dtype=float)
            else:
                xs = r["cum_len"].astype(float)
            ys = r[key]
            if x_key == "paths":
                arr = _interp_to_grid(xs, ys, path_grid)
            else:
                arr = _interp_to_grid(xs, ys, effort_grid)
            arrays.append(arr)
        if not arrays:
            empty = np.full(grid_size, np.nan)
            return empty, empty, empty
        mat = np.vstack(arrays)
        return (
            np.nanmedian(mat, axis=0),
            np.nanpercentile(mat, 25, axis=0),
            np.nanpercentile(mat, 75, axis=0),
        )

    return {
        "path_grid": path_grid,
        "effort_grid": effort_grid,
        "pcross_by_paths": _collect("pcross_ab", "paths"),
        "pcross_by_effort": _collect("pcross_ab", "cum_len"),
        "maxop_by_paths": _collect("max_ops", "paths"),
        "seed_data": seed_data,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_band(ax, x, median, p25, p75, color, linestyle, label):
    ax.plot(x, median, color=color, linestyle=linestyle, label=label, linewidth=1.5)
    ax.fill_between(x, p25, p75, color=color, alpha=0.15)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_pcross(agg_data, x_key, xlabel, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in ARM_ORDER:
        if arm not in agg_data:
            continue
        d = agg_data[arm]
        grid = d["path_grid"] if x_key == "path_grid" else d["effort_grid"]
        key = "pcross_by_paths" if x_key == "path_grid" else "pcross_by_effort"
        med, p25, p75 = d[key]
        _plot_band(ax, grid, med, p25, p75,
                   ARM_COLORS[arm], ARM_STYLES[arm], ARM_LABELS[arm])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Running P(A→B)")
    ax.set_title("Overall crossing probability convergence")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_local_pcross(agg_data, interfaces, out_path):
    import matplotlib.pyplot as plt
    n_ifaces = len(interfaces)
    n_panels = n_ifaces - 1
    fig, axes = plt.subplots(2, math.ceil(n_panels / 2), figsize=(14, 6),
                             sharex=False, sharey=True)
    axes = axes.flatten()
    for j in range(n_panels):
        ax = axes[j]
        for arm in ARM_ORDER:
            if arm not in agg_data:
                continue
            d = agg_data[arm]
            # local_p per seed at panel j
            arrays = []
            for r in d["seed_data"]:
                if r["n_paths"] == 0:
                    continue
                xs = np.arange(1, r["n_paths"] + 1, dtype=float)
                ys = r["local_p"][:, j] if r["local_p"].shape[1] > j else np.full(r["n_paths"], np.nan)
                grid = d["path_grid"]
                arrays.append(_interp_to_grid(xs, ys, grid))
            if not arrays:
                continue
            mat = np.vstack(arrays)
            med = np.nanmedian(mat, axis=0)
            p25 = np.nanpercentile(mat, 25, axis=0)
            p75 = np.nanpercentile(mat, 75, axis=0)
            _plot_band(ax, d["path_grid"], med, p25, p75,
                       ARM_COLORS[arm], ARM_STYLES[arm],
                       ARM_LABELS[arm] if j == 0 else None)
        ax.set_title(f"P(λ={interfaces[j+1]:.2f} | λ={interfaces[j]:.2f}+)")
        ax.set_xlabel("Accepted paths")
        ax.set_ylabel("Local p_cross")
        ax.set_ylim(0, 1.05)
    if n_panels < len(axes):
        for ax in axes[n_panels:]:
            ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_controller_diagnostics(agg_data, out_path):
    """n_jumps_new and reward traces from epoch_summary.tsv for SD arms."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    ax_nj, ax_rew = axes

    for arm in SD_ARMS:
        if arm not in agg_data:
            continue
        d = agg_data[arm]
        for sidx, r in enumerate(d["seed_data"]):
            rows = r["epoch_rows"]
            if not rows:
                continue
            epochs, n_jumps_vals, rewards = [], [], []
            for row in rows:
                try:
                    ep = float(row.get("epoch_idx", row.get("epoch", "nan")))
                    nj = float(row.get("n_jumps_new", "nan"))
                    rw = float(row.get("reward_eff", row.get("reward", "nan")))
                    epochs.append(ep)
                    n_jumps_vals.append(nj)
                    rewards.append(rw)
                except (ValueError, KeyError):
                    continue
            lbl = f"{ARM_LABELS[arm]} s{sidx}"
            color = ARM_COLORS[arm]
            alpha = 0.5 + 0.25 * sidx
            ax_nj.plot(epochs, n_jumps_vals, color=color, alpha=alpha,
                       linewidth=1, label=lbl)
            ax_rew.plot(epochs, rewards, color=color, alpha=alpha,
                        linewidth=1, label=lbl)

    ax_nj.set_ylabel("n_jumps_new")
    ax_nj.set_title("n_jumps choice per epoch")
    ax_nj.legend(fontsize=7)

    ax_rew.set_ylabel("reward")
    ax_rew.set_xlabel("epoch index")
    ax_rew.set_title("Epoch reward signal")
    ax_rew.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_max_op(agg_data, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in ARM_ORDER:
        if arm not in agg_data:
            continue
        d = agg_data[arm]
        med, p25, p75 = d["maxop_by_paths"]
        _plot_band(ax, d["path_grid"], med, p25, p75,
                   ARM_COLORS[arm], ARM_STYLES[arm], ARM_LABELS[arm])
    ax.set_xlabel("Accepted paths")
    ax.set_ylabel("Running mean max_OP")
    ax.set_title("Max order parameter convergence (secondary)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(agg_data):
    header = (
        f"{'Arm':<22} {'Final P(A→B) (med)':<22} "
        f"{'at N paths':<12} {'at M MD steps':<14} {'wall-clock (s)':<14}"
    )
    print("\n" + header)
    print("-" * len(header))
    for arm in ARM_ORDER:
        if arm not in agg_data:
            continue
        d = agg_data[arm]
        med_pcross, _, _ = d["pcross_by_paths"]
        # Final value: last non-nan
        final_vals = med_pcross[~np.isnan(med_pcross)]
        final_p = float(final_vals[-1]) if len(final_vals) > 0 else float("nan")
        n_paths = d["path_grid"][-1]
        m_steps = d["effort_grid"][-1]

        wall_times = [r["wall_total"] for r in d["seed_data"]
                      if r["wall_total"] is not None]
        wall_str = (f"{np.median(wall_times):.1f}"
                    if wall_times else "n/a")

        print(
            f"{ARM_LABELS[arm]:<22} {final_p:<22.6g} "
            f"{n_paths:<12.0f} {m_steps:<14.0f} {wall_str:<14}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="benchmark/results",
                        help="Directory containing arm/seed* sub-dirs")
    parser.add_argument("--out", default="benchmark/figs",
                        help="Output directory for figures")
    parser.add_argument("--interfaces", nargs="+", type=float,
                        default=DEFAULT_INTERFACES,
                        help="Interface positions (space-separated floats)")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    os.makedirs(args.out, exist_ok=True)

    print(f"Discovering runs in: {args.results}")
    runs = discover_runs(args.results)
    if not runs:
        sys.exit(f"No run directories found under {args.results}. "
                 "Run benchmark/run_benchmark.sh first.")

    print(f"Found arms: {sorted(runs)}")
    print(f"Interfaces: {args.interfaces}")

    print("\nLoading and aggregating data...")
    agg = {}
    for arm, arm_runs in runs.items():
        print(f"  {arm}: {len(arm_runs)} seeds")
        agg[arm] = aggregate_arm(arm_runs, args.interfaces)

    print("\nGenerating plots...")
    plot_pcross(agg, "path_grid", "Accepted paths",
                os.path.join(args.out, "pcross_by_paths.png"))
    plot_pcross(agg, "effort_grid", "Cumulative MD steps",
                os.path.join(args.out, "pcross_by_effort.png"))
    plot_local_pcross(agg, args.interfaces,
                      os.path.join(args.out, "local_pcross_by_paths.png"))
    plot_controller_diagnostics(agg,
                                os.path.join(args.out, "controller_diagnostics.png"))
    plot_max_op(agg,
                os.path.join(args.out, "max_op_convergence.png"))

    print_summary(agg)
    print(f"\nFigures written to: {args.out}")


if __name__ == "__main__":
    main()
