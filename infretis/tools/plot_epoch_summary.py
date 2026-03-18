"""Plot per-ensemble epoch_summary.tsv diagnostics.

Reads the TSV produced by the epoch controller and draws one figure per
targeted ensemble, each containing four panels:

  top-left   — n_jumps (old value entering epoch, new value set by ctrl)
  top-right  — acceptance rate
  bottom-left — average path length
  bottom-right — average lambda_max

Usage::

    python -m infretis.tools.plot_epoch_summary epoch_summary.tsv
    python -m infretis.tools.plot_epoch_summary epoch_summary.tsv --out figs/
    python -m infretis.tools.plot_epoch_summary epoch_summary.tsv --show

Figures are saved as ``epoch_summary_ens<NNN>.png`` in the output directory
(default: same directory as the TSV file).  Pass ``--show`` to also open an
interactive window.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from infretis.tools._io import read_epoch_summary_rows


def _load(tsv_path: Path) -> dict[str, dict[str, list]]:
    """Return ``{ens_name: {col: [values...]}}`` from *tsv_path*."""
    raw_rows, _ = read_epoch_summary_rows(tsv_path)
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        ens = row["ens_name"]
        data[ens]["epoch_idx"].append(int(row["epoch_idx"]))
        data[ens]["n_jumps_old"].append(int(row["n_jumps_old"]))
        data[ens]["n_jumps_new"].append(int(row["n_jumps_new"]))
        data[ens]["acc_rate"].append(float(row["acc_rate"]))
        data[ens]["avg_path_length"].append(float(row["avg_path_length"]))
        data[ens]["avg_lambda_max"].append(float(row["avg_lambda_max"]))
        data[ens]["ctrl_action"].append(row.get("ctrl_action", ""))
    return data


def _action_colour(action: str) -> str:
    return {"inc": "#2ca02c", "dec": "#d62728", "hold": "#aec7e8"}.get(
        action, "#aec7e8"
    )


def _set_epoch_xlim(ax, epochs: list[int]) -> None:
    """Give all epoch axes a clean 0.5-padded integer range."""
    lo, hi = epochs[0], epochs[-1]
    ax.set_xlim(lo - 0.5, hi + 0.5)
    if len(epochs) <= 20:
        ax.set_xticks(range(lo, hi + 1))


def _plot_ensemble(ens_name: str, d: dict[str, list], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = d["epoch_idx"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Ensemble {ens_name} — epoch summary", fontsize=13)

    # --- top-left: n_jumps ---
    ax = axes[0, 0]
    # Background band per epoch coloured by ctrl_action; clipped to data range.
    x_end = epochs[-1] + 0.5
    for i, (e, action) in enumerate(zip(epochs, d["ctrl_action"])):
        x0 = e - 0.5
        x1 = epochs[i + 1] - 0.5 if i + 1 < len(epochs) else x_end
        ax.axvspan(x0, x1, alpha=0.13, color=_action_colour(action), linewidth=0)
    ax.step(epochs, d["n_jumps_old"], where="mid", color="#1f77b4",
            linestyle="--", label="n_jumps (before epoch)", alpha=0.8)
    ax.step(epochs, d["n_jumps_new"], where="mid", color="#ff7f0e",
            linewidth=2, label="n_jumps (after epoch)")
    _set_epoch_xlim(ax, epochs)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("n_jumps")
    ax.set_title("n_jumps per epoch")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- top-right: acceptance rate ---
    ax = axes[0, 1]
    ax.plot(epochs, d["acc_rate"], marker="o", color="#9467bd",
            linewidth=1.5, markersize=4)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    _set_epoch_xlim(ax, epochs)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Acceptance rate per epoch")
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- bottom-left: avg path length ---
    ax = axes[1, 0]
    ax.plot(epochs, d["avg_path_length"], marker="s", color="#8c564b",
            linewidth=1.5, markersize=4)
    _set_epoch_xlim(ax, epochs)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Avg path length (steps)")
    ax.set_title("Average path length per epoch")
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- bottom-right: avg lambda_max ---
    ax = axes[1, 1]
    ax.plot(epochs, d["avg_lambda_max"], marker="^", color="#17becf",
            linewidth=1.5, markersize=4)
    _set_epoch_xlim(ax, epochs)
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("avg λ_max")
    ax.set_title("Average λ_max per epoch")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = out_dir / f"epoch_summary_ens{ens_name}.png"
    fig.savefig(out_path, dpi=150)
    print(f"  saved {out_path}")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-ensemble epoch_summary.tsv diagnostics."
    )
    parser.add_argument(
        "tsv", type=Path, help="Path to epoch_summary.tsv"
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory for PNG files (default: same dir as TSV).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open interactive matplotlib window after saving.",
    )
    args = parser.parse_args(argv)

    tsv_path: Path = args.tsv.resolve()
    if not tsv_path.exists():
        print(f"error: file not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir: Path = (args.out or tsv_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(tsv_path)
    if not data:
        print("error: no rows found in TSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {sum(len(v['epoch_idx']) for v in data.values())} rows "
          f"across {len(data)} ensemble(s): {', '.join(sorted(data))}.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401 — ensure backend applied

    for ens_name in sorted(data):
        _plot_ensemble(ens_name, data[ens_name], out_dir)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
