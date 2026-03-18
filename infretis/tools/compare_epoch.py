"""Compare epoch_summary.tsv files across multiple seeds/arms."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated as Atd, List, Optional

import typer
from typer import Argument as Arg, Option as Opt

from infretis.tools._io import read_epoch_summary_rows, safe_float, safe_int, write_tsv

# Columns that _load() converts to numeric for plotting.
# Everything else in present_optional is stored as a raw string.
_INT_COLS = {"n_jumps_old", "n_attempted", "n_accepted"}
_FLOAT_COLS = {"avg_lambda_max", "reward_eff", "avg_subcycles",
               "avg_subpath_length", "lp_over_ls_target_value"}


def _load(tsv_path: Path):
    """Return (data, present_optional) from tsv_path, schema-tolerant."""
    raw_rows, present_optional = read_epoch_summary_rows(tsv_path)

    data = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        ens = row["ens_name"]
        # Required columns — always present after validation.
        data[ens]["epoch_idx"].append(safe_int(row["epoch_idx"], default=0))
        data[ens]["n_jumps_new"].append(safe_int(row["n_jumps_new"], default=0))
        data[ens]["acc_rate"].append(safe_float(row["acc_rate"]))
        data[ens]["avg_path_length"].append(safe_float(row["avg_path_length"]))

        for col in present_optional:
            val = row.get(col, "")
            if col in _INT_COLS:
                parsed = safe_int(val)
                data[ens][col].append(parsed if parsed is not None else float("nan"))
            elif col in _FLOAT_COLS:
                data[ens][col].append(safe_float(val))
            else:
                data[ens][col].append(val)

    return data, present_optional


def _write_summary_tsv(path: Path, all_ens: list, all_data: list, labels: list, col: str) -> None:
    import numpy as np

    rows = []
    for ens in sorted(all_ens):
        row = [ens]
        for data in all_data:
            vals = data.get(ens, {}).get(col, [])
            row.append(f"{float(np.nanmean(vals)):.6g}" if vals else "")
        rows.append(row)
    write_tsv(path, ["ens_name"] + labels, rows)


def compare_epoch(
    tsvs: Atd[List[str], Arg(help="epoch_summary.tsv files to compare")],
    labels: Atd[Optional[List[str]], Opt("--labels", help="Labels for each TSV (default: filename stems)")] = None,
    outdir: Atd[str, Opt("--outdir", help="Output directory")] = "epoch_compare",
    show: Atd[bool, Opt("--show", help="Show interactive matplotlib window")] = False,
) -> None:
    """Compare epoch_summary.tsv files across multiple seeds/arms."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not tsvs:
        print("No TSV files provided.")
        raise typer.Exit(code=1)

    tsv_paths = [Path(t) for t in tsvs]
    for p in tsv_paths:
        if not p.exists():
            print(f"File not found: {p}")
            raise typer.Exit(code=1)

    if labels is None:
        labels = [p.stem for p in tsv_paths]
    elif len(labels) != len(tsv_paths):
        print(f"Number of labels ({len(labels)}) must match number of TSV files ({len(tsv_paths)}).")
        raise typer.Exit(code=1)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    all_data = []
    all_optional: list[set] = []
    for p in tsv_paths:
        data, present_opt = _load(p)
        all_data.append(data)
        all_optional.append(present_opt)

    all_ens: set[str] = set()
    for data in all_data:
        all_ens.update(data.keys())
    sorted_ens = sorted(all_ens)

    has_reward = any("reward_eff" in opt for opt in all_optional)
    has_lambda = any("avg_lambda_max" in opt for opt in all_optional)

    def _make_fig(metric: str, ylabel: str, title_prefix: str) -> dict[str, plt.Figure]:
        figs = {}
        for ens in sorted_ens:
            fig, ax = plt.subplots(figsize=(8, 4))
            for data, label in zip(all_data, labels):
                ens_data = data.get(ens, {})
                epochs = ens_data.get("epoch_idx", [])
                vals = ens_data.get(metric, [])
                if epochs and vals:
                    ax.plot(epochs, vals, marker="o", markersize=3, label=label)
            ax.set_xlabel("Epoch index")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix} — ensemble {ens}")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            figs[ens] = fig
        return figs

    def _save_figs(figs: dict, filename_prefix: str) -> None:
        for ens, fig in figs.items():
            safe_ens = ens.replace("/", "_")
            fig.savefig(outdir_path / f"{filename_prefix}_ens{safe_ens}.png", dpi=150)
            plt.close(fig)

    # Always-written plots
    _save_figs(_make_fig("n_jumps_new", "n_jumps", "n_jumps traces"), "n_jumps_traces")
    _save_figs(_make_fig("acc_rate", "Acceptance rate", "Acceptance rate traces"), "acceptance_traces")
    _save_figs(_make_fig("avg_path_length", "Avg path length", "Path length traces"), "path_length_traces")

    # Conditional plots
    if has_reward:
        _save_figs(_make_fig("reward_eff", "Reward (eff)", "Reward traces"), "reward_traces")
    if has_lambda:
        _save_figs(_make_fig("avg_lambda_max", "avg λ_max", "λ_max traces"), "lambda_max_traces")

    # Summary TSVs
    _write_summary_tsv(
        outdir_path / "acceptance_summary.tsv",
        sorted_ens, all_data, labels, "acc_rate",
    )
    _write_summary_tsv(
        outdir_path / "path_length_summary.tsv",
        sorted_ens, all_data, labels, "avg_path_length",
    )
    if has_lambda:
        _write_summary_tsv(
            outdir_path / "lambda_max_summary.tsv",
            sorted_ens, all_data, labels, "avg_lambda_max",
        )

    if show:
        plt.show()

    print(f"[OK] wrote comparison outputs to {outdir_path}")
