"""Clean and structure infRETIS worker TSV logs into block sequences.

A "block" corresponds to one MC attempt at a given `(pin, ens_name, cstep,
parent_move)` and contains the one-or-more shooting-point vectors generated
for that attempt (WF/MWF often yields multiple rows per cstep).

This script only prepares clean inputs and summaries for downstream analysis
(e.g., RKHS/HSIC decorrelation, Mahalanobis kernels, whitening). It does not
run HSIC itself.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


MOVE_HEADER = [
    "cstep",
    "pin",
    "ens_name",
    "move",
    "path_n",
    "accepted",
    "status",
    "n_jumps",
]
SHOOT_HEADER = [
    "cstep",
    "pin",
    "ens_name",
    "parent_move",
    "path_n",
    "idx",
    "dek",
    "op_before",
    "op_after",
]


@dataclass
class MoveRow:
    cstep: int
    pin: int
    ens_name: str
    move: str
    path_n: Optional[int]
    accepted: Optional[int]
    status: Optional[str]
    n_jumps: Optional[int]
    source: str
    line_no: int


@dataclass
class ShootRow:
    cstep: int
    pin: int
    ens_name: str
    parent_move: str
    path_n: Optional[int]
    idx: int
    dek: float
    op_before: np.ndarray
    op_after: np.ndarray
    source: str
    line_no: int
    row_order: int


@dataclass
class Block:
    key: tuple[int, str, int, str]  # (pin, ens_name, cstep, parent_move)
    X: np.ndarray
    meta: dict[str, Any]
    raw_n_points: int
    missing_move_rows: int


class AnomalyLog:
    """Collect counts and a few examples for human-readable anomaly output."""

    def __init__(self, max_examples: int = 10):
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = defaultdict(list)
        self.max_examples = max_examples

    def add(self, kind: str, detail: str) -> None:
        self.counts[kind] += 1
        if len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append(detail)

    def write(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            fh.write("# anomalies.log\n")
            fh.write("# counts and first examples per category\n\n")
            for kind in sorted(self.counts):
                fh.write(f"[{kind}] count={self.counts[kind]}\n")
                for ex in self.examples.get(kind, []):
                    fh.write(f"  - {ex}\n")
                fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean/structure infRETIS worker TSV logs into block sequences"
    )
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument(
        "--outdir", default="cleaned_blocks", help="Output directory"
    )
    parser.add_argument(
        "--move-blocks",
        action="append",
        nargs="+",
        help="Explicit move_blocks.tsv path(s) (may be repeated)",
    )
    parser.add_argument(
        "--shoot-points",
        action="append",
        nargs="+",
        help="Explicit shoot_points_all.tsv path(s) (may be repeated)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_mode",
        choices=("accepted", "all", "mc"),
        default="accepted",
        help="Block filtering mode",
    )
    parser.add_argument(
        "--use",
        dest="use_mode",
        choices=("op_before", "op_after", "both"),
        default="op_before",
        help="Feature source per shoot row",
    )
    parser.add_argument(
        "--group-by-move",
        action="store_true",
        help="Group outputs by (ens_name, parent_move) instead of ens_name",
    )
    parser.add_argument(
        "--compute-cov",
        action="store_true",
        help="Compute per-group mean/cov/cov_reg (no HSIC)",
    )
    parser.add_argument(
        "--whiten-output",
        action="store_true",
        help="Whiten blocks using Cholesky(cov_reg)^-1 (implies --compute-cov)",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="Cov reg epsilon")
    parser.add_argument(
        "--shrink",
        type=float,
        default=0.05,
        help="Diagonal shrinkage coefficient in [0,1]",
    )
    parser.add_argument(
        "--min-blocks",
        type=int,
        default=20,
        help="Flag groups with fewer kept blocks than this threshold",
    )
    return parser.parse_args()


def flatten_paths(arg: Optional[list[list[str]]]) -> list[Path]:
    out: list[Path] = []
    if not arg:
        return out
    seen: set[Path] = set()
    for group in arg:
        for raw in group:
            p = Path(raw)
            if p not in seen:
                out.append(p)
                seen.add(p)
    return out


def discover_inputs(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    explicit_moves = flatten_paths(args.move_blocks)
    explicit_shoots = flatten_paths(args.shoot_points)

    if explicit_moves or explicit_shoots:
        return explicit_moves, explicit_shoots

    root = Path(args.root)
    move_paths = sorted(root.glob("worker*/move_blocks.tsv"))
    shoot_paths = sorted(root.glob("worker*/shoot_points_all.tsv"))
    return move_paths, shoot_paths


def parse_int_opt(text: str) -> Optional[int]:
    txt = text.strip()
    if txt in ("", "NA", "None", "nan"):
        return None
    return int(txt)


def parse_float(text: str) -> float:
    return float(text.strip())


def parse_vector(text: str) -> np.ndarray:
    vals = [v for v in text.strip().split(",") if v != ""]
    if not vals:
        raise ValueError("empty vector")
    arr = np.asarray([float(v) for v in vals], dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _candidate_ens_parent_from_fields(
    fields: list[str], kind: str
) -> tuple[Optional[str], Optional[str]]:
    try:
        if kind == "shoot" and len(fields) >= 4:
            return fields[2], fields[3]
        if kind == "move" and len(fields) >= 3:
            return fields[2], None
    except Exception:
        pass
    return None, None


def bump_group_metric(
    group_metrics: dict[tuple[str, str], Counter[str]],
    ens_name: Optional[str],
    parent_move: Optional[str],
    metric: str,
    n: int = 1,
) -> None:
    if ens_name is None:
        return
    pm = parent_move if parent_move is not None else "__ANY_MOVE__"
    group_metrics[(ens_name, pm)][metric] += n


def read_move_blocks(
    paths: Iterable[Path],
    anomalies: AnomalyLog,
    group_metrics: dict[tuple[str, str], Counter[str]],
) -> list[MoveRow]:
    rows: list[MoveRow] = []
    seen_exact: set[tuple[Any, ...]] = set()
    for path in paths:
        if not path.is_file():
            anomalies.add("missing_move_file", str(path))
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                fields = line.split("\t")
                if fields and fields[0] == "cstep":
                    if line_no > 1:
                        anomalies.add(
                            "repeated_header_move",
                            f"{path}:{line_no} {line[:120]}",
                        )
                    continue
                if len(fields) not in (7, 8):
                    ens_name, _ = _candidate_ens_parent_from_fields(fields, "move")
                    bump_group_metric(
                        group_metrics,
                        ens_name,
                        None,
                        "move_malformed_rows_dropped",
                        1,
                    )
                    anomalies.add(
                        "malformed_move_row",
                        f"{path}:{line_no} fields={len(fields)} line={line[:200]}",
                    )
                    continue
                try:
                    # n_jumps column was added later; absent in old log files.
                    n_jumps_raw = fields[7] if len(fields) >= 8 else None
                    n_jumps: Optional[int] = None
                    if n_jumps_raw is not None and n_jumps_raw not in (
                        "",
                        "NA",
                        "None",
                        "nan",
                    ):
                        n_jumps = int(n_jumps_raw)
                    row = MoveRow(
                        cstep=int(fields[0]),
                        pin=int(fields[1]),
                        ens_name=fields[2],
                        move=fields[3],
                        path_n=parse_int_opt(fields[4]),
                        accepted=parse_int_opt(fields[5]),
                        status=fields[6] if fields[6] not in ("", "NA") else None,
                        n_jumps=n_jumps,
                        source=str(path),
                        line_no=line_no,
                    )
                except Exception as exc:
                    ens_name, _ = _candidate_ens_parent_from_fields(fields, "move")
                    bump_group_metric(
                        group_metrics,
                        ens_name,
                        None,
                        "move_malformed_rows_dropped",
                        1,
                    )
                    anomalies.add(
                        "move_parse_failure",
                        f"{path}:{line_no} {exc} line={line[:200]}",
                    )
                    continue

                sig = (
                    row.cstep,
                    row.pin,
                    row.ens_name,
                    row.move,
                    row.path_n,
                    row.accepted,
                    row.status,
                    row.n_jumps,
                )
                if sig in seen_exact:
                    anomalies.add(
                        "duplicate_move_row",
                        f"{path}:{line_no} key={(row.pin,row.cstep,row.ens_name)}",
                    )
                    continue
                seen_exact.add(sig)
                rows.append(row)
    return rows


def read_shoot_points(
    paths: Iterable[Path],
    anomalies: AnomalyLog,
    group_metrics: dict[tuple[str, str], Counter[str]],
) -> list[ShootRow]:
    rows: list[ShootRow] = []
    seen_exact: set[tuple[Any, ...]] = set()
    row_order = 0
    for path in paths:
        if not path.is_file():
            anomalies.add("missing_shoot_file", str(path))
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                fields = line.split("\t")
                if fields and fields[0] == "cstep":
                    if line_no > 1:
                        anomalies.add(
                            "repeated_header_shoot",
                            f"{path}:{line_no} {line[:120]}",
                        )
                    continue
                if len(fields) != 9:
                    ens_name, parent_move = _candidate_ens_parent_from_fields(
                        fields, "shoot"
                    )
                    bump_group_metric(
                        group_metrics,
                        ens_name,
                        parent_move,
                        "malformed_rows_dropped",
                        1,
                    )
                    anomalies.add(
                        "malformed_shoot_row",
                        f"{path}:{line_no} fields={len(fields)} line={line[:200]}",
                    )
                    continue

                ens_name = fields[2]
                parent_move = fields[3]
                try:
                    op_before = parse_vector(fields[7])
                    op_after = parse_vector(fields[8])
                    if op_before.shape[0] != op_after.shape[0]:
                        raise ValueError(
                            f"op_before dim {op_before.shape[0]} != "
                            f"op_after dim {op_after.shape[0]}"
                        )

                    row = ShootRow(
                        cstep=int(fields[0]),
                        pin=int(fields[1]),
                        ens_name=ens_name,
                        parent_move=parent_move,
                        path_n=parse_int_opt(fields[4]),
                        idx=int(fields[5]),
                        dek=parse_float(fields[6]),
                        op_before=op_before,
                        op_after=op_after,
                        source=str(path),
                        line_no=line_no,
                        row_order=row_order,
                    )
                except Exception as exc:
                    bump_group_metric(
                        group_metrics,
                        ens_name,
                        parent_move,
                        "malformed_rows_dropped",
                        1,
                    )
                    anomalies.add(
                        "shoot_parse_failure",
                        f"{path}:{line_no} {exc} line={line[:200]}",
                    )
                    continue

                sig = (
                    row.cstep,
                    row.pin,
                    row.ens_name,
                    row.parent_move,
                    row.path_n,
                    row.idx,
                    float(row.dek),
                    tuple(np.asarray(row.op_before, float).tolist()),
                    tuple(np.asarray(row.op_after, float).tolist()),
                )
                if sig in seen_exact:
                    bump_group_metric(
                        group_metrics,
                        ens_name,
                        parent_move,
                        "shoot_duplicates_removed",
                        1,
                    )
                    anomalies.add(
                        "duplicate_shoot_row",
                        f"{path}:{line_no} key={(row.pin,row.ens_name,row.cstep,row.parent_move)} idx={row.idx}",
                    )
                    continue
                seen_exact.add(sig)
                rows.append(row)
                row_order += 1
    return rows


def build_move_lookup(
    move_rows: list[MoveRow], anomalies: AnomalyLog
) -> tuple[
    dict[tuple[int, int, str], MoveRow],
    dict[tuple[str, str], tuple[int, float]],
    dict[str, tuple[int, float]],
]:
    lookup: dict[tuple[int, int, str], MoveRow] = {}
    acceptance_counts_move: dict[tuple[str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    acceptance_counts_ens: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for row in move_rows:
        key = (row.pin, row.cstep, row.ens_name)
        if key in lookup:
            prev = lookup[key]
            prev_sig = (prev.move, prev.path_n, prev.accepted, prev.status)
            new_sig = (row.move, row.path_n, row.accepted, row.status)
            if prev_sig != new_sig:
                anomalies.add(
                    "conflicting_move_rows",
                    f"key={key} prev={prev_sig} new={new_sig}",
                )
            continue
        lookup[key] = row

        # acceptance counts for summary, grouped by ens and by (ens, move)
        if row.accepted in (0, 1):
            acceptance_counts_move[(row.ens_name, row.move)][0] += row.accepted
            acceptance_counts_move[(row.ens_name, row.move)][1] += 1
            if row.move != "swap":
                acceptance_counts_ens[row.ens_name][0] += row.accepted
                acceptance_counts_ens[row.ens_name][1] += 1

    accept_stats = {
        key: (counts[1], counts[0] / counts[1])  # (n, rate)
        for key, counts in acceptance_counts_move.items()
        if counts[1] > 0
    }
    accept_stats_ens = {
        key: (counts[1], counts[0] / counts[1])  # (n, rate), excluding swaps
        for key, counts in acceptance_counts_ens.items()
        if counts[1] > 0
    }
    return lookup, accept_stats, accept_stats_ens


def _feature_rows_for_shoot(row: ShootRow, use_mode: str) -> list[np.ndarray]:
    if use_mode == "op_before":
        return [row.op_before]
    if use_mode == "op_after":
        return [row.op_after]
    if use_mode == "both":
        return [row.op_before, row.op_after]
    raise ValueError(f"Unknown --use mode: {use_mode}")


def build_blocks(
    shoot_rows: list[ShootRow],
    move_lookup: dict[tuple[int, int, str], MoveRow],
    use_mode: str,
    anomalies: AnomalyLog,
    group_metrics: dict[tuple[str, str], Counter[str]],
) -> list[Block]:
    buckets: dict[tuple[int, str, int, str], list[ShootRow]] = defaultdict(list)
    for row in shoot_rows:
        buckets[(row.pin, row.ens_name, row.cstep, row.parent_move)].append(row)

    blocks: list[Block] = []
    for key in sorted(buckets.keys(), key=lambda k: (k[0], k[2], k[1], k[3])):
        pin, ens_name, cstep, parent_move = key
        rows = sorted(buckets[key], key=lambda r: (r.idx, r.row_order))
        move_row = move_lookup.get((pin, cstep, ens_name))

        if move_row is None:
            anomalies.add(
                "missing_move_join",
                f"key={(pin, cstep, ens_name)} parent_move={parent_move}",
            )
            bump_group_metric(
                group_metrics, ens_name, parent_move, "missing_move_rows", len(rows)
            )

        # Determine consistent vector dimension for the selected feature mode.
        kept_rows: list[ShootRow] = []
        block_dim: Optional[int] = None
        for row in rows:
            feats = _feature_rows_for_shoot(row, use_mode)
            row_dims = {int(np.asarray(f).shape[0]) for f in feats}
            if len(row_dims) != 1:
                anomalies.add(
                    "row_feature_dim_mismatch",
                    f"{row.source}:{row.line_no} dims={sorted(row_dims)} key={key}",
                )
                bump_group_metric(
                    group_metrics, ens_name, parent_move, "malformed_rows_dropped", 1
                )
                continue
            row_dim = next(iter(row_dims))
            if block_dim is None:
                block_dim = row_dim
            elif row_dim != block_dim:
                anomalies.add(
                    "block_dim_mismatch",
                    f"{row.source}:{row.line_no} block_dim={block_dim} row_dim={row_dim} key={key}",
                )
                bump_group_metric(
                    group_metrics, ens_name, parent_move, "malformed_rows_dropped", 1
                )
                continue
            kept_rows.append(row)

        if not kept_rows:
            anomalies.add("empty_block_after_cleaning", f"key={key}")
            continue

        x_rows: list[np.ndarray] = []
        for row in kept_rows:
            x_rows.extend(_feature_rows_for_shoot(row, use_mode))
        X = np.asarray(x_rows, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        path_ns = [r.path_n for r in kept_rows if r.path_n is not None]
        path_n = path_ns[0] if path_ns else None
        if any(p != path_n for p in path_ns[1:]):
            anomalies.add(
                "block_pathn_inconsistent",
                f"key={key} path_ns={path_ns[:10]}",
            )

        meta: dict[str, Any] = {
            "pin": pin,
            "cstep": cstep,
            "ens_name": ens_name,
            "parent_move": parent_move,
            "move": move_row.move if move_row else None,
            "path_n": path_n,
            "accepted": move_row.accepted if move_row else None,
            "status": move_row.status if move_row else None,
            "n_jumps": move_row.n_jumps if move_row else None,
            "n_shoot_rows": len(kept_rows),
            "n_points": int(X.shape[0]),
            "raw_n_shoot_rows": len(rows),
        }
        if move_row and path_n is None and move_row.path_n is not None:
            meta["path_n"] = move_row.path_n

        blocks.append(
            Block(
                key=key,
                X=X,
                meta=meta,
                raw_n_points=len(rows),
                missing_move_rows=len(kept_rows) if move_row is None else 0,
            )
        )
    return blocks


def clone_block_with(
    block: Block, X: np.ndarray, meta_updates: Optional[dict[str, Any]] = None
) -> Block:
    meta = dict(block.meta)
    if meta_updates:
        meta.update(meta_updates)
    return Block(
        key=block.key,
        X=np.asarray(X, dtype=float).copy(),
        meta=meta,
        raw_n_points=block.raw_n_points,
        missing_move_rows=block.missing_move_rows,
    )


def apply_filter_mode(
    blocks: list[Block], filter_mode: str, anomalies: AnomalyLog
) -> tuple[list[Block], Counter[str]]:
    if filter_mode in ("accepted", "all"):
        kept: list[Block] = []
        stats = Counter[str]()
        for block in blocks:
            if filter_mode == "all":
                kept.append(block)
                continue
            accepted = block.meta.get("accepted")
            status = block.meta.get("status")
            if accepted == 1 and status == "ACC":
                kept.append(block)
        return kept, stats

    # Markov-chain self-loop mode, per (pin, ens_name, parent_move)
    seqs: dict[tuple[int, str, str], list[Block]] = defaultdict(list)
    for block in blocks:
        seq_key = (
            int(block.meta["pin"]),
            str(block.meta["ens_name"]),
            str(block.meta["parent_move"]),
        )
        seqs[seq_key].append(block)

    kept = []
    stats = Counter[str]()
    for seq_key, seq_blocks in seqs.items():
        seq_sorted = sorted(seq_blocks, key=lambda b: (int(b.meta["cstep"]),))
        prev_accepted: Optional[Block] = None
        for block in seq_sorted:
            accepted = block.meta.get("accepted")
            status = block.meta.get("status")
            if accepted == 0:
                if prev_accepted is None:
                    stats["mc_reject_no_prev_kept"] += 1
                    kept.append(clone_block_with(block, block.X))
                else:
                    stats["mc_self_loops"] += 1
                    replay_meta = {
                        "n_points": int(prev_accepted.meta["n_points"]),
                        "n_shoot_rows": int(prev_accepted.meta["n_shoot_rows"]),
                        "path_n": prev_accepted.meta.get("path_n"),
                        "mc_self_loop": 1,
                        "mc_source_cstep": int(prev_accepted.meta["cstep"]),
                    }
                    kept.append(clone_block_with(block, prev_accepted.X, replay_meta))
                continue

            if accepted is None:
                stats["mc_missing_accepted_kept"] += 1
                block2 = clone_block_with(block, block.X, {"mc_self_loop": 0})
                kept.append(block2)
                continue

            block2 = clone_block_with(block, block.X, {"mc_self_loop": 0})
            kept.append(block2)
            if accepted == 1 and status == "ACC":
                prev_accepted = block
            elif accepted not in (0, 1):
                anomalies.add(
                    "unexpected_accepted_value",
                    f"seq={seq_key} cstep={block.meta['cstep']} accepted={accepted}",
                )

    kept.sort(key=lambda b: (int(b.meta["pin"]), int(b.meta["cstep"]), b.meta["ens_name"], b.meta["parent_move"]))
    return kept, stats


def final_group_key(block: Block, group_by_move: bool) -> tuple[str, ...]:
    if group_by_move:
        return (str(block.meta["ens_name"]), str(block.meta["parent_move"]))
    return (str(block.meta["ens_name"]),)


def sanitize_component(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    safe = safe.strip("._")
    return safe or "group"


def group_id_from_key(group_key: tuple[str, ...]) -> str:
    return "__".join(sanitize_component(part) for part in group_key)


def compute_covariance_products(
    X: np.ndarray,
    eps: float,
    shrink: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu = np.mean(X, axis=0)
    d = X.shape[1]
    if X.shape[0] <= 1:
        cov = np.zeros((d, d), dtype=float)
    else:
        cov = np.asarray(np.cov(X, rowvar=False, bias=False), dtype=float)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
    if shrink:
        cov = (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))
    cov_reg = cov + float(eps) * np.eye(d, dtype=float)
    return mu, cov, cov_reg


def whiten_block_rows(X: np.ndarray, mu: np.ndarray, cov_reg: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    centered = X - mu
    try:
        L = np.linalg.cholesky(cov_reg)
        return np.linalg.solve(L, centered.T).T
    except np.linalg.LinAlgError:
        # Fallback for numerical edge cases; still deterministic and safe.
        evals, evecs = np.linalg.eigh(cov_reg)
        evals = np.maximum(evals, 1e-15)
        invsqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        return centered @ invsqrt.T


def normalize_group_dims(
    blocks: list[np.ndarray], meta: list[dict[str, Any]]
) -> tuple[list[np.ndarray], list[dict[str, Any]], Optional[int], list[dict[str, Any]]]:
    """Drop blocks not matching the modal vector dimension within a group."""
    if not blocks:
        return [], [], None, []

    dims = [int(np.asarray(b).shape[1]) for b in blocks]
    counts = Counter(dims)
    max_count = max(counts.values())
    # Deterministic tie break: first occurring dim among the most frequent dims.
    target_dim: Optional[int] = next(dim for dim in dims if counts[dim] == max_count)

    if all(dim == target_dim for dim in dims):
        return blocks, meta, target_dim, []

    blocks2: list[np.ndarray] = []
    meta2: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for idx, (X, m) in enumerate(zip(blocks, meta)):
        dim = int(np.asarray(X).shape[1])
        if dim == target_dim:
            blocks2.append(X)
            meta2.append(m)
            continue
        dropped.append(
            {
                "index": idx,
                "dim": dim,
                "meta": dict(m),
            }
        )
    return blocks2, meta2, target_dim, dropped


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


from infretis.tools._io import write_tsv  # noqa: E402


def main() -> int:
    args = parse_args()
    if args.whiten_output:
        args.compute_cov = True
    if args.eps < 0:
        raise SystemExit("--eps must be >= 0")
    if not (0.0 <= args.shrink <= 1.0):
        raise SystemExit("--shrink must be in [0, 1]")

    anomalies = AnomalyLog()
    # keyed by (ens_name, parent_move or '__ANY_MOVE__')
    group_metrics: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    move_paths, shoot_paths = discover_inputs(args)
    if not move_paths and not shoot_paths:
        print("No input TSVs found.")
        return 1
    if not shoot_paths:
        print("No shoot_points_all.tsv inputs found.")
        return 1

    move_rows = read_move_blocks(move_paths, anomalies, group_metrics)
    shoot_rows = read_shoot_points(shoot_paths, anomalies, group_metrics)
    move_lookup, accept_stats, accept_stats_ens = build_move_lookup(
        move_rows, anomalies
    )

    raw_blocks = build_blocks(
        shoot_rows=shoot_rows,
        move_lookup=move_lookup,
        use_mode=args.use_mode,
        anomalies=anomalies,
        group_metrics=group_metrics,
    )
    kept_blocks, mc_stats = apply_filter_mode(raw_blocks, args.filter_mode, anomalies)

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # Group raw/kept blocks for summaries and outputs.
    raw_groups: dict[tuple[str, ...], list[Block]] = defaultdict(list)
    for block in raw_blocks:
        raw_groups[final_group_key(block, args.group_by_move)].append(block)

    kept_groups: dict[tuple[str, ...], list[Block]] = defaultdict(list)
    for block in kept_blocks:
        kept_groups[final_group_key(block, args.group_by_move)].append(block)
    for group_key in kept_groups:
        kept_groups[group_key].sort(
            key=lambda b: (int(b.meta["pin"]), int(b.meta["cstep"]))
        )

    all_group_keys = sorted(set(raw_groups) | set(kept_groups))
    blocks_index_rows: list[list[Any]] = []
    summary_rows: list[list[Any]] = []

    for group_key in all_group_keys:
        ens_name = group_key[0]
        parent_move_group = group_key[1] if len(group_key) > 1 else None
        group_id = group_id_from_key(group_key)
        output_file = f"blocks_{group_id}.pkl"
        group_raw = raw_groups.get(group_key, [])
        group_kept_pre = kept_groups.get(group_key, [])

        # Summary stats (raw and kept)
        n_blocks_raw = len(group_raw)
        n_shoot_rows_raw = sum(b.raw_n_points for b in group_raw)
        missing_move_rows = sum(b.missing_move_rows for b in group_raw)

        if args.group_by_move:
            shoot_dup_removed = group_metrics[(ens_name, parent_move_group or "")].get(
                "shoot_duplicates_removed", 0
            )
            shoot_malformed_dropped = group_metrics[
                (ens_name, parent_move_group or "")
            ].get(
                "malformed_rows_dropped", 0
            )
        else:
            shoot_dup_removed = 0
            shoot_malformed_dropped = 0
            for (ens_k, pm_k), ctr in group_metrics.items():
                if ens_k != ens_name:
                    continue
                if pm_k == "__ANY_MOVE__":
                    continue
                shoot_dup_removed += ctr.get("shoot_duplicates_removed", 0)
                shoot_malformed_dropped += ctr.get("malformed_rows_dropped", 0)

        acceptance_rate = None
        acceptance_rate_n: Optional[int] = None
        acceptance_rate_fallback = 0
        if not args.group_by_move:
            if ens_name in accept_stats_ens:
                acceptance_rate_n, acceptance_rate = accept_stats_ens[ens_name]
        else:
            parent_move_key = parent_move_group or ""
            if (ens_name, parent_move_key) in accept_stats:
                acceptance_rate_n, acceptance_rate = accept_stats[
                    (ens_name, parent_move_key)
                ]
            elif ens_name in accept_stats_ens:
                acceptance_rate_n, acceptance_rate = accept_stats_ens[ens_name]
                acceptance_rate_fallback = 1

        # Optional covariance and optional whitening use filtered blocks after
        # group-wise dimension normalization.
        blocks_for_output: list[np.ndarray] = [b.X.copy() for b in group_kept_pre]
        meta_for_output: list[dict[str, Any]] = [dict(b.meta) for b in group_kept_pre]
        (
            blocks_for_output,
            meta_for_output,
            dim_mode,
            dim_dropped,
        ) = normalize_group_dims(blocks_for_output, meta_for_output)
        n_blocks_dim_dropped = len(dim_dropped)
        for item in dim_dropped:
            m = item["meta"]
            anomalies.add(
                "group_dim_dropped_block",
                f"group={group_id} cstep={m.get('cstep')} pin={m.get('pin')} dim={item['dim']} target_dim={dim_mode}",
            )

        n_blocks_kept = len(blocks_for_output)
        n_shoot_rows_kept = sum(int(m["n_shoot_rows"]) for m in meta_for_output)

        if blocks_for_output:
            vector_dim = int(np.asarray(blocks_for_output[0]).shape[1])
            block_sizes = np.asarray([m["n_points"] for m in meta_for_output], dtype=float)
            median_block_size = float(np.median(block_sizes))
            mean_block_size = float(np.mean(block_sizes))
            csteps = [int(m["cstep"]) for m in meta_for_output]
            cstep_min = int(min(csteps))
            cstep_max = int(max(csteps))
        elif group_raw:
            vector_dim = int(group_raw[0].X.shape[1])
            block_sizes = np.asarray([b.meta["n_points"] for b in group_raw], dtype=float)
            median_block_size = float(np.median(block_sizes))
            mean_block_size = float(np.mean(block_sizes))
            csteps = [int(b.meta["cstep"]) for b in group_raw]
            cstep_min = int(min(csteps))
            cstep_max = int(max(csteps))
        else:
            vector_dim = 0
            median_block_size = float("nan")
            mean_block_size = float("nan")
            cstep_min = None
            cstep_max = None

        mu = cov = cov_reg = None
        if args.compute_cov and blocks_for_output:
            pooled = np.vstack(blocks_for_output)
            mu, cov, cov_reg = compute_covariance_products(
                pooled, eps=args.eps, shrink=args.shrink
            )
            cov_dir = outdir / f"cov_{group_id}"
            ensure_outdir(cov_dir)
            np.save(cov_dir / "mu.npy", mu)
            np.save(cov_dir / "cov.npy", cov)
            np.save(cov_dir / "cov_reg.npy", cov_reg)
            if args.whiten_output:
                blocks_for_output = [
                    whiten_block_rows(X, mu=mu, cov_reg=cov_reg) for X in blocks_for_output
                ]
                for meta in meta_for_output:
                    meta["whitened"] = True
        elif args.whiten_output and not blocks_for_output:
            anomalies.add("whiten_skipped_empty_group", f"group={group_id}")

        if blocks_for_output:
            payload = {
                "blocks": blocks_for_output,
                "meta": meta_for_output,
                "group_id": group_id,
                "group_key": group_key,
                "feature_mode": args.use_mode,
                "filter_mode": args.filter_mode,
                "group_by_move": bool(args.group_by_move),
                "whitened": bool(args.whiten_output),
                "covariance_saved": bool(args.compute_cov and bool(blocks_for_output)),
            }
            with (outdir / output_file).open("wb") as fh:
                pickle.dump(payload, fh, protocol=4)

        for X, meta in zip(blocks_for_output, meta_for_output):
            blocks_index_rows.append(
                [
                    group_id,
                    meta.get("ens_name"),
                    meta.get("parent_move"),
                    meta.get("pin"),
                    meta.get("cstep"),
                    meta.get("accepted"),
                    meta.get("status"),
                    meta.get("n_points"),
                    int(np.asarray(X).shape[1]),
                    output_file,
                ]
            )

        summary_rows.append(
            [
                group_id,
                ens_name,
                parent_move_group if parent_move_group is not None else "",
                n_blocks_raw,
                n_blocks_kept,
                n_shoot_rows_raw,
                n_shoot_rows_kept,
                "" if acceptance_rate is None else f"{acceptance_rate:.10g}",
                "" if acceptance_rate_n is None else acceptance_rate_n,
                acceptance_rate_fallback,
                missing_move_rows,
                shoot_dup_removed,
                shoot_malformed_dropped,
                vector_dim,
                "" if dim_mode is None else dim_mode,
                n_blocks_dim_dropped,
                "" if np.isnan(median_block_size) else f"{median_block_size:.10g}",
                "" if np.isnan(mean_block_size) else f"{mean_block_size:.10g}",
                "" if cstep_min is None else cstep_min,
                "" if cstep_max is None else cstep_max,
                int(n_blocks_kept < args.min_blocks),
            ]
        )

    write_tsv(
        outdir / "blocks_index.tsv",
        [
            "group_id",
            "ens_name",
            "parent_move",
            "pin",
            "cstep",
            "accepted",
            "status",
            "n_points",
            "dim",
            "output_file",
        ],
        blocks_index_rows,
    )

    write_tsv(
        outdir / "summary.tsv",
        [
            "group_id",
            "ens_name",
            "parent_move",
            "n_blocks_raw",
            "n_blocks_kept",
            "n_shoot_rows_raw",
            "n_shoot_rows_kept",
            "acceptance_rate",
            "acceptance_rate_n",
            "acceptance_rate_fallback",
            "missing_move_rows",
            "shoot_duplicates_removed",
            "shoot_malformed_rows_dropped",
            "vector_dim",
            "dim_mode",
            "n_blocks_dim_dropped",
            "median_block_size",
            "mean_block_size",
            "cstep_min",
            "cstep_max",
            "below_min_blocks",
        ],
        summary_rows,
    )

    if mc_stats:
        for key, val in sorted(mc_stats.items()):
            anomalies.add("mc_stats", f"{key}={val}")

    anomalies.write(outdir / "anomalies.log")

    print(f"Read move rows: {len(move_rows)}")
    print(f"Read shoot rows: {len(shoot_rows)}")
    print(f"Built raw blocks: {len(raw_blocks)}")
    print(f"Kept blocks ({args.filter_mode}): {len(kept_blocks)}")
    print(f"Wrote outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
