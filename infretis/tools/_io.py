"""Shared I/O utilities for analysis tools.

Centralises TSV reading/writing, value formatting, and schema constants
used by multiple analysis scripts. No numpy dependency — lightweight enough
to import from anywhere without pulling in heavy libraries.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Collection, Iterable, Optional

# ---------------------------------------------------------------------------
# Read-side schema constants
# ---------------------------------------------------------------------------

EPOCH_SUMMARY_REQUIRED = {
    "epoch_idx", "ens_name", "n_jumps_new", "acc_rate", "avg_path_length",
}

EPOCH_SUMMARY_OPTIONAL = {
    "n_attempted", "n_accepted", "n_jumps_old", "avg_subcycles",
    "avg_lambda_max", "ctrl_action", "ctrl_reason", "reward_eff",
    "reward",  # old schema alias
    "avg_subpath_length", "lp_over_ls_target_value",
}


# ---------------------------------------------------------------------------
# Type-conversion helpers
# ---------------------------------------------------------------------------

_NA_STRINGS = frozenset(("", "NA", "None", "nan"))


def safe_int(text: str, default: Optional[int] = None) -> Optional[int]:
    """Parse *text* as int, returning *default* for NA-like values.

    Raises ``ValueError`` for non-numeric, non-NA strings like ``"abc"``.
    """
    txt = text.strip()
    if txt in _NA_STRINGS:
        return default
    return int(txt)


def safe_float(text: str, default: float = float("nan")) -> float:
    """Parse *text* as float, returning *default* for empty/NA strings."""
    txt = text.strip()
    if txt in _NA_STRINGS:
        return default
    return float(txt)


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def fmt_val(val: Any, fmt: str = ".6g") -> str:
    """Format a value for TSV output.

    Returns ``"NA"`` for ``None``, and the string representation for
    ``nan``/``inf`` floats.  Uses :func:`math.isnan`/:func:`math.isinf`
    instead of numpy to keep this module lightweight.
    """
    if val is None:
        return "NA"
    try:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return str(val)
        return format(val, fmt)
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------------------------
# TSV writing
# ---------------------------------------------------------------------------

def write_tsv(
    path: Path, header: list[str], rows: Iterable[list[Any]]
) -> None:
    """Write a tab-delimited file with *header* and *rows*."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# TSV reading
# ---------------------------------------------------------------------------

def read_tsv_rows(
    path: Path,
    *,
    required: Collection[str],
    optional: Collection[str] | None = None,
    delimiter: str = "\t",
) -> tuple[list[dict[str, str]], set[str]]:
    """Read a TSV file, validate required columns, report optional presence.

    Returns ``(rows, present_optional)`` where each row is a raw
    ``dict[str, str]`` (no type conversion — callers decide fallback
    semantics).

    Raises ``FileNotFoundError`` if *path* does not exist and
    ``ValueError`` if required columns are missing.
    """
    opt = set(optional) if optional else set()
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        available = set(reader.fieldnames or [])
        missing_req = set(required) - available
        if missing_req:
            raise ValueError(
                f"Missing required columns in {path}: {missing_req}"
            )
        present_optional = opt & available
        rows = [dict(row) for row in reader]
    return rows, present_optional


def read_epoch_summary_rows(
    path: Path,
) -> tuple[list[dict[str, str]], set[str]]:
    """Read an epoch-summary TSV with schema-tolerant column handling.

    Applies the ``reward`` → ``reward_eff`` alias for old-schema files.
    Returns ``(rows, present_optional)`` — same contract as
    :func:`read_tsv_rows`.
    """
    rows, present_optional = read_tsv_rows(
        path,
        required=EPOCH_SUMMARY_REQUIRED,
        optional=EPOCH_SUMMARY_OPTIONAL,
    )

    # Normalize old-schema "reward" → "reward_eff"
    if "reward" in present_optional and "reward_eff" not in present_optional:
        for row in rows:
            if "reward" in row:
                row["reward_eff"] = row["reward"]
        present_optional = (present_optional - {"reward"}) | {"reward_eff"}

    return rows, present_optional


def read_softmax_debug_rows(path: Path) -> list[dict[str, str]]:
    """Read a softmax debug TSV, validating against the debug schema.

    Imports the canonical column list from ``epoch_ctrl`` to avoid
    duplicating the schema definition.
    """
    from infretis.core.epoch_ctrl import _EPOCH_SOFTMAX_DEBUG_COLS

    required = set(_EPOCH_SOFTMAX_DEBUG_COLS)
    rows, _ = read_tsv_rows(path, required=required)
    return rows
