"""Epoch-wise per-ensemble subtrajectory-count controller for WF/MWF moves.

Three controller modes are supported, selected via config["simulation"]["epoch_ctrl_mode"]:

  ``static``
      n_jumps is never updated by the epoch controller.  Stats are
      accumulated for targeted ensembles (if ``epoch_nsubpath_ens`` is
      set) but no TSV row is flushed and no value is changed.

  ``scheduled``
      n_jumps cycles through the list in ``epoch_nsubpath_vals`` at
      each epoch boundary.  Backward-compatible: inferred automatically
      when ``epoch_nsubpath_vals`` is present and ``epoch_ctrl_mode`` is
      absent.

  ``adaptive``
      n_jumps is adjusted each epoch by a one-step bounded rule driven
      by per-epoch acceptance rate, average path length, and average
      lambda_max.  Requires ``adaptive_*`` config keys; rejects
      ``epoch_nsubpath_vals``.

The trigger mode (``epoch_mode``) controls *when* epochs fire:

  ``global_step`` (default)
      Fires for all targeted ensembles simultaneously when ``cstep`` is a
      positive multiple of ``epoch_size``.

  ``per_ensemble_moves``
      Fires independently for each targeted ensemble when that ensemble's
      completed-move counter reaches a multiple of ``epoch_move_k``.  The
      counter tracks either every attempted move or only accepted moves,
      depending on ``epoch_count`` (default: ``"attempted"``).

Each firing also flushes a per-epoch statistics summary to
``epoch_summary.tsv`` via ``update_epoch_stats`` / ``mirror_epoch_ctrl``.
"""

import json
import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_EPOCH_SUMMARY_FNAME = "epoch_summary.tsv"

_ADAPTIVE_KEYS = (
    "adaptive_nsubpath_min",
    "adaptive_nsubpath_max",
    "adaptive_accept_low",
    "adaptive_accept_high",
    "adaptive_lambda_gain_low",
    "adaptive_pathlen_high",
)

_SOFTMAX_KEYS = (
    "epoch_nsubpath_choices",
    "softmax_eta0",
    "softmax_beta",
    "softmax_tau",
    "softmax_explore_floor",
    "softmax_init",
    "reward_proxy",
    "reward_eps_gain",
    "reward_eps_cost",
    "epoch_ctrl_seed",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _k_list(sim: dict) -> list:
    """Normalise epoch_move_k to a per-target list of positive ints."""
    n = len(sim.get("epoch_nsubpath_ens", []))
    k_raw = sim.get("epoch_move_k", 1)
    if isinstance(k_raw, list):
        return [int(k) for k in k_raw]
    return [int(k_raw)] * n


def _softmax(x) -> np.ndarray:
    """Numerically stable softmax (subtract max before exp)."""
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def _compute_q(logits, tau: float, eps_explore: float) -> np.ndarray:
    """Return explore-floor mixture: (1-eps)*softmax(logits/tau) + eps/K."""
    K = len(logits)
    p = _softmax(np.asarray(logits, dtype=float) / tau)
    return (1.0 - eps_explore) * p + eps_explore / K


def _epoch_local_rng(seed, ens_i: int, epoch_idx: int) -> np.random.Generator:
    """Deterministic, stateless RNG seeded by (seed, ens_i, epoch_idx)."""
    return np.random.default_rng([int(seed), int(ens_i), int(epoch_idx)])


def _epoch_sample(q, seed, ens_i: int, epoch_idx: int) -> int:
    """Sample action index from q using deterministic epoch-local RNG."""
    rng = _epoch_local_rng(seed, ens_i, epoch_idx)
    return int(rng.choice(len(q), p=q))


def _bounded_lambda_obs(lambda_max_val: float, lambda_i: float, lambda_upper: float) -> float:
    """Bounded path observable h_i(X) clipped to [0, 1]."""
    if lambda_upper <= lambda_i:
        raise ValueError(
            f"lambda_upper={lambda_upper} must be > lambda_i={lambda_i}."
        )
    return float(np.clip(
        (lambda_max_val - lambda_i) / (lambda_upper - lambda_i), 0.0, 1.0
    ))


def _obs_bounds_for_ensemble(state, ens_idx: int) -> tuple:
    """Return (lambda_i, lambda_upper) for ensemble ens_idx.

    ens["interfaces"] = [lambda_A, lambda_i, lambda_B]
    lambda_upper = tis_set["interface_cap"] if set, else lambda_B.
    """
    ens          = state.ensembles[ens_idx]
    lambda_i     = float(ens["interfaces"][1])
    lambda_upper = float(ens["tis_set"].get("interface_cap", ens["interfaces"][2]))
    return lambda_i, lambda_upper


def _empty_stats() -> dict:
    return {
        "n_attempted": 0,
        "n_accepted": 0,
        "path_length_sum": 0.0,
        "path_length_n": 0,
        "subcycles_sum": 0,
        "subcycles_n": 0,
        "lambda_max": float("-inf"),
        "lambda_max_sum": 0.0,
        "lambda_max_n": 0,
        "prev_obs":    float("nan"),   # sentinel: not yet initialized
        "gain_sq_sum": 0.0,
        "gain_sq_n":   0,
    }


def _append_epoch_tsv(path: str, header: str, row: str) -> None:
    newfile = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if newfile:
            fh.write(header.rstrip("\n") + "\n")
        fh.write(row.rstrip("\n") + "\n")


def _init_softmax_ctrl_state(choices, current_n_jumps, sim: dict) -> dict:
    """Initialise per-ensemble softmax controller state.

    Args:
        choices: admissible n_jumps values for this ensemble.
        current_n_jumps: the live n_jumps at first epoch boundary.
        sim: simulation config dict (reads softmax_init).

    Raises:
        ValueError: if current_n_jumps is not in choices.
    """
    if current_n_jumps not in choices:
        raise ValueError(
            f"Initial n_jumps={current_n_jumps} is not in "
            f"epoch_nsubpath_choices={choices}. "
            "The initial active value must belong to the admissible set."
        )
    K = len(choices)
    if sim["softmax_init"] == "current":
        idx = choices.index(current_n_jumps)
        logits = [1.0 if i == idx else 0.0 for i in range(K)]
    else:
        logits = [0.0] * K  # uniform
    return {
        "choices": list(choices),
        "logits": logits,
        "update_count": 0,
        "last_choice_idx": None,
        "initialized": False,
        "ema_abs_reward": 1e-4,   # conservative init; updated each boundary
    }


def _normalize_reward(
    raw_reward: float,
    ema_abs_reward: float,
    rho: float = 0.1,
    floor: float = 1e-6,
    rclip: float = 5.0,
) -> tuple:
    """EMA-normalize raw_reward and clip.

    Non-finite raw_reward (nan / inf) is passed through as nan without
    updating the EMA so that first-epoch nans do not corrupt the state.

    Formula (finite case only):
        ema_abs_reward = (1 - rho) * ema_abs_reward + rho * |raw_reward|
        denom          = max(ema_abs_reward, floor)
        reward_eff     = clip(raw_reward / denom, -rclip, rclip)

    Args:
        raw_reward:     raw proxy value from _compute_epoch_reward.
        ema_abs_reward: current EMA of |raw_reward| (per-ensemble state).
        rho:            EMA smoothing factor (default 0.1).
        floor:          minimum denominator to avoid division by zero (default 1e-6).
        rclip:          symmetric clip bound (default 5.0).

    Returns:
        (reward_eff, updated_ema_abs_reward)
    """
    if not np.isfinite(raw_reward):
        return float("nan"), ema_abs_reward
    ema_abs_reward = (1.0 - rho) * ema_abs_reward + rho * abs(raw_reward)
    denom = max(ema_abs_reward, floor)
    reward_eff = raw_reward / denom
    reward_eff = max(-rclip, min(rclip, reward_eff))
    return reward_eff, ema_abs_reward


def _compute_epoch_reward(buf: dict, sim: dict) -> float:
    """Compute reward proxy for one completed epoch buffer.

    Dispatches on ``sim["reward_proxy"]``.  Guaranteed strictly positive.
    """
    proxy    = sim.get("reward_proxy", "lambda_vs_subcycles_v1")
    eps_gain = sim.get("reward_eps_gain", 1e-12)
    eps_cost = sim.get("reward_eps_cost", 1e-12)
    avg_sub  = (
        buf["subcycles_sum"] / buf["subcycles_n"]
        if buf["subcycles_n"] > 0
        else 1.0
    )
    if proxy == "empirical_dirichlet_lambda_v1":
        gain = (
            buf.get("gain_sq_sum", 0.0) / buf["gain_sq_n"]
            if buf.get("gain_sq_n", 0) > 0
            else 0.0
        )
        return (gain + eps_gain) / (avg_sub + eps_cost)
    else:  # lambda_vs_subcycles_v1 — existing behaviour unchanged
        avg_lmax = (
            buf["lambda_max_sum"] / buf["lambda_max_n"]
            if buf["lambda_max_n"] > 0
            else 0.0
        )
        return (avg_lmax + eps_gain) / (avg_sub + eps_cost)


def _flush_epoch_stats(
    state,
    ens_i: int,
    epoch_idx: int,
    old_n_jumps,
    new_n_jumps: int,
    ctrl_mode: str,
    ctrl_action: str,
    ctrl_reason: str,
    sim: dict,
    sd_extras=None,
) -> None:
    """Write one summary row for ensemble ens_i and reset its buffer.

    ``sd_extras`` is an optional dict with softmax_dirichlet-specific
    fields (choice_idx, choice_value, reward, eta, tau, explore_floor,
    probs_json).  When present, extra columns are appended; old callers
    pass nothing and are unaffected.
    """
    buf = state.ensemble_epoch_stats.get(ens_i, _empty_stats())
    n_att = buf["n_attempted"]
    n_acc = buf["n_accepted"]
    acc_rate = n_acc / n_att if n_att > 0 else float("nan")
    avg_len = (
        buf["path_length_sum"] / buf["path_length_n"]
        if buf["path_length_n"] > 0
        else float("nan")
    )
    avg_sub = (
        buf["subcycles_sum"] / buf["subcycles_n"]
        if buf["subcycles_n"] > 0
        else float("nan")
    )
    lmax = (
        buf["lambda_max"]
        if buf["lambda_max"] != float("-inf")
        else float("nan")
    )
    lmax_n = buf.get("lambda_max_n", 0)
    avg_lmax = (
        buf.get("lambda_max_sum", 0.0) / lmax_n
        if lmax_n > 0
        else float("nan")
    )

    epoch_mode = sim.get("epoch_mode", "global_step")
    epoch_count = sim.get("epoch_count", "attempted")

    header = (
        "epoch_idx\tens_name\tn_attempted\tn_accepted\t"
        "acc_rate\tavg_path_length\tavg_subcycles\tlambda_max\t"
        "n_jumps_old\tn_jumps_new\t"
        "epoch_mode\tepoch_count\tctrl_mode\tavg_lambda_max\t"
        "ctrl_action\tctrl_reason"
    )
    row = (
        f"{epoch_idx}\t{ens_i:03d}\t{n_att}\t{n_acc}\t"
        f"{acc_rate:.4f}\t{avg_len:.2f}\t{avg_sub:.2f}\t"
        f"{lmax:.6g}\t{old_n_jumps}\t{new_n_jumps}\t"
        f"{epoch_mode}\t{epoch_count}\t{ctrl_mode}\t"
        f"{avg_lmax:.6g}\t{ctrl_action}\t{ctrl_reason}"
    )

    if sd_extras is not None:
        header += (
            "\tchoice_idx\tchoice_value\treward_raw\treward_eff\tema_abs_reward\teta"
            "\ttau\texplore_floor\tprobs_json"
        )
        row += (
            f"\t{sd_extras['choice_idx']}\t{sd_extras['choice_value']}"
            f"\t{sd_extras['reward_raw']:.6g}\t{sd_extras['reward_eff']:.6g}"
            f"\t{sd_extras['ema_abs_reward']:.6g}\t{sd_extras['eta']:.6g}"
            f"\t{sd_extras['tau']:.6g}\t{sd_extras['explore_floor']:.6g}"
            f"\t{sd_extras['probs_json']}"
        )

    out_dir = state.config.get("output", {}).get("data_dir", ".")
    _append_epoch_tsv(
        os.path.join(out_dir, _EPOCH_SUMMARY_FNAME), header, row
    )
    state.ensemble_epoch_stats[ens_i] = _empty_stats()


def _infer_ctrl_mode(sim: dict) -> str:
    """Return the effective epoch ctrl mode for sim config.

    If ``epoch_ctrl_mode`` is explicit, return it directly.  Otherwise
    infer from the presence of ``epoch_nsubpath_vals``.
    """
    if "epoch_ctrl_mode" in sim:
        return sim["epoch_ctrl_mode"]
    if sim.get("epoch_nsubpath_vals"):
        return "scheduled"
    return "static"


def _decide_adaptive_n_jumps(state, ens_i: int, sim: dict):
    """Compute new n_jumps for adaptive mode.

    Pure function — no side effects.

    Returns:
        Tuple of (new_n_jumps, ctrl_action, ctrl_reason).
    """
    buf = state.ensemble_epoch_stats.get(ens_i, _empty_stats())
    n_att = buf["n_attempted"]
    n_acc = buf["n_accepted"]
    acc_rate = n_acc / n_att if n_att > 0 else 0.0

    lmax_n = buf.get("lambda_max_n", 0)
    avg_lambda_max = (
        buf.get("lambda_max_sum", 0.0) / lmax_n if lmax_n > 0 else 0.0
    )

    path_n = buf["path_length_n"]
    avg_path_len = buf["path_length_sum"] / path_n if path_n > 0 else 0.0

    accept_low = sim.get("adaptive_accept_low", 0.20)
    lambda_gain_low = sim.get("adaptive_lambda_gain_low", 0.05)
    pathlen_high = sim.get("adaptive_pathlen_high", 400)
    n_min = sim.get("adaptive_nsubpath_min", 1)
    n_max = sim.get("adaptive_nsubpath_max", 8)

    old_n_jumps = state.ensembles[ens_i]["tis_set"].get("n_jumps", 2)

    if acc_rate < accept_low and avg_lambda_max < lambda_gain_low:
        delta = +1
        action, reason = "inc", "low_accept_low_explore"
    elif avg_path_len > pathlen_high and acc_rate >= accept_low:
        delta = -1
        action, reason = "dec", "high_cost_low_gain"
    else:
        delta = 0
        action, reason = "hold", "within_band"

    new_n_jumps = max(n_min, min(n_max, old_n_jumps + delta))
    return new_n_jumps, action, reason


def _apply_scheduled_ctrl(
    state, ens_i: int, vals: list, epoch_idx: int, sim: dict
) -> None:
    """Flush epoch stats and set n_jumps from schedule for ensemble ens_i."""
    if ens_i in state.ensembles and vals:
        new_val = int(vals[epoch_idx % len(vals)])
        old_val = state.ensembles[ens_i]["tis_set"].get("n_jumps")
        _flush_epoch_stats(
            state, ens_i, epoch_idx, old_val, new_val,
            ctrl_mode="scheduled",
            ctrl_action="set",
            ctrl_reason="scheduled_epoch_value",
            sim=sim,
        )
        state.ensembles[ens_i]["tis_set"]["n_jumps"] = new_val
        logger.info(
            "Epoch %d: ensemble %03d n_jumps %s -> %d",
            epoch_idx, ens_i, old_val, new_val,
        )


def _apply_adaptive_ctrl(
    state, ens_i: int, epoch_idx: int, sim: dict
) -> None:
    """Flush epoch stats and set n_jumps from adaptive rule for ensemble ens_i."""
    if ens_i not in state.ensembles:
        return
    old_val = state.ensembles[ens_i]["tis_set"].get("n_jumps")
    new_val, ctrl_action, ctrl_reason = _decide_adaptive_n_jumps(
        state, ens_i, sim
    )
    _flush_epoch_stats(
        state, ens_i, epoch_idx, old_val, new_val,
        ctrl_mode="adaptive",
        ctrl_action=ctrl_action,
        ctrl_reason=ctrl_reason,
        sim=sim,
    )
    state.ensembles[ens_i]["tis_set"]["n_jumps"] = new_val
    logger.info(
        "Epoch %d: ensemble %03d n_jumps %s -> %d (%s)",
        epoch_idx, ens_i, old_val, new_val, ctrl_reason,
    )


def _apply_softmax_dirichlet_ctrl(
    state, ens_i: int, epoch_idx: int, sim: dict
) -> None:
    """Flush epoch stats and apply softmax-Dirichlet n_jumps update.

    On the first epoch boundary for an ensemble, initialises the
    controller state and samples the first action (no logit update —
    there is no prior reward for epoch 0).  On subsequent boundaries,
    computes the importance-weighted reward, updates the logit for the
    action taken, and samples the next action.
    """
    if ens_i not in state.ensembles:
        return

    ctrl_state = state.softmax_ctrl_state.get(ens_i)
    target_idx = sim["epoch_nsubpath_ens"].index(ens_i)
    choices_cfg = sim["epoch_nsubpath_choices"][target_idx]

    if ctrl_state is None:
        # First boundary: initialise, sample first action, no logit update.
        current_n_jumps = state.ensembles[ens_i]["tis_set"].get("n_jumps")
        ctrl_state = _init_softmax_ctrl_state(choices_cfg, current_n_jumps, sim)
        q = _compute_q(
            ctrl_state["logits"], sim["softmax_tau"], sim["softmax_explore_floor"]
        )
        choice_idx = _epoch_sample(q, sim["epoch_ctrl_seed"], ens_i, epoch_idx)
        raw_reward = float("nan")
        reward_eff = float("nan")
        eta = float("nan")
    else:
        # Subsequent boundary: update logits from reward, sample next action.
        old_q = _compute_q(
            ctrl_state["logits"], sim["softmax_tau"], sim["softmax_explore_floor"]
        )
        buf = state.ensemble_epoch_stats.get(ens_i, _empty_stats())
        raw_reward = _compute_epoch_reward(buf, sim)
        reward_eff, ctrl_state["ema_abs_reward"] = _normalize_reward(
            raw_reward,
            float(ctrl_state.get("ema_abs_reward", 1e-4)),
            rho=float(sim.get("softmax_reward_ema_rho",   0.1)),
            floor=float(sim.get("softmax_reward_ema_floor", 1e-6)),
            rclip=float(sim.get("softmax_reward_eff_clip",  5.0)),
        )
        uc = ctrl_state["update_count"]
        eta = sim["softmax_eta0"] / (uc + 1) ** sim["softmax_beta"]
        A = ctrl_state["last_choice_idx"]
        new_logits = list(ctrl_state["logits"])
        new_logits[A] += eta * reward_eff / old_q[A]
        ctrl_state["logits"] = new_logits
        ctrl_state["update_count"] = uc + 1
        q = _compute_q(
            ctrl_state["logits"], sim["softmax_tau"], sim["softmax_explore_floor"]
        )
        choice_idx = _epoch_sample(q, sim["epoch_ctrl_seed"], ens_i, epoch_idx)

    old_n_jumps = state.ensembles[ens_i]["tis_set"].get("n_jumps")
    new_n_jumps = ctrl_state["choices"][choice_idx]
    action = "sample_hold" if new_n_jumps == old_n_jumps else "sample_set"

    ctrl_state["last_choice_idx"] = choice_idx
    ctrl_state["initialized"] = True
    state.softmax_ctrl_state[ens_i] = ctrl_state

    _flush_epoch_stats(
        state, ens_i, epoch_idx, old_n_jumps, new_n_jumps,
        ctrl_mode="softmax_dirichlet",
        ctrl_action=action,
        ctrl_reason="softmax_dirichlet_epoch_update",
        sim=sim,
        sd_extras=dict(
            choice_idx=choice_idx,
            choice_value=new_n_jumps,
            reward_raw=raw_reward,
            reward_eff=reward_eff,
            ema_abs_reward=ctrl_state.get("ema_abs_reward", 1e-4),
            eta=eta,
            tau=sim["softmax_tau"],
            explore_floor=sim["softmax_explore_floor"],
            probs_json=json.dumps([round(float(x), 6) for x in q]),
        ),
    )
    state.ensembles[ens_i]["tis_set"]["n_jumps"] = new_n_jumps
    logger.info(
        "Epoch %d: ensemble %03d n_jumps %s -> %d (softmax_dirichlet, choice_idx=%d)",
        epoch_idx, ens_i, old_n_jumps, new_n_jumps, choice_idx,
    )


def _dispatch_ctrl(
    state, ens_i: int, epoch_idx: int, ctrl_mode: str, sim: dict, vals=None
) -> None:
    """Dispatch to the appropriate ctrl function for the given mode."""
    if ctrl_mode == "scheduled":
        _apply_scheduled_ctrl(state, ens_i, vals, epoch_idx, sim)
    elif ctrl_mode == "adaptive":
        _apply_adaptive_ctrl(state, ens_i, epoch_idx, sim)
    elif ctrl_mode == "softmax_dirichlet":
        _apply_softmax_dirichlet_ctrl(state, ens_i, epoch_idx, sim)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_epoch_ctrl(config: dict, state=None) -> None:
    """Validate epoch controller config keys.

    Args:
        config: full simulation config dict.
        state: unused; reserved for future live-state checks.

    Raises:
        ValueError: on any configuration inconsistency.
    """
    sim = config["simulation"]
    n_ens = len(sim["interfaces"])
    sh_moves = sim["shooting_moves"]
    n_sh_moves = len(sh_moves)

    # Validate epoch_ctrl_mode if explicitly set
    explicit_ctrl_mode = sim.get("epoch_ctrl_mode")
    if explicit_ctrl_mode is not None and explicit_ctrl_mode not in (
        "static", "scheduled", "adaptive", "softmax_dirichlet"
    ):
        raise ValueError(
            f"epoch_ctrl_mode '{explicit_ctrl_mode}' is not recognised; "
            "valid options are 'static', 'scheduled', 'adaptive', "
            "and 'softmax_dirichlet'."
        )

    ctrl_mode = _infer_ctrl_mode(sim)
    ens_targets = sim.get("epoch_nsubpath_ens", [])
    val_schedules = sim.get("epoch_nsubpath_vals", [])
    has_adaptive_keys = any(k in sim for k in _ADAPTIVE_KEYS)

    # Mode-specific key checks
    if ctrl_mode == "static":
        if val_schedules:
            raise ValueError(
                "epoch_nsubpath_vals must not be set in static mode."
            )
        if has_adaptive_keys:
            raise ValueError(
                "adaptive_* keys must not be set in static mode."
            )

    elif ctrl_mode == "scheduled":
        if has_adaptive_keys:
            raise ValueError(
                "adaptive_* keys must not be set in scheduled mode."
            )
        if len(ens_targets) != len(val_schedules):
            raise ValueError(
                f"epoch_nsubpath_ens has {len(ens_targets)} entries but "
                f"epoch_nsubpath_vals has {len(val_schedules)} entries; "
                "they must be the same length."
            )

    elif ctrl_mode == "adaptive":
        if val_schedules:
            raise ValueError(
                "epoch_nsubpath_vals must not be set in adaptive mode."
            )
        missing = [k for k in _ADAPTIVE_KEYS if k not in sim]
        if missing:
            raise ValueError(
                f"adaptive mode requires these config keys: "
                f"{', '.join(missing)}."
            )
        n_min = sim["adaptive_nsubpath_min"]
        n_max = sim["adaptive_nsubpath_max"]
        accept_low = sim["adaptive_accept_low"]
        accept_high = sim["adaptive_accept_high"]
        if n_min < 1:
            raise ValueError(
                "adaptive_nsubpath_min must be >= 1."
            )
        if n_max < n_min:
            raise ValueError(
                "adaptive_nsubpath_max must be >= adaptive_nsubpath_min."
            )
        if not (0 < accept_low < accept_high < 1):
            raise ValueError(
                "Required: 0 < adaptive_accept_low < adaptive_accept_high < 1."
            )

    elif ctrl_mode == "softmax_dirichlet":
        if sim.get("epoch_mode") != "per_ensemble_moves":
            raise ValueError(
                "epoch_ctrl_mode='softmax_dirichlet' requires "
                "epoch_mode='per_ensemble_moves'."
            )
        if sim.get("epoch_count", "attempted") != "attempted":
            raise ValueError(
                "epoch_ctrl_mode='softmax_dirichlet' requires "
                "epoch_count='attempted'."
            )
        if val_schedules:
            raise ValueError(
                "epoch_nsubpath_vals must not be set in softmax_dirichlet mode."
            )
        if has_adaptive_keys:
            raise ValueError(
                "adaptive_* keys must not be set in softmax_dirichlet mode."
            )
        missing = [k for k in _SOFTMAX_KEYS if k not in sim]
        if missing:
            raise ValueError(
                f"softmax_dirichlet mode requires these config keys: "
                f"{', '.join(missing)}."
            )
        choices_list = sim["epoch_nsubpath_choices"]
        if len(choices_list) != len(ens_targets):
            raise ValueError(
                f"epoch_nsubpath_choices has {len(choices_list)} entries but "
                f"epoch_nsubpath_ens has {len(ens_targets)} entries; "
                "they must be the same length."
            )
        for idx, choices in enumerate(choices_list):
            if not choices:
                raise ValueError(
                    f"epoch_nsubpath_choices[{idx}] must be non-empty."
                )
            if any(int(v) < 1 for v in choices):
                raise ValueError(
                    f"epoch_nsubpath_choices[{idx}] must contain only "
                    "integers >= 1."
                )
        for ens_i in ens_targets:
            if not (0 <= ens_i < n_ens):
                raise ValueError(
                    f"epoch_nsubpath_ens contains invalid ensemble index "
                    f"{ens_i}; valid range is 0 \u2026 {n_ens - 1}."
                )
            ens_move = sh_moves[ens_i] if ens_i < n_sh_moves else None
            if ens_move not in ("wf", "mwf"):
                raise ValueError(
                    f"epoch_nsubpath_ens targets ensemble {ens_i} which uses "
                    f"move '{ens_move}'; only 'wf' and 'mwf' ensembles support "
                    "the n_jumps parameter."
                )
        if sim["softmax_eta0"] <= 0:
            raise ValueError("softmax_eta0 must be > 0.")
        beta = sim["softmax_beta"]
        if not (0.5 < beta <= 1.0):
            raise ValueError(
                "softmax_beta must satisfy 0.5 < softmax_beta <= 1.0."
            )
        if sim["softmax_tau"] <= 0:
            raise ValueError("softmax_tau must be > 0.")
        eps = sim["softmax_explore_floor"]
        if not (0 <= eps < 1):
            raise ValueError(
                "softmax_explore_floor must satisfy 0 <= eps < 1."
            )
        if sim["softmax_init"] not in ("uniform", "current"):
            raise ValueError(
                "softmax_init must be 'uniform' or 'current'."
            )
        if sim["reward_proxy"] not in (
            "lambda_vs_subcycles_v1", "empirical_dirichlet_lambda_v1"
        ):
            raise ValueError(
                "reward_proxy must be 'lambda_vs_subcycles_v1' or "
                "'empirical_dirichlet_lambda_v1'."
            )
        # Extra check for empirical proxy: validate upper bound > lambda_i per target.
        if sim["reward_proxy"] == "empirical_dirichlet_lambda_v1":
            ifaces  = sim["interfaces"]
            cap_cfg = sim.get("tis_set", {}).get("interface_cap", None)
            for ens_i in ens_targets:
                lambda_i     = ifaces[ens_i - 1]
                lambda_upper = cap_cfg if cap_cfg is not None else ifaces[-1]
                if lambda_upper <= lambda_i:
                    raise ValueError(
                        f"Upper bound {lambda_upper} must be > lambda_i={lambda_i} "
                        f"for ensemble {ens_i}."
                    )
        if not isinstance(sim["epoch_ctrl_seed"], int):
            raise ValueError("epoch_ctrl_seed must be an integer.")
        if sim.get("epoch_move_k") is None:
            raise ValueError(
                "epoch_move_k must be set when "
                "epoch_ctrl_mode='softmax_dirichlet'."
            )

    # Validate ensemble indices and move types for scheduled/adaptive
    if ctrl_mode in ("scheduled", "adaptive"):
        for ens_i in ens_targets:
            if not (0 <= ens_i < n_ens):
                raise ValueError(
                    f"epoch_nsubpath_ens contains invalid ensemble index {ens_i}; "
                    f"valid range is 0 \u2026 {n_ens - 1}."
                )
            ens_move = sh_moves[ens_i] if ens_i < n_sh_moves else None
            if ens_move not in ("wf", "mwf"):
                raise ValueError(
                    f"epoch_nsubpath_ens targets ensemble {ens_i} which uses "
                    f"move '{ens_move}'; only 'wf' and 'mwf' ensembles support "
                    "the n_jumps parameter."
                )

    # Trigger mode (epoch_mode) validation
    mode = sim.get("epoch_mode", "global_step")
    if mode not in ("global_step", "per_ensemble_moves"):
        raise ValueError(
            f"epoch_mode '{mode}' is not recognised; "
            "valid options are 'global_step' and 'per_ensemble_moves'."
        )

    if mode == "per_ensemble_moves" and ens_targets:
        k_raw = sim.get("epoch_move_k")
        if k_raw is None:
            raise ValueError(
                "epoch_move_k must be set when "
                "epoch_mode='per_ensemble_moves'."
            )
        if isinstance(k_raw, list):
            if len(k_raw) != len(ens_targets):
                raise ValueError(
                    f"epoch_move_k has {len(k_raw)} entries but "
                    f"epoch_nsubpath_ens has {len(ens_targets)}; "
                    "they must be the same length."
                )
            if any(int(k) <= 0 for k in k_raw):
                raise ValueError(
                    "All epoch_move_k values must be positive integers."
                )
        elif int(k_raw) <= 0:
            raise ValueError("epoch_move_k must be a positive integer.")

    count_mode = sim.get("epoch_count", "attempted")
    if count_mode not in ("attempted", "accepted"):
        raise ValueError(
            f"epoch_count '{count_mode}' is not recognised; "
            "valid options are 'attempted' and 'accepted'."
        )


def update_epoch_stats(
    state,
    ens_idx: int,
    accepted: bool,
    path_length: float,
    subcycles: int,
    lambda_max: float,
) -> None:
    """Accumulate one move into the epoch stats buffer for ensemble ens_idx.

    Only accumulates for ensembles listed in ``epoch_nsubpath_ens``; no-ops
    for all others.

    Args:
        state: REPEX_state instance.
        ens_idx: self.ensembles index for this move.
        accepted: whether the move was accepted.
        path_length: length of the trial path in MD steps.
        subcycles: number of MD subcycles consumed by the move.
        lambda_max: maximum order parameter of the trial path.
    """
    ens_targets = state.config["simulation"].get("epoch_nsubpath_ens", [])
    if ens_idx not in ens_targets:
        return
    buf = state.ensemble_epoch_stats.setdefault(ens_idx, _empty_stats())
    buf["n_attempted"] += 1
    if accepted:
        buf["n_accepted"] += 1
    buf["path_length_sum"] += path_length
    buf["path_length_n"] += 1
    buf["subcycles_sum"] += subcycles
    buf["subcycles_n"] += 1
    if lambda_max > buf["lambda_max"]:
        buf["lambda_max"] = lambda_max
    buf["lambda_max_sum"] += lambda_max
    buf["lambda_max_n"] += 1

    proxy = state.config["simulation"].get("reward_proxy")
    if proxy == "empirical_dirichlet_lambda_v1":
        lambda_i, lambda_upper = _obs_bounds_for_ensemble(state, ens_idx)
        curr_obs = (
            _bounded_lambda_obs(lambda_max, lambda_i, lambda_upper)
            if accepted
            else buf.get("prev_obs", float("nan"))    # rejected: realized state unchanged
        )
        prev = buf.get("prev_obs", float("nan"))
        if math.isnan(prev):
            if not math.isnan(curr_obs):              # first accepted move → initialise
                buf["prev_obs"] = curr_obs
        else:
            delta = curr_obs - prev
            buf["gain_sq_sum"] = buf.get("gain_sq_sum", 0.0) + delta * delta
            buf["gain_sq_n"]   = buf.get("gain_sq_n", 0) + 1
            buf["prev_obs"]    = curr_obs


def apply_epoch_ctrl(state, cstep: int) -> None:
    """Apply epoch-wise n_jumps updates to targeted ensembles.

    Dispatches first on ``epoch_ctrl_mode`` (static / scheduled / adaptive),
    then on ``epoch_mode`` (global_step / per_ensemble_moves) to determine
    when to fire.

    * ``static``: never updates n_jumps; returns immediately.
    * ``scheduled``: cycles n_jumps through ``epoch_nsubpath_vals`` at each
      epoch boundary.
    * ``adaptive``: adjusts n_jumps by a one-step bounded rule driven by
      per-epoch acceptance rate, average path length, and average lambda_max.

    Args:
        state: REPEX_state instance whose ensembles are mutated in-place.
        cstep: current completed-move step count (used in global_step mode).
    """
    sim = state.config["simulation"]
    ctrl_mode = _infer_ctrl_mode(sim)

    if ctrl_mode == "static":
        return

    ens_targets = sim.get("epoch_nsubpath_ens", [])
    if not ens_targets:
        return

    val_schedules = (
        sim.get("epoch_nsubpath_vals", [])
        if ctrl_mode == "scheduled"
        else [None] * len(ens_targets)
    )

    mode = sim.get("epoch_mode", "global_step")

    if mode == "global_step":
        epoch_size = sim.get("epoch_size", 0)
        if not (epoch_size > 0 and cstep > 0 and cstep % epoch_size == 0):
            return
        epoch_idx = cstep // epoch_size
        for ens_i, vals in zip(ens_targets, val_schedules):
            _dispatch_ctrl(state, ens_i, epoch_idx, ctrl_mode, sim, vals)

    elif mode == "per_ensemble_moves":
        for ens_i, vals, k in zip(ens_targets, val_schedules, _k_list(sim)):
            count = state.ensemble_move_counts.get(ens_i, 0)
            last_fired = state.ensemble_last_fired_count.get(ens_i, -1)
            if count > 0 and count % k == 0 and count != last_fired:
                state.ensemble_last_fired_count[ens_i] = count
                epoch_idx = count // k
                _dispatch_ctrl(state, ens_i, epoch_idx, ctrl_mode, sim, vals)


def mirror_epoch_ctrl(state, config: dict) -> None:
    """Mirror per-ensemble n_jumps, mwf_nsubpath, and move counts into config.

    Writes ``ensemble_nsubpath``, ``ensemble_mwf_nsubpath`` (into
    ``config["simulation"]``) and ``ensemble_move_counts`` (into
    ``config["current"]``) so that ``write_toml()`` serializes the current
    live state and ``initiate_ensembles()`` restores it exactly on restart.

    Args:
        state: REPEX_state instance to read current values from.
        config: full simulation config dict; mutated in-place.
    """
    global_tis = config["simulation"]["tis_set"]
    ens_keys = sorted(state.ensembles.keys())

    config["simulation"]["ensemble_nsubpath"] = [
        state.ensembles[i]["tis_set"].get(
            "n_jumps", global_tis.get("n_jumps", 2)
        )
        for i in ens_keys
    ]
    config["simulation"]["ensemble_mwf_nsubpath"] = [
        state.ensembles[i]["tis_set"].get(
            "mwf_nsubpath", global_tis.get("mwf_nsubpath", 3)
        )
        for i in ens_keys
    ]
    # TOML requires string keys; int keys are stored as strings and converted
    # back to int on restore in REPEX_state.__init__().
    config["current"]["ensemble_move_counts"] = {
        str(i): state.ensemble_move_counts.get(i, 0)
        for i in ens_keys
    }
    config["current"]["ensemble_last_fired_count"] = {
        str(i): state.ensemble_last_fired_count.get(i, -1)
        for i in ens_keys
    }

    # Persist softmax_dirichlet controller state and partial-epoch stats so
    # that restarts preserve logits, update_count, and mid-epoch rewards.
    if state.softmax_ctrl_state:
        config["simulation"]["softmax_ctrl_state"] = {
            str(i): {
                "choices":         cs["choices"],
                "logits":          cs["logits"],
                "update_count":    cs["update_count"],
                "last_choice_idx": cs["last_choice_idx"],
                "initialized":     cs["initialized"],
                "ema_abs_reward":  cs.get("ema_abs_reward", 1e-4),
            }
            for i, cs in state.softmax_ctrl_state.items()
        }
        config["simulation"]["softmax_epoch_stats"] = {
            str(i): state.ensemble_epoch_stats.get(i, _empty_stats())
            for i in state.softmax_ctrl_state
        }
