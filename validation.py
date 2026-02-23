from __future__ import annotations

"""
Validation pipeline for main1-style training artifacts.

This script evaluates one-step-ahead prediction error on a chosen validation
time window for three models:
1) Federated estimated model (run-specific A_mn/B_mn from Monte Carlo history)
2) DKF baseline (independent local Kalman filters)
3) CKF baseline (centralized Kalman filter with true full model)

Important indexing convention used everywhere in this file:
  - At time t, prediction uses control u_{t-1}.
  - For t = 0, control is a zero vector.
This keeps the evaluator consistent with the system equation:
  x_t = A x_{t-1} + B u_{t-1} + w_{t-1}.

No CLI is used by design. All knobs are constants in the config section below.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_retriever import RetrieveData
from kalman_filter import KalmanFilter

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

colorblind_colors = [
    "#f781bf",
    "#377eb8",
    "#984ea3",
    "#ff7f00",
    "#4daf4a",
    "#a65628",
    "#ffff33",
    "#e41a1c",
]


# -----------------------------------------------------------------------------
# Validation configuration (no CLI; edit these directly)
# -----------------------------------------------------------------------------
# Path to the same training config used by main1.
CONFIG_PATH = "config.ini"

# If None, defaults to:
#   <results_location>/main1/monte_carlo_history.pkl
# where results_location comes from config.ini loaded by RetrieveData.
MONTE_HISTORY_PATH: str = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/uai2026/fedcount-results/main1/monte_carlo_history.pkl"

# If None, defaults to:
#   <directory_of_monte_history>/validation
VALIDATION_OUTPUT_DIR: str = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/uai2026/fedcount-results/valid_results"

# Validation window:
# - If VALID_T0 is None, it defaults to training_time.
# - If VALID_T1 is None, it defaults to min(total_time, VALID_T0 + VALID_WINDOW).
VALID_T0: int | None = 15000
VALID_T1: int | None = 20000
VALID_WINDOW: int = 5000

# Warm-start Kalman filters by running predict/update from t=0..VALID_T0-1.
WARM_START: bool = True

# If True, validation start must be >= training_time to avoid leakage.
REQUIRE_HOLDOUT_AFTER_TRAIN: bool = True

# Plot smoothing window for squared-error traces.
# Set to 1 to disable smoothing.
SMOOTH_WINDOW: int = 100


def _to_1d_int(arr: np.ndarray) -> np.ndarray:
    """Convert any numeric array-like to a flattened 1D integer NumPy array."""
    return np.asarray(arr).astype(int).reshape(-1)


def _resolve_paths(data: RetrieveData) -> tuple[Path, Path]:
    """
    Resolve the Monte Carlo artifact path and output directory.

    Parameters
    ----------
    data : RetrieveData
        Loaded data object used only to access config-derived `results_location`
        when MONTE_HISTORY_PATH is not explicitly set.

    Returns
    -------
    tuple[Path, Path]
        (monte_path, out_dir), where:
        - monte_path points to monte_carlo_history.pkl
        - out_dir is the validation output folder and is created if missing
    """
    if MONTE_HISTORY_PATH is None:
        monte_path = Path(data.results_location) / "main1" / "monte_carlo_history.pkl"
    else:
        monte_path = Path(MONTE_HISTORY_PATH)

    if not monte_path.exists():
        raise FileNotFoundError(f"Monte Carlo artifact not found: {monte_path}")

    if VALIDATION_OUTPUT_DIR is None:
        out_dir = monte_path.parent / "validation"
    else:
        out_dir = Path(VALIDATION_OUTPUT_DIR)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    return monte_path, out_dir


def _resolve_validation_window(data: RetrieveData) -> tuple[int, int]:
    """
    Determine the absolute validation window [t0, t1).

    Behavior:
    - If VALID_T0 is not set, use training_time from config/data.
    - If VALID_T1 is not set, use t0 + VALID_WINDOW (capped by total_time).
    - Always validates bounds against total_time.

    Returns
    -------
    tuple[int, int]
        (t0, t1) with t0 < t1 and 0 <= t0,t1 <= total_time.
    """
    t0 = int(data.training_time) if VALID_T0 is None else int(VALID_T0)
    if VALID_T1 is None:
        t1 = min(int(data.total_time), t0 + int(VALID_WINDOW))
    else:
        t1 = int(VALID_T1)

    if t0 < 0 or t1 <= t0 or t1 > int(data.total_time):
        raise ValueError(
            f"Invalid validation window: [{t0}, {t1}) with total_time={data.total_time}."
        )
    if REQUIRE_HOLDOUT_AFTER_TRAIN and t0 < int(data.training_time):
        raise ValueError(
            f"Validation start t0={t0} is before training_time={data.training_time}. "
            "Set VALID_T0 >= training_time or disable REQUIRE_HOLDOUT_AFTER_TRAIN."
        )
    return t0, t1


def _build_global_streams_from_local_packs(
    data: RetrieveData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build full global observation/control streams from per-client full streams.

    The data retriever stores each client's full trajectory in:
      local_learners_pack[comp_k]["Y_full"], local_learners_pack[comp_k]["U_full"].
    This function stitches those local blocks into global matrices using the
    configured start-index vectors.

    Returns:
      Y_full_global, U_full_global, d_vec, s_vec, d_starts, s_starts
    """
    m_count = int(data.num_components)
    d_vec = _to_1d_int(data.output_size_vec)
    s_vec = _to_1d_int(data.input_size_vec)
    d_starts = _to_1d_int(data.output_start_vec)
    s_starts = _to_1d_int(data.input_start_vec)
    total_time = int(data.total_time)

    y_global = np.zeros((int(np.sum(d_vec)), total_time))
    u_global = np.zeros((int(np.sum(s_vec)), total_time))

    for m in range(m_count):
        cid = f"{m + 1}"
        comp = data.local_learners_pack[f"comp_{cid}"]
        y_local = np.asarray(comp["Y_full"], dtype=float)
        u_local = np.asarray(comp["U_full"], dtype=float)

        # Global row ranges for this client's observation and input blocks.
        d0 = int(d_starts[m])
        d1 = d0 + int(d_vec[m])
        s0 = int(s_starts[m])
        s1 = s0 + int(s_vec[m])

        y_global[d0:d1, :] = y_local
        u_global[s0:s1, :] = u_local

    return y_global, u_global, d_vec, s_vec, d_starts, s_starts


def _kalman_predict_eval(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    c_mat: np.ndarray,
    q_mat: np.ndarray,
    r_mat: np.ndarray,
    p0_mat: np.ndarray,
    x0_vec: np.ndarray,
    y_full: np.ndarray,
    u_full: np.ndarray,
    valid_t0: int,
    valid_t1: int,
    d_vec: np.ndarray,
    d_starts: np.ndarray,
    warm_start: bool,
) -> dict[str, np.ndarray | float]:
    """
    Evaluate one-step-ahead prediction on [valid_t0, valid_t1) for a given KF.

    Protocol per time t:
      1) predict using u_{t-1}
      2) compute one-step prediction error: e_t = y_t - yhat_{t|t-1}
      3) accumulate squared errors for RMSE
      4) update filter with y_t

    If warm_start=True:
      The same predict/update steps are run from t=0..valid_t0-1 without
      scoring. This initializes filter state before holdout scoring starts.

    Returns
    -------
    dict with keys:
      - rmse: per-client RMSE over validation window
      - rmse_per_dim: per-client RMSE normalized by output dimension
      - rmse_overall: global RMSE across all outputs
      - sqerr_time: per-client squared-error trace over validation window
      - sqerr_sum: per-client total squared error
      - sqerr_total: total squared error across all clients
    """
    m_count = len(d_vec)
    n_steps = valid_t1 - valid_t0
    s_dim = int(b_mat.shape[1])

    kf = KalmanFilter(
        np.asarray(a_mat, dtype=float),
        np.asarray(b_mat, dtype=float),
        np.asarray(c_mat, dtype=float),
        np.asarray(q_mat, dtype=float),
        np.asarray(r_mat, dtype=float),
        np.asarray(p0_mat, dtype=float).copy(),
        np.asarray(x0_vec, dtype=float).reshape(-1, 1).copy(),
    )

    sqerr_sum = np.zeros(m_count, dtype=float)
    sqerr_time = np.zeros((m_count, n_steps), dtype=float)
    total_sqerr = 0.0

    # `start_t` controls whether we first warm-initialize the filter.
    start_t = 0 if warm_start else valid_t0
    step_idx = 0

    for t in range(start_t, valid_t1):
        # Lagged input usage by construction: u_{t-1}, with zero at t=0.
        u_prev = u_full[:, t - 1:t] if t > 0 else np.zeros((s_dim, 1))
        kf.predict(u_prev)

        y_t = y_full[:, t:t + 1]
        y_pred = c_mat @ kf.get_state()
        err = y_t - y_pred

        if t >= valid_t0:
            # Slice residual by client output blocks and accumulate ||e_t^(m)||_2^2.
            for m in range(m_count):
                d0 = int(d_starts[m])
                d1 = d0 + int(d_vec[m])
                sq = float(np.sum(err[d0:d1, :] ** 2))
                sqerr_sum[m] += sq
                sqerr_time[m, step_idx] = sq
            # Also keep a total error over all outputs.
            total_sqerr += float(np.sum(err ** 2))
            step_idx += 1

        # Standard Kalman correction using current measurement.
        kf.update(y_t)

    if step_idx != n_steps:
        raise RuntimeError(f"Internal indexing mismatch: step_idx={step_idx}, n_steps={n_steps}")

    rmse = np.sqrt(sqerr_sum / float(n_steps))
    rmse_per_dim = np.sqrt((sqerr_sum / d_vec.astype(float)) / float(n_steps))
    rmse_overall = float(np.sqrt(total_sqerr / float(n_steps)))

    return {
        "rmse": rmse,
        "rmse_per_dim": rmse_per_dim,
        "rmse_overall": rmse_overall,
        "sqerr_time": sqerr_time,
        "sqerr_sum": sqerr_sum,
        "sqerr_total": total_sqerr,
    }


def _evaluate_dkf_local(
    data: RetrieveData,
    valid_t0: int,
    valid_t1: int,
    warm_start: bool,
) -> dict[str, np.ndarray | float]:
    """
    Evaluate DKF baseline with independent local KFs (one per client).

    Each client m uses its own true local blocks:
      A_mm, B_mm, C_mm, Q_mm, R_mm, P0_mm, x0_mm
    and local full streams:
      Y_full_m, U_full_m

    The same one-step scoring protocol is used as in `_kalman_predict_eval`.
    """
    m_count = int(data.num_components)
    n_steps = valid_t1 - valid_t0

    sqerr_sum = np.zeros(m_count, dtype=float)
    sqerr_time = np.zeros((m_count, n_steps), dtype=float)
    total_sqerr = 0.0
    d_vec = _to_1d_int(data.output_size_vec)

    for m in range(m_count):
        cid = f"{m + 1}"
        comp = data.local_learners_pack[f"comp_{cid}"]

        a_mat = np.asarray(comp["A"], dtype=float)
        b_mat = np.asarray(comp["B"], dtype=float)
        c_mat = np.asarray(comp["C"], dtype=float)
        q_mat = np.asarray(comp["Q"], dtype=float)
        r_mat = np.asarray(comp["R"], dtype=float)
        p0_mat = np.asarray(comp["P0"], dtype=float)
        x0_vec = np.asarray(comp["x0"], dtype=float).reshape(-1, 1)
        y_full = np.asarray(comp["Y_full"], dtype=float)
        u_full = np.asarray(comp["U_full"], dtype=float)
        s_dim = int(b_mat.shape[1])

        kf = KalmanFilter(a_mat, b_mat, c_mat, q_mat, r_mat, p0_mat.copy(), x0_vec.copy())
        start_t = 0 if warm_start else valid_t0
        step_idx = 0

        for t in range(start_t, valid_t1):
            # Lagged control input for one-step prediction at time t.
            u_prev = u_full[:, t - 1:t] if t > 0 else np.zeros((s_dim, 1))
            kf.predict(u_prev)

            y_t = y_full[:, t:t + 1]
            y_pred = c_mat @ kf.get_state()
            err = y_t - y_pred

            if t >= valid_t0:
                sq = float(np.sum(err ** 2))
                sqerr_sum[m] += sq
                sqerr_time[m, step_idx] = sq
                total_sqerr += sq
                step_idx += 1

            kf.update(y_t)

        if step_idx != n_steps:
            raise RuntimeError(
                f"DKF indexing mismatch at client {cid}: step_idx={step_idx}, n_steps={n_steps}"
            )

    rmse = np.sqrt(sqerr_sum / float(n_steps))
    rmse_per_dim = np.sqrt((sqerr_sum / d_vec.astype(float)) / float(n_steps))
    rmse_overall = float(np.sqrt(total_sqerr / float(n_steps)))

    return {
        "rmse": rmse,
        "rmse_per_dim": rmse_per_dim,
        "rmse_overall": rmse_overall,
        "sqerr_time": sqerr_time,
        "sqerr_sum": sqerr_sum,
        "sqerr_total": total_sqerr,
    }


def _compute_dkf_states_full(data: RetrieveData) -> dict[str, np.ndarray]:
    """
    Compute full-horizon DKF cooperative states for each client.

    The returned state trajectory matches local_learner1's convention:
      X_dkf[:, t] = posterior state after processing y_t.
    """
    dkf_states: dict[str, np.ndarray] = {}
    m_count = int(data.num_components)
    t_total = int(data.total_time)

    for m in range(m_count):
        cid = f"{m + 1}"
        comp = data.local_learners_pack[f"comp_{cid}"]

        a_mat = np.asarray(comp["A"], dtype=float)
        b_mat = np.asarray(comp["B"], dtype=float)
        c_mat = np.asarray(comp["C"], dtype=float)
        q_mat = np.asarray(comp["Q"], dtype=float)
        r_mat = np.asarray(comp["R"], dtype=float)
        p0_mat = np.asarray(comp["P0"], dtype=float)
        x0_vec = np.asarray(comp["x0"], dtype=float).reshape(-1, 1)
        y_full = np.asarray(comp["Y_full"], dtype=float)
        u_full = np.asarray(comp["U_full"], dtype=float)
        s_dim = int(b_mat.shape[1])
        p_dim = int(a_mat.shape[0])

        kf = KalmanFilter(a_mat, b_mat, c_mat, q_mat, r_mat, p0_mat.copy(), x0_vec.copy())
        x_dkf = np.zeros((p_dim, t_total), dtype=float)

        for t in range(t_total):
            u_prev = u_full[:, t - 1:t] if t > 0 else np.zeros((s_dim, 1))
            kf.predict(u_prev)
            kf.update(y_full[:, t:t + 1])
            x_dkf[:, t:t + 1] = kf.get_state()

        dkf_states[cid] = x_dkf

    return dkf_states


def _evaluate_fed_server_model(
    data: RetrieveData,
    dkf_states: dict[str, np.ndarray],
    a_offdiag: dict[str, np.ndarray],
    b_offdiag: dict[str, np.ndarray],
    valid_t0: int,
    valid_t1: int,
) -> dict[str, np.ndarray | float]:
    """
    Evaluate one-step output prediction error for the learned server dynamics.

    This follows the server state equation used in global_learner1.py:
      h_s_t^m = A_mm h_c_{t-1}^m + B_mm u_{t-1}^m
                + sum_{n!=m} (A_mn h_c_{t-1}^n + B_mn u_{t-1}^n)

    and predicts y_t^m by:
      y_hat_t^m = C_mm h_s_t^m
    """
    m_count = int(data.num_components)
    n_steps = valid_t1 - valid_t0
    d_vec = _to_1d_int(data.output_size_vec)

    comp_data = {f"{m + 1}": data.local_learners_pack[f"comp_{m + 1}"] for m in range(m_count)}

    # Pre-cache local blocks and streams for speed/readability.
    a_mm = {}
    b_mm = {}
    c_mm = {}
    x0_blocks = {}
    y_blocks = {}
    u_blocks = {}
    s_dims = {}
    for m in range(m_count):
        cid = f"{m + 1}"
        comp = comp_data[cid]
        a_mm[cid] = np.asarray(comp["A"], dtype=float)
        b_mm[cid] = np.asarray(comp["B"], dtype=float)
        c_mm[cid] = np.asarray(comp["C"], dtype=float)
        x0_blocks[cid] = np.asarray(comp["x0"], dtype=float).reshape(-1, 1)
        y_blocks[cid] = np.asarray(comp["Y_full"], dtype=float)
        u_blocks[cid] = np.asarray(comp["U_full"], dtype=float)
        s_dims[cid] = int(b_mm[cid].shape[1])

    sqerr_sum = np.zeros(m_count, dtype=float)
    sqerr_time = np.zeros((m_count, n_steps), dtype=float)
    total_sqerr = 0.0

    for k, t in enumerate(range(valid_t0, valid_t1)):
        for m in range(m_count):
            cid_m = f"{m + 1}"

            # Previous cooperative state h_c_{t-1}; for t=0 fallback to x0.
            if t > 0:
                h_c_prev_m = dkf_states[cid_m][:, t - 1:t]
                u_prev_m = u_blocks[cid_m][:, t - 1:t]
            else:
                h_c_prev_m = x0_blocks[cid_m]
                u_prev_m = np.zeros((s_dims[cid_m], 1))

            h_s_t = a_mm[cid_m] @ h_c_prev_m + b_mm[cid_m] @ u_prev_m

            # Add off-diagonal peer contributions.
            for n in range(m_count):
                if n == m:
                    continue
                cid_n = f"{n + 1}"
                key = f"{m + 1}{n + 1}"
                if t > 0:
                    h_c_prev_n = dkf_states[cid_n][:, t - 1:t]
                    u_prev_n = u_blocks[cid_n][:, t - 1:t]
                else:
                    h_c_prev_n = x0_blocks[cid_n]
                    u_prev_n = np.zeros((s_dims[cid_n], 1))

                if key in a_offdiag:
                    h_s_t += np.asarray(a_offdiag[key], dtype=float) @ h_c_prev_n
                if key in b_offdiag:
                    h_s_t += np.asarray(b_offdiag[key], dtype=float) @ u_prev_n

            y_pred = c_mm[cid_m] @ h_s_t
            y_t = y_blocks[cid_m][:, t:t + 1]
            err = y_t - y_pred
            sq = float(np.sum(err ** 2))

            sqerr_sum[m] += sq
            sqerr_time[m, k] = sq
            total_sqerr += sq

    rmse = np.sqrt(sqerr_sum / float(n_steps))
    rmse_per_dim = np.sqrt((sqerr_sum / d_vec.astype(float)) / float(n_steps))
    rmse_overall = float(np.sqrt(total_sqerr / float(n_steps)))

    return {
        "rmse": rmse,
        "rmse_per_dim": rmse_per_dim,
        "rmse_overall": rmse_overall,
        "sqerr_time": sqerr_time,
        "sqerr_sum": sqerr_sum,
        "sqerr_total": total_sqerr,
    }


def _load_run_theta_phi(
    monte_root: Path,
    run_idx: int,
    round_hint: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load run-specific theta/phi from the snapshot in mc_<run>/snapshots.

    Preference:
      1) round_<round_hint>.pkl if provided and present
      2) latest available round_*.pkl
    """
    snap_dir = monte_root / f"mc_{run_idx:03d}" / "snapshots"
    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found for run {run_idx}: {snap_dir}")

    snap_path: Path | None = None
    if round_hint is not None:
        candidate = snap_dir / f"round_{int(round_hint):04d}.pkl"
        if candidate.exists():
            snap_path = candidate

    if snap_path is None:
        snapshots = sorted(snap_dir.glob("round_*.pkl"))
        if not snapshots:
            raise FileNotFoundError(f"No snapshot files found in {snap_dir}")
        snap_path = snapshots[-1]

    with snap_path.open("rb") as handle:
        payload = pickle.load(handle)

    if "theta" not in payload or "phi" not in payload:
        raise KeyError(f"Snapshot missing theta/phi: {snap_path}")

    theta = {k: np.asarray(v, dtype=float) for k, v in payload["theta"].items()}
    phi = {}
    for k, v in payload["phi"].items():
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2 and arr.shape[1] != 1:
            arr = arr.reshape(arr.shape[0], 1)
        phi[k] = arr

    return theta, phi


def _evaluate_local_model_prediction(
    data: RetrieveData,
    dkf_states: dict[str, np.ndarray],
    theta_by_client: dict[str, np.ndarray],
    phi_by_client: dict[str, np.ndarray],
    valid_t0: int,
    valid_t1: int,
    warm_start: bool,
) -> dict[str, np.ndarray | float]:
    """
    Evaluate one-step prediction error using local model dynamics in local_learner1:
      h_pred_t = A_mm h_est_{t-1} + B_mm u_{t-1} + phi_m
      y_hat_t  = C_mm h_pred_t
      h_est_t  = h_c_t + theta_m y_t
    """
    m_count = int(data.num_components)
    n_steps = valid_t1 - valid_t0
    d_vec = _to_1d_int(data.output_size_vec)

    sqerr_sum = np.zeros(m_count, dtype=float)
    sqerr_time = np.zeros((m_count, n_steps), dtype=float)
    total_sqerr = 0.0

    for m in range(m_count):
        cid = f"{m + 1}"
        comp = data.local_learners_pack[f"comp_{cid}"]

        a_mat = np.asarray(comp["A"], dtype=float)
        b_mat = np.asarray(comp["B"], dtype=float)
        c_mat = np.asarray(comp["C"], dtype=float)
        y_full = np.asarray(comp["Y_full"], dtype=float)
        u_full = np.asarray(comp["U_full"], dtype=float)
        x0_vec = np.asarray(comp["x0"], dtype=float).reshape(-1, 1)
        s_dim = int(b_mat.shape[1])

        theta = np.asarray(theta_by_client[cid], dtype=float)
        phi = np.asarray(phi_by_client[cid], dtype=float).reshape(-1, 1)

        # warm-start=True replicates the full local recursion before scoring.
        start_t = 0 if warm_start else valid_t0
        prev_est = x0_vec.copy()
        step_idx = 0

        for t in range(start_t, valid_t1):
            u_prev = u_full[:, t - 1:t] if t > 0 else np.zeros((s_dim, 1))
            y_t = y_full[:, t:t + 1]

            h_pred_t = a_mat @ prev_est + b_mat @ u_prev + phi
            y_pred_t = c_mat @ h_pred_t

            if t >= valid_t0:
                err = y_t - y_pred_t
                sq = float(np.sum(err ** 2))
                sqerr_sum[m] += sq
                sqerr_time[m, step_idx] = sq
                total_sqerr += sq
                step_idx += 1

            h_coop_t = dkf_states[cid][:, t:t + 1]
            h_est_t = h_coop_t + theta @ y_t
            prev_est = h_est_t

        if step_idx != n_steps:
            raise RuntimeError(
                f"Local-model indexing mismatch at client {cid}: step_idx={step_idx}, n_steps={n_steps}"
            )

    rmse = np.sqrt(sqerr_sum / float(n_steps))
    rmse_per_dim = np.sqrt((sqerr_sum / d_vec.astype(float)) / float(n_steps))
    rmse_overall = float(np.sqrt(total_sqerr / float(n_steps)))

    return {
        "rmse": rmse,
        "rmse_per_dim": rmse_per_dim,
        "rmse_overall": rmse_overall,
        "sqerr_time": sqerr_time,
        "sqerr_sum": sqerr_sum,
        "sqerr_total": total_sqerr,
    }


def _reconstruct_full_from_blocks(
    m_count: int,
    p_vec: np.ndarray,
    s_vec: np.ndarray,
    p_starts: np.ndarray,
    s_starts: np.ndarray,
    a_diag_blocks: dict[str, np.ndarray],
    b_diag_blocks: dict[str, np.ndarray],
    a_offdiag: dict[str, np.ndarray],
    b_offdiag: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct full-system (A_hat, B_hat) from diagonal true blocks and
    run-specific off-diagonal estimated blocks.

    This exactly matches the intended evaluation setting:
    - diagonal blocks are fixed/known (not learned)
    - off-diagonal blocks come from each Monte Carlo run estimate
    """
    p_total = int(np.sum(p_vec))
    s_total = int(np.sum(s_vec))
    a_hat = np.zeros((p_total, p_total), dtype=float)
    b_hat = np.zeros((p_total, s_total), dtype=float)

    # Diagonal blocks from true model
    for i in range(m_count):
        key = f"{i + 1}{i + 1}"
        r0 = int(p_starts[i])
        r1 = r0 + int(p_vec[i])
        c0 = int(p_starts[i])
        c1 = c0 + int(p_vec[i])
        u0 = int(s_starts[i])
        u1 = u0 + int(s_vec[i])
        a_hat[r0:r1, c0:c1] = np.asarray(a_diag_blocks[key], dtype=float)
        b_hat[r0:r1, u0:u1] = np.asarray(b_diag_blocks[key], dtype=float)

    # Off-diagonal blocks from one run estimate.
    # If any block key is missing, the corresponding block stays zero.
    for i in range(m_count):
        r0 = int(p_starts[i])
        r1 = r0 + int(p_vec[i])
        for j in range(m_count):
            if i == j:
                continue
            key = f"{i + 1}{j + 1}"
            c0 = int(p_starts[j])
            c1 = c0 + int(p_vec[j])
            u0 = int(s_starts[j])
            u1 = u0 + int(s_vec[j])
            if key in a_offdiag:
                a_hat[r0:r1, c0:c1] = np.asarray(a_offdiag[key], dtype=float)
            if key in b_offdiag:
                b_hat[r0:r1, u0:u1] = np.asarray(b_offdiag[key], dtype=float)

    return a_hat, b_hat


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Simple centered moving average used only for trace visualization."""
    if w <= 1:
        return x.copy()
    kernel = np.ones(int(w), dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")


def _payload_or_data_vector(
    payload: dict,
    payload_key: str,
    data_vec: np.ndarray,
    vec_name: str,
) -> np.ndarray:
    """
    Return payload vector if available; otherwise use data vector.
    Also validate shape/value consistency when both are available.
    """
    if payload_key not in payload:
        return data_vec.copy()

    payload_vec = _to_1d_int(payload[payload_key])
    if payload_vec.shape != data_vec.shape:
        raise ValueError(
            f"Mismatch in {vec_name} shape between payload and data: "
            f"{payload_vec.shape} vs {data_vec.shape}."
        )
    if not np.array_equal(payload_vec, data_vec):
        raise ValueError(
            f"Mismatch in {vec_name} values between payload and current data/config. "
            "This suggests the loaded data is not the same setup as the training artifact."
        )
    return payload_vec


def _plot_rmse_summary(
    out_dir: Path,
    client_labels: list[str],
    fed_server_mean: np.ndarray,
    fed_server_std: np.ndarray,
    local_model_mean: np.ndarray,
    local_model_std: np.ndarray,
    dkf: np.ndarray,
    ckf: np.ndarray,
) -> None:
    """Create bar plot: Fed-server/local-model mean±std vs DKF vs CKF."""
    x = np.arange(len(client_labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - 1.5 * width, dkf, width=width, label="DKF", color=colorblind_colors[1])
    ax.bar(x - 0.5 * width, ckf, width=width, label="CKF", color=colorblind_colors[4])
    ax.bar(
        x + 0.5 * width,
        fed_server_mean,
        width=width,
        yerr=fed_server_std,
        capsize=4,
        label="Fed-server (mean$\pm$std)",
        color=colorblind_colors[3],
    )
    ax.bar(
        x + 1.5 * width,
        local_model_mean,
        width=width,
        yerr=local_model_std,
        capsize=4,
        label="Local-model (mean$\pm$std)",
        color=colorblind_colors[2],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(client_labels, fontsize=20)
    ax.set_ylabel("One-step RMSE", fontsize=25)
    ax.set_xlabel("Client", fontsize=25)
    ax.set_title("Validation RMSE by Client", fontsize=25)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=20)
    fig.tight_layout()
    fig.savefig(
        out_dir / "plots" / "rmse_comparison_per_client.pdf",
        format="pdf",
        dpi=800,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_sqerr_traces(
    out_dir: Path,
    valid_t0: int,
    fed_server_sqerr_runs: np.ndarray,
    local_model_sqerr_runs: np.ndarray,
    dkf_sqerr: np.ndarray,
    ckf_sqerr: np.ndarray,
    smooth_window: int,
) -> None:
    """
    Plot per-client validation squared-error traces over time.

    Curves shown:
      - Fed-server mean sq-error across Monte Carlo runs
      - Fed-server ±1 std band
      - Local-model mean sq-error across Monte Carlo runs
      - Local-model ±1 std band
      - DKF sq-error
      - CKF sq-error
    """
    m_count = fed_server_sqerr_runs.shape[1]
    n_steps = fed_server_sqerr_runs.shape[2]
    t_axis = np.arange(valid_t0, valid_t0 + n_steps)

    for m in range(m_count):
        fed_mean = np.mean(fed_server_sqerr_runs[:, m, :], axis=0)
        fed_std = np.std(fed_server_sqerr_runs[:, m, :], axis=0)
        local_mean = np.mean(local_model_sqerr_runs[:, m, :], axis=0)
        local_std = np.std(local_model_sqerr_runs[:, m, :], axis=0)
        dkf = dkf_sqerr[m, :]
        ckf = ckf_sqerr[m, :]

        fed_mean_s = _moving_average(fed_mean, smooth_window)
        fed_up_s = _moving_average(fed_mean + fed_std, smooth_window)
        fed_lo_s = _moving_average(np.maximum(fed_mean - fed_std, 0.0), smooth_window)
        local_mean_s = _moving_average(local_mean, smooth_window)
        local_up_s = _moving_average(local_mean + local_std, smooth_window)
        local_lo_s = _moving_average(np.maximum(local_mean - local_std, 0.0), smooth_window)
        dkf_s = _moving_average(dkf, smooth_window)
        ckf_s = _moving_average(ckf, smooth_window)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t_axis, fed_mean_s, label="Fed-server mean sq-error", color=colorblind_colors[3])
        ax.fill_between(
            t_axis,
            fed_lo_s,
            fed_up_s,
            color=colorblind_colors[3],
            alpha=0.25,
            label="Fed-server $\pm$1 std",
        )
        ax.plot(t_axis, local_mean_s, label="Local-model mean sq-error", color=colorblind_colors[2])
        ax.fill_between(
            t_axis,
            local_lo_s,
            local_up_s,
            color=colorblind_colors[2],
            alpha=0.20,
            label="Local-model $\pm$1 std",
        )
        ax.plot(t_axis, dkf_s, label="DKF sq-error", color=colorblind_colors[1])
        ax.plot(t_axis, ckf_s, label="CKF sq-error", color=colorblind_colors[4])
        ax.set_xlabel("Time index ($t$)", fontsize=25)
        ax.set_ylabel("Squared error norm", fontsize=25)
        ax.set_title(f"Validation Squared Error (Client {m + 1})", fontsize=25)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=20)
        fig.tight_layout()
        fig.savefig(
            out_dir / "plots" / f"sqerr_trace_client_{m + 1}.pdf",
            format="pdf",
            dpi=800,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Load data/config and resolve files/directories
    # ------------------------------------------------------------------
    data = RetrieveData(CONFIG_PATH)
    monte_path, out_dir = _resolve_paths(data)
    valid_t0, valid_t1 = _resolve_validation_window(data)
    n_steps = valid_t1 - valid_t0

    # Monte Carlo artifact produced by main1.py contains per-run final off-diagonal
    # estimates and some structural metadata (diag blocks, starts, etc.).
    with monte_path.open("rb") as handle:
        monte_payload = pickle.load(handle)

    # Required run-specific model parameters.
    final_a_runs = monte_payload.get("final_A_mn", None)
    final_b_runs = monte_payload.get("final_B_mn", None)
    if final_a_runs is None or final_b_runs is None:
        raise KeyError("monte_carlo_history.pkl missing required keys: final_A_mn and/or final_B_mn")
    if len(final_a_runs) != len(final_b_runs):
        raise ValueError("Mismatch: len(final_A_mn) != len(final_B_mn)")
    if len(final_a_runs) == 0:
        raise ValueError("No Monte Carlo runs found in monte_carlo_history.pkl")

    # Build full global trajectories from local Y_full/U_full streams.
    y_full, u_full, d_vec, s_vec_data, d_starts, s_starts_data = _build_global_streams_from_local_packs(data)

    # State/input structure metadata.
    p_vec_data = _to_1d_int(data.comp_size_vec)
    p_starts_data = _to_1d_int(data.comp_start_vec)
    m_count = int(data.num_components)

    # Prefer structure vectors from payload when present, but enforce that they
    # match the currently loaded data/config.
    p_vec = _payload_or_data_vector(monte_payload, "p_vec", p_vec_data, "p_vec")
    s_vec = _payload_or_data_vector(monte_payload, "s_vec", s_vec_data, "s_vec")
    p_starts = _payload_or_data_vector(monte_payload, "comp_start_vec", p_starts_data, "comp_start_vec")
    s_starts = _payload_or_data_vector(monte_payload, "input_start_vec", s_starts_data, "input_start_vec")

    # True complete model from CKF pack (used for baselines and C/Q/R/P0/x0).
    ckf = data.CKF_pack
    c_full = np.asarray(ckf["C_complete"], dtype=float)
    q_full = np.asarray(ckf["Q"], dtype=float)
    r_full = np.asarray(ckf["R"], dtype=float)
    p0_full = np.asarray(ckf["P0"], dtype=float)
    x0_full = np.asarray(ckf["x0"], dtype=float).reshape(-1, 1)

    rounds_completed = monte_payload.get("rounds_completed", [])
    if rounds_completed and len(rounds_completed) != len(final_a_runs):
        raise ValueError(
            "Mismatch: len(rounds_completed) does not match number of Monte Carlo runs."
        )

    # Cooperative DKF states used by both server-model and local-model prediction paths.
    dkf_states_full = _compute_dkf_states_full(data)

    # ------------------------------------------------------------------
    # 2) Evaluate fixed baselines once (same for all MC runs)
    # ------------------------------------------------------------------
    dkf_metrics = _evaluate_dkf_local(data, valid_t0, valid_t1, WARM_START)
    ckf_metrics = _kalman_predict_eval(
        a_mat=np.asarray(ckf["A_complete"], dtype=float),
        b_mat=np.asarray(ckf["B_complete"], dtype=float),
        c_mat=c_full,
        q_mat=q_full,
        r_mat=r_full,
        p0_mat=p0_full,
        x0_vec=x0_full,
        y_full=y_full,
        u_full=u_full,
        valid_t0=valid_t0,
        valid_t1=valid_t1,
        d_vec=d_vec,
        d_starts=d_starts,
        warm_start=WARM_START,
    )

    # ------------------------------------------------------------------
    # 3) Evaluate run-specific learned models
    # ------------------------------------------------------------------
    run_rows: list[dict[str, float | int]] = []
    fed_server_rmse_runs = []
    fed_server_sqerr_time_runs = []
    fed_server_rmse_overall_runs = []
    local_model_rmse_runs = []
    local_model_sqerr_time_runs = []
    local_model_rmse_overall_runs = []

    monte_root = monte_path.parent

    for run_idx, (a_off, b_off) in enumerate(zip(final_a_runs, final_b_runs), start=1):
        round_hint = int(rounds_completed[run_idx - 1]) if rounds_completed else None
        theta_run, phi_run = _load_run_theta_phi(
            monte_root=monte_root,
            run_idx=run_idx,
            round_hint=round_hint,
        )

        # Ensure local params are available for every client.
        expected_clients = {f"{m + 1}" for m in range(m_count)}
        if not expected_clients.issubset(theta_run.keys()):
            raise KeyError(
                f"Run {run_idx}: missing theta keys for clients {sorted(expected_clients - set(theta_run.keys()))}"
            )
        if not expected_clients.issubset(phi_run.keys()):
            raise KeyError(
                f"Run {run_idx}: missing phi keys for clients {sorted(expected_clients - set(phi_run.keys()))}"
            )

        fed_server_metrics = _evaluate_fed_server_model(
            data=data,
            dkf_states=dkf_states_full,
            a_offdiag=a_off,
            b_offdiag=b_off,
            valid_t0=valid_t0,
            valid_t1=valid_t1,
        )

        local_model_metrics = _evaluate_local_model_prediction(
            data=data,
            dkf_states=dkf_states_full,
            theta_by_client=theta_run,
            phi_by_client=phi_run,
            valid_t0=valid_t0,
            valid_t1=valid_t1,
            warm_start=WARM_START,
        )

        fed_server_rmse = np.asarray(fed_server_metrics["rmse"], dtype=float)
        fed_server_rmse_per_dim = np.asarray(fed_server_metrics["rmse_per_dim"], dtype=float)
        fed_server_sqerr_time = np.asarray(fed_server_metrics["sqerr_time"], dtype=float)
        fed_server_rmse_overall = float(fed_server_metrics["rmse_overall"])

        local_model_rmse = np.asarray(local_model_metrics["rmse"], dtype=float)
        local_model_rmse_per_dim = np.asarray(local_model_metrics["rmse_per_dim"], dtype=float)
        local_model_sqerr_time = np.asarray(local_model_metrics["sqerr_time"], dtype=float)
        local_model_rmse_overall = float(local_model_metrics["rmse_overall"])

        fed_server_rmse_runs.append(fed_server_rmse)
        fed_server_sqerr_time_runs.append(fed_server_sqerr_time)
        fed_server_rmse_overall_runs.append(fed_server_rmse_overall)

        local_model_rmse_runs.append(local_model_rmse)
        local_model_sqerr_time_runs.append(local_model_sqerr_time)
        local_model_rmse_overall_runs.append(local_model_rmse_overall)

        # One output row per run for easy downstream stats/boxplots.
        row: dict[str, float | int] = {
            "run_idx": run_idx,
            "valid_t0": valid_t0,
            "valid_t1": valid_t1,
            "n_steps": n_steps,
            "fed_server_rmse_overall": fed_server_rmse_overall,
            "local_model_rmse_overall": local_model_rmse_overall,
        }
        for m in range(m_count):
            cid = f"c{m + 1}"
            row[f"fed_server_rmse_{cid}"] = float(fed_server_rmse[m])
            row[f"fed_server_rmse_per_dim_{cid}"] = float(fed_server_rmse_per_dim[m])
            row[f"local_model_rmse_{cid}"] = float(local_model_rmse[m])
            row[f"local_model_rmse_per_dim_{cid}"] = float(local_model_rmse_per_dim[m])
            row[f"dkf_rmse_{cid}"] = float(dkf_metrics["rmse"][m])
            row[f"ckf_rmse_{cid}"] = float(ckf_metrics["rmse"][m])
        run_rows.append(row)

    # ------------------------------------------------------------------
    # 4) Aggregate run-specific metrics across Monte Carlo runs
    # ------------------------------------------------------------------
    fed_server_rmse_runs_arr = np.asarray(fed_server_rmse_runs, dtype=float)  # (runs, M)
    fed_server_sqerr_runs_arr = np.asarray(fed_server_sqerr_time_runs, dtype=float)  # (runs, M, n_steps)
    fed_server_rmse_overall_arr = np.asarray(fed_server_rmse_overall_runs, dtype=float)

    local_model_rmse_runs_arr = np.asarray(local_model_rmse_runs, dtype=float)  # (runs, M)
    local_model_sqerr_runs_arr = np.asarray(local_model_sqerr_time_runs, dtype=float)  # (runs, M, n_steps)
    local_model_rmse_overall_arr = np.asarray(local_model_rmse_overall_runs, dtype=float)

    fed_server_mean = np.mean(fed_server_rmse_runs_arr, axis=0)
    fed_server_std = np.std(fed_server_rmse_runs_arr, axis=0)
    fed_server_overall_mean = float(np.mean(fed_server_rmse_overall_arr))
    fed_server_overall_std = float(np.std(fed_server_rmse_overall_arr))

    local_model_mean = np.mean(local_model_rmse_runs_arr, axis=0)
    local_model_std = np.std(local_model_rmse_runs_arr, axis=0)
    local_model_overall_mean = float(np.mean(local_model_rmse_overall_arr))
    local_model_overall_std = float(np.std(local_model_rmse_overall_arr))

    dkf_rmse = np.asarray(dkf_metrics["rmse"], dtype=float)
    ckf_rmse = np.asarray(ckf_metrics["rmse"], dtype=float)
    dkf_overall = float(dkf_metrics["rmse_overall"])
    ckf_overall = float(ckf_metrics["rmse_overall"])

    # ------------------------------------------------------------------
    # 5) Save CSV outputs
    # ------------------------------------------------------------------
    # CSV 1: one row per MC run
    run_df = pd.DataFrame(run_rows)
    run_csv = out_dir / "validation_run_metrics.csv"
    run_df.to_csv(run_csv, index=False)

    # CSV 2: aggregated summary per client + one overall row
    summary_rows: list[dict[str, float | str | int]] = []
    for m in range(m_count):
        dkf_val = float(dkf_rmse[m])
        ckf_val = float(ckf_rmse[m])
        row = {
            "client": f"c{m + 1}",
            "n_runs": int(fed_server_rmse_runs_arr.shape[0]),
            "n_steps": int(n_steps),
            "valid_t0": int(valid_t0),
            "valid_t1": int(valid_t1),
            "fed_server_mean": float(fed_server_mean[m]),
            "fed_server_std": float(fed_server_std[m]),
            "local_model_mean": float(local_model_mean[m]),
            "local_model_std": float(local_model_std[m]),
            "dkf": dkf_val,
            "ckf": ckf_val,
            "gap_fed_server_minus_dkf": float(fed_server_mean[m] - dkf_val),
            "gap_fed_server_minus_ckf": float(fed_server_mean[m] - ckf_val),
            "gap_local_model_minus_dkf": float(local_model_mean[m] - dkf_val),
            "gap_local_model_minus_ckf": float(local_model_mean[m] - ckf_val),
            "pct_gap_fed_server_vs_dkf": 100.0 * float(fed_server_mean[m] - dkf_val) / max(1e-12, dkf_val),
            "pct_gap_fed_server_vs_ckf": 100.0 * float(fed_server_mean[m] - ckf_val) / max(1e-12, ckf_val),
            "pct_gap_local_model_vs_dkf": 100.0 * float(local_model_mean[m] - dkf_val) / max(1e-12, dkf_val),
            "pct_gap_local_model_vs_ckf": 100.0 * float(local_model_mean[m] - ckf_val) / max(1e-12, ckf_val),
        }
        summary_rows.append(row)

    summary_rows.append(
        {
            "client": "overall",
            "n_runs": int(fed_server_rmse_runs_arr.shape[0]),
            "n_steps": int(n_steps),
            "valid_t0": int(valid_t0),
            "valid_t1": int(valid_t1),
            "fed_server_mean": fed_server_overall_mean,
            "fed_server_std": fed_server_overall_std,
            "local_model_mean": local_model_overall_mean,
            "local_model_std": local_model_overall_std,
            "dkf": dkf_overall,
            "ckf": ckf_overall,
            "gap_fed_server_minus_dkf": fed_server_overall_mean - dkf_overall,
            "gap_fed_server_minus_ckf": fed_server_overall_mean - ckf_overall,
            "gap_local_model_minus_dkf": local_model_overall_mean - dkf_overall,
            "gap_local_model_minus_ckf": local_model_overall_mean - ckf_overall,
            "pct_gap_fed_server_vs_dkf": 100.0 * (fed_server_overall_mean - dkf_overall) / max(1e-12, dkf_overall),
            "pct_gap_fed_server_vs_ckf": 100.0 * (fed_server_overall_mean - ckf_overall) / max(1e-12, ckf_overall),
            "pct_gap_local_model_vs_dkf": 100.0 * (local_model_overall_mean - dkf_overall) / max(1e-12, dkf_overall),
            "pct_gap_local_model_vs_ckf": 100.0 * (local_model_overall_mean - ckf_overall) / max(1e-12, ckf_overall),
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "validation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # CSV 3 (extra): compact time-series for plotting/debugging.
    # Stores fed-server/local-model mean/std sq-error traces + DKF/CKF traces.
    time_rows = []
    fed_server_sqerr_mean = np.mean(fed_server_sqerr_runs_arr, axis=0)  # (M, n_steps)
    fed_server_sqerr_std = np.std(fed_server_sqerr_runs_arr, axis=0)    # (M, n_steps)
    local_model_sqerr_mean = np.mean(local_model_sqerr_runs_arr, axis=0)  # (M, n_steps)
    local_model_sqerr_std = np.std(local_model_sqerr_runs_arr, axis=0)    # (M, n_steps)
    dkf_sqerr = np.asarray(dkf_metrics["sqerr_time"], dtype=float)
    ckf_sqerr = np.asarray(ckf_metrics["sqerr_time"], dtype=float)
    for m in range(m_count):
        for k in range(n_steps):
            t_abs = valid_t0 + k
            time_rows.append(
                {
                    "t": int(t_abs),
                    "client": f"c{m + 1}",
                    "fed_server_sqerr_mean": float(fed_server_sqerr_mean[m, k]),
                    "fed_server_sqerr_std": float(fed_server_sqerr_std[m, k]),
                    "local_model_sqerr_mean": float(local_model_sqerr_mean[m, k]),
                    "local_model_sqerr_std": float(local_model_sqerr_std[m, k]),
                    "dkf_sqerr": float(dkf_sqerr[m, k]),
                    "ckf_sqerr": float(ckf_sqerr[m, k]),
                }
            )
    time_df = pd.DataFrame(time_rows)
    time_csv = out_dir / "validation_time_series.csv"
    time_df.to_csv(time_csv, index=False)

    # ------------------------------------------------------------------
    # 6) Save plots
    # ------------------------------------------------------------------
    client_labels = [f"c{m + 1}" for m in range(m_count)]
    _plot_rmse_summary(
        out_dir,
        client_labels,
        fed_server_mean,
        fed_server_std,
        local_model_mean,
        local_model_std,
        dkf_rmse,
        ckf_rmse,
    )
    _plot_sqerr_traces(
        out_dir=out_dir,
        valid_t0=valid_t0,
        fed_server_sqerr_runs=fed_server_sqerr_runs_arr,
        local_model_sqerr_runs=local_model_sqerr_runs_arr,
        dkf_sqerr=dkf_sqerr,
        ckf_sqerr=ckf_sqerr,
        smooth_window=int(max(1, SMOOTH_WINDOW)),
    )

    # Console summary for quick sanity check.
    print("Validation completed.")
    print(f"Window: [{valid_t0}, {valid_t1}) with {n_steps} steps")
    print(f"Runs evaluated: {fed_server_rmse_runs_arr.shape[0]}")
    print(f"Run metrics: {run_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Time-series: {time_csv}")
    print(f"Plots dir: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
