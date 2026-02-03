from __future__ import annotations  # future annotations for consistency

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from data_retriever import RetrieveData
from local_learner1 import LocalModel
from counterfactual_exp1 import (
    piecewise_constant_controls,
    apply_intervention,
    split_by_client,
)
from counterfactual_exp import simulate_true_lti

# -----------------------------------------------------------------------------
# Configuration knobs
# -----------------------------------------------------------------------------
CONFIG_PATH = "config.ini"
RESULTS_SUBDIR = "main1"
TRAINING_RESULTS_DIR = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/norm_comp_2/Components_2/rebuttal/main1/mc_001"  # override path to trained results (if set, bypass results_location)
SNAPSHOTS_DIR = None  # override path to snapshots (if set, bypass <results>/snapshots)
COUNTERFACTUAL_OUTPUT_DIR = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/norm_comp_2/Components_2/rebuttal/count_exp_results/state_exp"  # override path to save metrics/plots (if set, bypass <results>/counterfactual)
T_TEST = 300  # length of the synthetic test trajectory
INTERVENED_CLIENT = 1
INTERVENTION_TIME = 100
DELTA = 1
CONTROL_SCALE = 1.0
PLOT_CLIENT = 1
SEED = None
REUSE_NOISE = True
MONTE_CARLO = 50
SAVE_ARTIFACTS = False  # set True to write CSVs for the first MC run


def load_trained_snapshot(snapshots_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    snapshot_files = sorted(snapshots_dir.glob("round_*.pkl"))
    if not snapshot_files:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    latest = snapshot_files[-1]
    with latest.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["A_mn"], payload["B_mn"], payload["theta"], payload["phi"]


def build_local_models(data: RetrieveData, Y_blocks: Dict[str, np.ndarray], U_blocks: Dict[str, np.ndarray], theta_trained: Dict[str, np.ndarray], phi_trained: Dict[str, np.ndarray], T: int) -> Tuple[Dict[str, LocalModel], Dict[str, np.ndarray]]:
    local_models: Dict[str, LocalModel] = {}
    dkf_states: Dict[str, np.ndarray] = {}
    for cid in Y_blocks.keys():
        comp_key = f"comp_{cid}"
        comp_data = data.local_learners_pack[comp_key].copy()
        comp_data["Y"] = Y_blocks[cid]
        comp_data["U"] = U_blocks[cid]
        comp_data["theta_init"] = theta_trained[cid]
        comp_data["phi_init"] = phi_trained[cid]
        model = LocalModel(T, comp_data)
        model.theta = theta_trained[cid].copy()
        model.phi = phi_trained[cid].copy()
        payload = model.local_forward_pass()
        local_models[cid] = model
        # DKF latent states (proprietary)
        dkf_states[cid] = model.X_dkf.copy()
    return local_models, dkf_states


def rollout_server_states(
    A_mm: Dict[str, np.ndarray],
    B_mm: Dict[str, np.ndarray],
    A_mn: Dict[str, np.ndarray],
    B_mn: Dict[str, np.ndarray],
    x_dkf: Dict[str, np.ndarray],
    U_blocks: Dict[str, np.ndarray],
    p_vec: np.ndarray,
    T: int,
    intervention_time: int,
) -> Dict[str, np.ndarray]:
    """Deterministic rollout of server states h_s_t using learned couplings and DKF states."""
    X_server = {cid: np.zeros((int(p_vec[int(cid) - 1]), T)) for cid in x_dkf.keys()}
    component_ids = sorted(x_dkf.keys(), key=lambda x: int(x))
    for t in range(1, T):
        for cid in component_ids:
            m = int(cid)
            key_diag = f"{m}{m}"
            h_c_prev = x_dkf[cid][:, t - 1:t]
            u_prev_m = U_blocks[cid][:, t - 1:t]
            h_s_t = A_mm[key_diag] @ h_c_prev + B_mm[key_diag] @ u_prev_m
            for peer in component_ids:
                if peer == cid:
                    continue
                n = int(peer)
                key_off = f"{m}{n}"
                h_c_peer = x_dkf[peer][:, t - 1:t]
                u_peer = U_blocks[peer][:, t - 1:t]
                if key_off in A_mn:
                    h_s_t += A_mn[key_off] @ h_c_peer
                if key_off in B_mn:
                    h_s_t += B_mn[key_off] @ u_peer
            X_server[cid][:, t:t + 1] = h_s_t
    return X_server


def mse_dict(pred: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> Dict[str, float]:
    errors = {}
    for cid in pred.keys():
        diff = pred[cid] - truth[cid]
        errors[cid] = float(np.mean(np.sum(diff * diff, axis=0)))
    return errors


def per_dim_rmse(mse_dict_in: Dict[str, float], p_vec: np.ndarray) -> Dict[str, float]:
    rmse = {}
    for cid, mse_val in mse_dict_in.items():
        p = float(p_vec[int(cid) - 1])
        rmse[cid] = float(np.sqrt(mse_val / max(p, 1.0)))
    return rmse


def main() -> None:
    rng = np.random.default_rng(SEED)

    def run_once(seed_override: int | None = None, save: bool = False, run_id: int = 1) -> dict:
        rng_local = np.random.default_rng(seed_override if seed_override is not None else SEED)
        data = RetrieveData(CONFIG_PATH)
        results_dir = Path(TRAINING_RESULTS_DIR) if TRAINING_RESULTS_DIR else Path(data.results_location) / RESULTS_SUBDIR
        snapshots_dir = Path(SNAPSHOTS_DIR) if SNAPSHOTS_DIR else results_dir / "snapshots"
        counterfactual_dir = Path(COUNTERFACTUAL_OUTPUT_DIR) if COUNTERFACTUAL_OUTPUT_DIR else results_dir / "counterfactual_server"
        A_mn_trained, B_mn_trained, theta_trained, phi_trained = load_trained_snapshot(snapshots_dir)
        ckf = data.CKF_pack
        A_true = ckf["A_complete"]
        B_true = ckf["B_complete"]
        C_true = ckf["C_complete"]
        Q_true = ckf["Q"]
        R_true = ckf["R"]
        x0_true = ckf["x0"]
        p_vec = data.comp_size_vec[:, 0]
        d_vec = data.output_size_vec[:, 0]
        s_vec = data.input_size_vec[:, 0]
        p_starts = data.comp_start_vec[:, 0]
        d_starts = data.output_start_vec[:, 0]
        s_starts = data.input_start_vec[:, 0]

        # Controls
        U_base_full = piecewise_constant_controls(s_vec, T_TEST, CONTROL_SCALE, rng_local)
        U_cf_full = apply_intervention(U_base_full, s_vec, INTERVENED_CLIENT, INTERVENTION_TIME, DELTA)
        U_blocks_base = split_by_client(U_base_full, s_starts, s_vec)
        U_blocks_cf = split_by_client(U_cf_full, s_starts, s_vec)

        # True simulation (oracle) for comparison
        X_base_true, Y_base_true, w_noise, v_noise = simulate_true_lti(A_true, B_true, C_true, Q_true, R_true, x0_true, U_base_full, rng_local)
        if REUSE_NOISE:
            X_cf_true, Y_cf_true, _, _ = simulate_true_lti(A_true, B_true, C_true, Q_true, R_true, x0_true, U_cf_full, rng_local, w_noise=w_noise, v_noise=v_noise)
        else:
            X_cf_true, Y_cf_true, _, _ = simulate_true_lti(A_true, B_true, C_true, Q_true, R_true, x0_true, U_cf_full, rng_local)

        X_true_blocks_base = split_by_client(X_base_true, p_starts, p_vec)
        X_true_blocks_cf = split_by_client(X_cf_true, p_starts, p_vec)
        Y_blocks_base = split_by_client(Y_base_true, d_starts, d_vec)

        # Local models to obtain DKF states
        local_models, dkf_states_base = build_local_models(data, Y_blocks_base, U_blocks_base, theta_trained, phi_trained, T_TEST)
        # For CF controls, DKF states remain proprietary; only controls change for server rollout

        # Server rollout with learned couplings
        A_mm = data.global_learners_pack["A_mm"]
        B_mm = data.global_learners_pack["B_mm"]
        X_server_base = rollout_server_states(A_mm, B_mm, A_mn_trained, B_mn_trained, dkf_states_base, U_blocks_base, p_vec, T_TEST, INTERVENTION_TIME)
        X_server_cf = rollout_server_states(A_mm, B_mm, A_mn_trained, B_mn_trained, dkf_states_base, U_blocks_cf, p_vec, T_TEST, INTERVENTION_TIME)

        # Metrics at t_eval
        t_eval = INTERVENTION_TIME + 1
        delta_x_cf = {
            cid: float(np.linalg.norm(X_server_cf[cid][:, t_eval:t_eval + 1] - X_true_blocks_cf[cid][:, t_eval:t_eval + 1]))
            for cid in X_server_cf.keys()
        }
        delta_x_base = {
            cid: float(np.linalg.norm(X_server_base[cid][:, t_eval:t_eval + 1] - X_true_blocks_base[cid][:, t_eval:t_eval + 1]))
            for cid in X_server_base.keys()
        }
        delta_x_diff = {cid: abs(delta_x_cf[cid] - delta_x_base[cid]) for cid in delta_x_cf.keys()}
        delta_x_cf_norm = {cid: delta_x_cf[cid] / max(1.0, float(p_vec[int(cid) - 1])) for cid in delta_x_cf.keys()}
        delta_x_base_norm = {cid: delta_x_base[cid] / max(1.0, float(p_vec[int(cid) - 1])) for cid in delta_x_base.keys()}
        delta_x_diff_norm = {cid: delta_x_diff[cid] / max(1.0, float(p_vec[int(cid) - 1])) for cid in delta_x_diff.keys()}
        delta_x_pct_err = {cid: (delta_x_diff[cid] / max(1e-9, abs(delta_x_base[cid]))) * 100.0 for cid in delta_x_diff.keys()}
        # Per-dim numerator only (avoid canceling by dividing both numerator and denominator)
        delta_x_pct_err_norm = {cid: (delta_x_diff_norm[cid] / max(1e-9, abs(delta_x_base[cid]))) * 100.0 for cid in delta_x_diff_norm.keys()}
        rel_err_x_cf_t = {
            cid: float(
                np.linalg.norm(X_server_cf[cid][:, t_eval:t_eval + 1] - X_true_blocks_cf[cid][:, t_eval:t_eval + 1])
                / max(1e-9, np.linalg.norm(X_true_blocks_cf[cid][:, t_eval:t_eval + 1]))
            )
            for cid in X_server_cf.keys()
        }
        percent_err_x_cf_t = {cid: rel_err_x_cf_t[cid] * 100.0 for cid in rel_err_x_cf_t.keys()}
        sym_rel_err_x_cf_t = {
            cid: float(
                np.linalg.norm(X_server_cf[cid][:, t_eval:t_eval + 1] - X_true_blocks_cf[cid][:, t_eval:t_eval + 1])
                / max(
                    1e-9,
                    np.linalg.norm(X_server_cf[cid][:, t_eval:t_eval + 1]) + np.linalg.norm(X_true_blocks_cf[cid][:, t_eval:t_eval + 1]),
                )
            )
            for cid in X_server_cf.keys()
        }
        norm_x_cf_t = {cid: float(np.linalg.norm(X_server_cf[cid][:, t_eval:t_eval + 1])) for cid in X_server_cf.keys()}
        norm_x_true_t = {cid: float(np.linalg.norm(X_true_blocks_cf[cid][:, t_eval:t_eval + 1])) for cid in X_true_blocks_cf.keys()}

        if save:
            counterfactual_dir.mkdir(parents=True, exist_ok=True)
            # Save server states (pred vs true) and controls
            np.savetxt(counterfactual_dir / f"run_{run_id}_X_server_base.csv", np.column_stack([np.arange(T_TEST), np.vstack([X_server_base[cid] for cid in sorted(X_server_base.keys(), key=lambda x: int(x))]).T]), delimiter=",")
            np.savetxt(counterfactual_dir / f"run_{run_id}_X_server_cf.csv", np.column_stack([np.arange(T_TEST), np.vstack([X_server_cf[cid] for cid in sorted(X_server_cf.keys(), key=lambda x: int(x))]).T]), delimiter=",")
            np.savetxt(counterfactual_dir / f"run_{run_id}_X_true_base.csv", np.column_stack([np.arange(T_TEST), X_base_true.T]), delimiter=",")
            np.savetxt(counterfactual_dir / f"run_{run_id}_X_true_cf.csv", np.column_stack([np.arange(T_TEST), X_cf_true.T]), delimiter=",")
            np.savetxt(counterfactual_dir / f"run_{run_id}_U_base.csv", np.column_stack([np.arange(T_TEST), U_base_full.T]), delimiter=",")
            np.savetxt(counterfactual_dir / f"run_{run_id}_U_cf.csv", np.column_stack([np.arange(T_TEST), U_cf_full.T]), delimiter=",")

        return {
            "delta_x_cf": delta_x_cf,
            "delta_x_base": delta_x_base,
            "delta_x_diff": delta_x_diff,
            "delta_x_cf_norm": delta_x_cf_norm,
            "delta_x_base_norm": delta_x_base_norm,
            "delta_x_diff_norm": delta_x_diff_norm,
            "delta_x_pct_err": delta_x_pct_err,
            "delta_x_pct_err_norm": delta_x_pct_err_norm,
            "rel_err_x_cf_t": rel_err_x_cf_t,
            "percent_err_x_cf_t": percent_err_x_cf_t,
            "sym_rel_err_x_cf_t": sym_rel_err_x_cf_t,
            "norm_x_cf_t": norm_x_cf_t,
            "norm_x_true_t": norm_x_true_t,
        }

    runs = []
    for mc_idx in range(MONTE_CARLO):
        seed_override = None if SEED is None else SEED + mc_idx
        print(f"Starting Monte Carlo run {mc_idx + 1}/{MONTE_CARLO} ...")
        runs.append(run_once(seed_override=seed_override, save=(SAVE_ARTIFACTS and mc_idx == 0), run_id=mc_idx + 1))

    def average_dict(key: str) -> dict:
        cids = runs[0][key].keys()
        return {cid: float(np.mean([r[key][cid] for r in runs])) for cid in cids}

    avg_delta_x_cf = average_dict("delta_x_cf")
    avg_delta_x_base = average_dict("delta_x_base")
    avg_delta_x_diff = average_dict("delta_x_diff")
    avg_delta_x_cf_norm = average_dict("delta_x_cf_norm")
    avg_delta_x_base_norm = average_dict("delta_x_base_norm")
    avg_delta_x_diff_norm = average_dict("delta_x_diff_norm")
    avg_delta_x_pct_err = average_dict("delta_x_pct_err")
    avg_delta_x_pct_err_norm = average_dict("delta_x_pct_err_norm")
    avg_rel_err_x_cf_t = average_dict("rel_err_x_cf_t")
    avg_percent_err_x_cf_t = average_dict("percent_err_x_cf_t")
    avg_sym_rel_err_x_cf_t = average_dict("sym_rel_err_x_cf_t")
    avg_norm_x_cf_t = average_dict("norm_x_cf_t")
    avg_norm_x_true_t = average_dict("norm_x_true_t")

    print(f"Monte Carlo runs: {MONTE_CARLO}")
    print("Avg Delta_x_cf at t+1:", avg_delta_x_cf)
    print("Avg Delta_x_base at t+1:", avg_delta_x_base)
    print("Avg Abs(delta_x_cf - delta_x_base) at t+1:", avg_delta_x_diff)
    print("Avg Delta_x_cf at t+1 (per-dim):", avg_delta_x_cf_norm)
    print("Avg Delta_x_base at t+1 (per-dim):", avg_delta_x_base_norm)
    print("Avg Abs(delta_x_cf - delta_x_base) at t+1 (per-dim):", avg_delta_x_diff_norm)
    print("Avg Percent error (effect gap / baseline) at t+1 (%):", avg_delta_x_pct_err)
    print("Avg Percent error (effect gap per-dim / baseline) at t+1 (%):", avg_delta_x_pct_err_norm)
    print("Avg Relative error (CF vs true state) at t+1:", avg_rel_err_x_cf_t)
    print("Avg Percent error (CF vs true state) at t+1 (%):", avg_percent_err_x_cf_t)
    print("Avg Federated symmetric relative error x at t+1:", avg_sym_rel_err_x_cf_t)
    print("Avg ||x_cf_fed|| at t+1:", avg_norm_x_cf_t)
    print("Avg ||x_cf_true|| at t+1:", avg_norm_x_true_t)


if __name__ == "__main__":
    main()
