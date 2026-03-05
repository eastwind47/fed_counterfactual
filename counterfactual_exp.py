"""Counterfactual experiment for the v1 pipeline with detailed inline comments."""
# Standard library imports with purpose comments
from __future__ import annotations  # future annotations for consistency with v1 code
import pickle  # load saved snapshots from training
from pathlib import Path  # filesystem paths that work across OSes
from typing import Dict, Tuple  # type hints for clarity

# Third-party imports
import numpy as np  # numerical routines
import matplotlib.pyplot as plt  # plotting for the visualization step

# Project imports from the v1 stack
from data_retriever import RetrieveData  # loads config + data packs
from local_learner1 import LocalModel  # client-side model (v1)

# -----------------------------------------------------------------------------
# Configuration knobs for this experiment (put all file locations here)
# -----------------------------------------------------------------------------
CONFIG_PATH = "config.ini"  # configuration file to reuse the training setup
RESULTS_SUBDIR = "main1"  # default subfolder under results_location for v1 artifacts
TRAINING_RESULTS_DIR = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/uai2026/exp-results/fedcount/dissimilar-dkf/main1/mc_003"  # override path to trained results (if set, bypass results_location)
SNAPSHOTS_DIR = None  # override path to snapshots (if set, bypass <results>/snapshots)
COUNTERFACTUAL_OUTPUT_DIR = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/uai2026/exp-results/fedcount/dissimilar-dkf/main1/mc_003/count_exp_results"  # override path to save metrics/plots (if set, bypass <results>/counterfactual)
T_TEST = 300  # length of the synthetic test trajectory
INTERVENED_CLIENT = 1  # client id (1-based) that receives the intervention
INTERVENTION_TIME = 100  # single timestep for the intervention
DELTA = 0.1  # step magnitude added to the intervened client input
CONTROL_SCALE = 1  # scale of the zero-mean training-like controls
PLOT_CLIENT = 1  # which client to visualize in the trajectory plot
SEED = None  # RNG seed; set to an int for reproducibility, None for fresh draws each run
REUSE_NOISE = True  # if True, reuse the same noise realization for baseline and CF
MONTE_CARLO = 50  # number of repeated runs for averaging


def load_trained_snapshot(snapshots_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load the latest snapshot produced by main1 (A_mn, B_mn, theta, phi)."""
    snapshot_files = sorted(snapshots_dir.glob("round_*.pkl"))  # list available snapshots
    if not snapshot_files:  # bail out if training artifacts are missing
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    latest = snapshot_files[-1]  # pick the newest snapshot by filename ordering
    with latest.open("rb") as handle:  # open snapshot on disk
        payload = pickle.load(handle)  # load python dict with parameters
    return payload["A_mn"], payload["B_mn"], payload["theta"], payload["phi"]  # unpack learned params


def piecewise_constant_controls(s_vec: np.ndarray, T: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Generate zero-mean controls resampled independently at every timestep."""
    s_total = int(np.sum(s_vec))  # total input dimension across clients
    return rng.normal(0.0, scale, size=(s_total, T))  # independent draws each step


def apply_intervention(U: np.ndarray, s_vec: np.ndarray, client_idx: int, t_idx: int, delta: float) -> np.ndarray:
    """Add a step delta to a single client's input at a single timestep."""
    U_cf = U.copy()  # start from the baseline control
    start_offsets = np.cumsum(np.concatenate(([0], s_vec[:-1])))  # compute per-client input starts
    u_start = start_offsets[client_idx - 1]  # starting row for this client's input
    u_end = u_start + s_vec[client_idx - 1]  # ending row (exclusive)
    U_cf[u_start:u_end, t_idx] += delta  # add delta at the chosen timestep
    return U_cf  # return intervened control


def simulate_true_lti(A: np.ndarray, B: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, x0: np.ndarray, U: np.ndarray, rng: np.random.Generator, w_noise: np.ndarray | None = None, v_noise: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the ground-truth LTI; optionally reuse provided noise draws."""
    T = U.shape[1]  # horizon length
    p = A.shape[0]  # state dimension
    d = C.shape[0]  # output dimension
    X = np.zeros((p, T))  # latent states over time
    Y = np.zeros((d, T))  # observations over time
    x = x0.reshape(p, 1)  # current state column vector
    if w_noise is None:
        w_noise = rng.multivariate_normal(mean=np.zeros(p), cov=Q, size=T).T  # process noise
    if v_noise is None:
        v_noise = rng.multivariate_normal(mean=np.zeros(d), cov=R, size=T).T  # measurement noise
    for t in range(T):  # loop over timesteps
        u_t = U[:, t:t + 1]  # control at time t
        w_t = w_noise[:, t:t + 1]  # process noise column
        x = A @ x + B @ u_t + w_t  # state update
        v_t = v_noise[:, t:t + 1]  # measurement noise column
        y_t = C @ x + v_t  # observation
        X[:, t:t + 1] = x  # store state
        Y[:, t:t + 1] = y_t  # store observation
    return X, Y, w_noise, v_noise  # return latent/observed trajectories and noise used


def split_by_client(mat: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> Dict[str, np.ndarray]:
    """Slice a global matrix into per-client blocks along rows."""
    blocks = {}  # container for slices
    for idx, (start, width) in enumerate(zip(starts, sizes)):  # iterate over clients
        cid = f"{idx + 1}"  # 1-based id
        start_int = int(start)  # ensure python int
        width_int = int(width)  # ensure python int
        blocks[cid] = mat[start_int:start_int + width_int, :]  # take rows for this client
    return blocks  # return dict keyed by client id


def build_local_models(data: RetrieveData, Y_blocks: Dict[str, np.ndarray], U_blocks: Dict[str, np.ndarray], theta_trained: Dict[str, np.ndarray], phi_trained: Dict[str, np.ndarray], T: int) -> Dict[str, LocalModel]:
    """Create LocalModel instances seeded with trained theta/phi and new streams."""
    locals_dict: Dict[str, LocalModel] = {}  # holder for instantiated models
    for cid in Y_blocks.keys():  # iterate over clients
        comp_key = f"comp_{cid}"  # key used in data packs
        comp_data = data.local_learners_pack[comp_key].copy()  # shallow copy base pack
        comp_data["Y"] = Y_blocks[cid]  # plug in new observations
        comp_data["U"] = U_blocks[cid]  # plug in new controls
        comp_data["theta_init"] = theta_trained[cid]  # load trained theta
        comp_data["phi_init"] = phi_trained[cid]  # load trained phi
        model = LocalModel(T, comp_data)  # create local learner
        model.theta = theta_trained[cid].copy()  # ensure trained theta is set
        model.phi = phi_trained[cid].copy()  # ensure trained phi is set
        locals_dict[cid] = model  # register model
    return locals_dict  # return all local learners


def build_C_mm_from_locals(data: RetrieveData) -> Dict[str, np.ndarray]:
    """Assemble per-client C_mm directly from local_learners_pack."""
    C_mm: Dict[str, np.ndarray] = {}  # container for observation matrices
    for m in range(data.num_components):  # loop over components
        cid = f"{m + 1}"  # 1-based id
        comp_key = f"comp_{cid}"  # key in the pack
        diag_key = f"{cid}{cid}"  # align with A_mm/B_mm keys
        C_mm[diag_key] = data.local_learners_pack[comp_key]["C"]  # grab C from local pack
    return C_mm  # return assembled dict


def abduction_pass(local_models: Dict[str, LocalModel]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run local forward passes to obtain abducted states and model baseline preds."""
    abducted = {}  # store h_aug_est per client
    h_pred = {}  # store h_aug_pred per client
    y_pred = {}  # store model-predicted measurements per client
    for cid, model in local_models.items():  # iterate over clients
        payload = model.local_forward_pass()  # run DKF + augmented forward pass
        abducted[cid] = payload["h_aug_est"]  # cache augmented estimate
        h_pred[cid] = payload["h_aug_pred"]  # model predicted latent
        y_pred[cid] = model.C @ payload["h_aug_pred"]  # predicted measurement
    return abducted, h_pred, y_pred  # return per-client latent trajectories and preds


def rollout_counterfactual(A_mm: Dict[str, np.ndarray], B_mm: Dict[str, np.ndarray], A_mn: Dict[str, np.ndarray], B_mn: Dict[str, np.ndarray], C_mm: Dict[str, np.ndarray], phi: Dict[str, np.ndarray], init_states: Dict[str, np.ndarray], U_blocks: Dict[str, np.ndarray], delta_u_blocks: Dict[str, np.ndarray], p_vec: np.ndarray, s_vec: np.ndarray, d_vec: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Roll the learned coupled system forward under intervened controls."""
    component_ids = sorted(U_blocks.keys(), key=lambda x: int(x))  # stable id order
    T = next(iter(U_blocks.values())).shape[1]  # horizon inferred from controls
    X_hat = {cid: np.zeros((p_vec[int(cid) - 1], T)) for cid in component_ids}  # latent buffer
    Y_hat = {cid: np.zeros((d_vec[int(cid) - 1], T)) for cid in component_ids}  # obs buffer
    x_prev = {cid: init_states[cid][:, 0:1] for cid in component_ids}  # initial states at t=0
    for t in range(T):  # loop over time
        x_next = {}  # temporary holder for next states
        for cid in component_ids:  # per client update
            u_self = U_blocks[cid][:, t - 1:t] if t > 0 else np.zeros((s_vec[int(cid) - 1], 1))  # own control
            key_diag = f"{cid}{cid}"  # diagonal key
            # Adjust phi by intervention-induced delta_u from peers, but only at the timestep immediately after intervention
            phi_adjusted = phi[cid].copy()
            if t == INTERVENTION_TIME + 1:
                for peer in component_ids:
                    if peer == cid:
                        continue
                    key_off = f"{cid}{peer}"
                    if key_off in B_mn:
                        delta_u_peer = delta_u_blocks[peer][:, t - 1:t] if t > 0 else np.zeros((s_vec[int(peer) - 1], 1))
                        phi_adjusted += B_mn[key_off] @ delta_u_peer
                    # if 100 < t < 180:
                    #     print(f'key_off: {key_off}', B_mn[key_off], delta_u_peer)
            x_t = A_mm[key_diag] @ x_prev[cid] + B_mm[key_diag] @ u_self + phi_adjusted  # local dynamics with phi shift
            x_next[cid] = x_t  # store next state
            X_hat[cid][:, t:t + 1] = x_t  # record latent trajectory
            Y_hat[cid][:, t:t + 1] = C_mm[key_diag] @ x_t  # record predicted observation
        x_prev = x_next  # advance states for next step
    return X_hat, Y_hat  # return latent and observation predictions


def rollout_proprietary(A_mm: Dict[str, np.ndarray], B_mm: Dict[str, np.ndarray], C_mm: Dict[str, np.ndarray], phi: Dict[str, np.ndarray], init_states: Dict[str, np.ndarray], U_blocks: Dict[str, np.ndarray], delta_u_blocks: Dict[str, np.ndarray], B_mn: Dict[str, np.ndarray], p_vec: np.ndarray, s_vec: np.ndarray, d_vec: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Roll each client independently ignoring off-diagonal couplings."""
    component_ids = sorted(U_blocks.keys(), key=lambda x: int(x))  # stable order
    T = next(iter(U_blocks.values())).shape[1]  # horizon
    X_hat = {cid: np.zeros((p_vec[int(cid) - 1], T)) for cid in component_ids}  # latent buffer
    Y_hat = {cid: np.zeros((d_vec[int(cid) - 1], T)) for cid in component_ids}  # obs buffer
    x_prev = {cid: init_states[cid][:, 0:1] for cid in component_ids}  # initial states
    for t in range(T):  # time loop
        x_next = {}  # next-state holder
        for cid in component_ids:  # per client
            u_self = U_blocks[cid][:, t - 1:t] if t > 0 else np.zeros((s_vec[int(cid) - 1], 1))  # own control
            key_diag = f"{cid}{cid}"  # diagonal key
            phi_adjusted = phi[cid].copy()
            if t == INTERVENTION_TIME + 1:
                for peer in component_ids:
                    if peer == cid:
                        continue
                    key_off = f"{cid}{peer}"
                    if key_off in B_mn:
                        delta_u_peer = delta_u_blocks[peer][:, t - 1:t] if t > 0 else np.zeros((s_vec[int(peer) - 1], 1))
                        phi_adjusted += B_mn[key_off] @ delta_u_peer
            x_t = A_mm[key_diag] @ x_prev[cid] + B_mm[key_diag] @ u_self + phi_adjusted  # local dynamics only
            x_next[cid] = x_t  # store
            X_hat[cid][:, t:t + 1] = x_t  # record latent
            Y_hat[cid][:, t:t + 1] = C_mm[key_diag] @ x_t  # record observation
        x_prev = x_next  # step forward
    return X_hat, Y_hat  # return trajectories


def mse_dict(pred: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute MSE per client between predicted and true trajectories."""
    errors = {}  # store per-client MSE
    for cid in pred.keys():  # iterate over clients
        diff = pred[cid] - truth[cid]  # difference matrix
        errors[cid] = float(np.mean(np.sum(diff * diff, axis=0)))  # mean per-timestep squared norm
    return errors  # return MSE dict


def effect_mse_dict(pred: Dict[str, np.ndarray], truth: Dict[str, np.ndarray], base: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute effect MSE per client: Δy_pred vs Δy_true where Δy = y_cf - y_base."""
    errors = {}
    for cid in pred.keys():
        delta_pred = pred[cid] - base[cid]
        delta_true = truth[cid] - base[cid]
        diff = delta_pred - delta_true
        errors[cid] = float(np.mean(np.sum(diff * diff, axis=0)))
    return errors


def effect_mse_windowed(pred: Dict[str, np.ndarray], truth: Dict[str, np.ndarray], base: Dict[str, np.ndarray], t0: int, t1: int) -> Dict[str, float]:
    """Effect MSE restricted to a window [t0, t1] (inclusive)."""
    errors = {}
    for cid in pred.keys():
        delta_pred = pred[cid][:, t0:t1 + 1] - base[cid][:, t0:t1 + 1]
        delta_true = truth[cid][:, t0:t1 + 1] - base[cid][:, t0:t1 + 1]
        diff = delta_pred - delta_true
        errors[cid] = float(np.mean(np.sum(diff * diff, axis=0)))
    return errors


def per_dim_rmse(mse_dict_in: Dict[str, float], d_vec: np.ndarray) -> Dict[str, float]:
    """Convert summed MSE to per-dimension RMSE using client output dims."""
    rmse = {}
    for cid, mse_val in mse_dict_in.items():
        d = float(d_vec[int(cid) - 1])
        rmse[cid] = float(np.sqrt(mse_val / max(d, 1.0)))
    return rmse


def nrmse_effect(pred: Dict[str, np.ndarray], truth: Dict[str, np.ndarray], base: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Normalized RMSE on the effect: RMSE(Δy_pred-Δy_true) / mean||Δy_true||."""
    scores = {}
    for cid in pred.keys():
        delta_pred = pred[cid] - base[cid]
        delta_true = truth[cid] - base[cid]
        diff = delta_pred - delta_true
        rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=0))))
        denom = float(np.mean(np.linalg.norm(delta_true, axis=0)))
        scores[cid] = rmse / denom if denom > 1e-9 else float("nan")
    return scores


def relative_error_dict(pred: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute ||pred - truth|| / ||truth|| per client over the full horizon."""
    rel = {}
    for cid in pred.keys():
        diff = pred[cid] - truth[cid]
        num = float(np.linalg.norm(diff))
        denom = float(np.linalg.norm(truth[cid]))
        rel[cid] = num / denom if denom > 1e-9 else float("nan")
    return rel


def save_plot(Y_base: Dict[str, np.ndarray], Y_true_cf: Dict[str, np.ndarray], Y_fed_cf: Dict[str, np.ndarray], Y_prop_cf: Dict[str, np.ndarray] | None, client_id: int, out_path: Path) -> None:
    """Plot baseline vs counterfactual trajectories for a single client."""
    cid = f"{client_id}"  # string id
    plt.figure(figsize=(10, 4))  # set figure size
    plt.plot(Y_base[cid].T, label="Baseline y_m")  # baseline observation
    plt.plot(Y_true_cf[cid].T, label="Oracle CF")  # ground-truth counterfactual
    plt.plot(Y_fed_cf[cid].T, label="Federated CF")  # federated prediction
    if Y_prop_cf is not None:
        plt.plot(Y_prop_cf[cid].T, label="Proprietary CF")  # proprietary prediction
    plt.xlabel("Time")  # x-axis label
    plt.ylabel("Observation")  # y-axis label
    plt.title(f"Client {client_id} counterfactual response")  # plot title
    plt.legend()  # add legend
    plt.tight_layout()  # tidy layout
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    plt.savefig(out_path)  # save plot to disk
    plt.close()  # close figure to free memory


def save_arrays_to_csv(prefix: str, arrays: Dict[str, np.ndarray], out_dir: Path, t0: int, t1: int) -> None:
    """Save a dict of arrays to CSV (windowed) with time as first column."""
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    t0_int = int(t0)
    t1_int = int(t1)
    time_vec = np.arange(t0_int, t1_int + 1)  # time indices for window
    for cid, arr in arrays.items():  # loop over clients
        arr_slice = arr[:, t0_int:t1_int + 1]  # slice to window
        out_mat = np.column_stack([time_vec, arr_slice.T])  # shape (window_len, 1 + dim)
        filename = out_dir / f"{prefix}_client_{cid}.csv"  # build filename
        np.savetxt(filename, out_mat, delimiter=",")  # write array to CSV


def save_controls_to_csv(prefix: str, U_blocks: Dict[str, np.ndarray], out_dir: Path) -> None:
    """Save full control trajectories (all timesteps) with time as first column."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # assume all clients have the same T
    T = next(iter(U_blocks.values())).shape[1]
    time_vec = np.arange(T)
    for cid, arr in U_blocks.items():
        out_mat = np.column_stack([time_vec, arr.T])  # (T, 1 + s_m)
        filename = out_dir / f"{prefix}_client_{cid}.csv"
        np.savetxt(filename, out_mat, delimiter=",")


def main() -> None:
    """Run the single-client intervention experiment end-to-end."""
    def run_once(seed_override: int | None = None, save: bool = False, run_id: int = 1) -> dict:
        rng_local = np.random.default_rng(seed_override if seed_override is not None else SEED)
        # Load config and data
        data_local = RetrieveData(CONFIG_PATH)
        results_dir_local = Path(TRAINING_RESULTS_DIR) if TRAINING_RESULTS_DIR else Path(data_local.results_location) / RESULTS_SUBDIR
        snapshots_dir_local = Path(SNAPSHOTS_DIR) if SNAPSHOTS_DIR else results_dir_local / "snapshots"
        counterfactual_dir = Path(COUNTERFACTUAL_OUTPUT_DIR) if COUNTERFACTUAL_OUTPUT_DIR else results_dir_local / "counterfactual"
        A_mn_trained, B_mn_trained, theta_trained, phi_trained = load_trained_snapshot(snapshots_dir_local)
        ckf_local = data_local.CKF_pack
        A_true = ckf_local["A_complete"]
        B_true = ckf_local["B_complete"]
        C_true = ckf_local["C_complete"]
        Q_true = ckf_local["Q"]
        R_true = ckf_local["R"]
        x0_true = ckf_local["x0"]
        p_vec = data_local.comp_size_vec[:, 0]
        d_vec = data_local.output_size_vec[:, 0]
        s_vec = data_local.input_size_vec[:, 0]
        p_starts = data_local.comp_start_vec[:, 0]
        d_starts = data_local.output_start_vec[:, 0]
        s_starts = data_local.input_start_vec[:, 0]
        U_base = piecewise_constant_controls(s_vec, T_TEST, CONTROL_SCALE, rng_local)
        U_cf = apply_intervention(U_base, s_vec, INTERVENED_CLIENT, INTERVENTION_TIME, DELTA)
        X_base, Y_base, w_noise, v_noise = simulate_true_lti(A_true, B_true, C_true, Q_true, R_true, x0_true, U_base, rng_local)
        if REUSE_NOISE:
            _, Y_cf_true, _, _ = simulate_true_lti(A_true, B_true, C_true, Q_true, R_true, x0_true, U_cf, rng_local, w_noise=w_noise, v_noise=v_noise)
        else:
            _, Y_cf_true, _, _ = simulate_true_lti(A_true, B_true, C_true, Q_true, R_true, x0_true, U_cf, rng_local)
        Y_blocks_base = split_by_client(Y_base, d_starts, d_vec)
        Y_blocks_true_cf = split_by_client(Y_cf_true, d_starts, d_vec)
        X_blocks_true_cf = split_by_client(X_base * 0 + X_base, p_starts, p_vec)  # placeholder not used now
        U_blocks_base = split_by_client(U_base, s_starts, s_vec)
        U_blocks_cf = split_by_client(U_cf, s_starts, s_vec)
        local_models = build_local_models(data_local, Y_blocks_base, U_blocks_base, theta_trained, phi_trained, T_TEST)
        abducted_states, h_pred_base, y_pred_base = abduction_pass(local_models)
        C_mm = build_C_mm_from_locals(data_local)
        B_true_full = ckf_local["B_complete"]
        B_true_offdiag = {}
        for i in range(data_local.num_components):
            for j in range(data_local.num_components):
                if i == j:
                    continue
                row_start = int(p_starts[i])
                row_end = row_start + int(p_vec[i])
                col_start = int(s_starts[j])
                col_end = col_start + int(s_vec[j])
                key = f"{i+1}{j+1}"
                B_true_offdiag[key] = B_true_full[row_start:row_end, col_start:col_end]
        A_mm = data_local.global_learners_pack["A_mm"]
        B_mm = data_local.global_learners_pack["B_mm"]
        delta_u_blocks = {cid: U_blocks_cf[cid] - U_blocks_base[cid] for cid in U_blocks_cf.keys()}
        X_cf_fed, Y_cf_fed = rollout_counterfactual(A_mm, B_mm, A_mn_trained, B_true_offdiag, C_mm, phi_trained, abducted_states, U_blocks_cf, delta_u_blocks, p_vec, s_vec, d_vec)
        mse_h_fed = mse_dict(X_cf_fed, X_blocks_true_cf)
        mse_y_fed = mse_dict(Y_cf_fed, Y_blocks_true_cf)
        effect_mse_fed = effect_mse_dict(Y_cf_fed, Y_blocks_true_cf, Y_blocks_base)
        effect_mse_fed_win = effect_mse_windowed(Y_cf_fed, Y_blocks_true_cf, Y_blocks_base, INTERVENTION_TIME, INTERVENTION_TIME)
        rmse_y_fed = per_dim_rmse(mse_y_fed, d_vec)
        nrmse_eff_fed = nrmse_effect(Y_cf_fed, Y_blocks_true_cf, Y_blocks_base)
        t_eval = INTERVENTION_TIME + 1
        delta_y_cf = {
            cid: float(np.linalg.norm(Y_cf_fed[cid][:, t_eval:t_eval + 1] - Y_blocks_true_cf[cid][:, t_eval:t_eval + 1]))
            for cid in Y_cf_fed.keys()
        }
        delta_y_base = {
            cid: float(np.linalg.norm(y_pred_base[cid][:, t_eval:t_eval + 1] - Y_blocks_base[cid][:, t_eval:t_eval + 1]))
            for cid in y_pred_base.keys()
        }
        delta_y_diff = {cid: abs(delta_y_cf[cid] - delta_y_base[cid]) for cid in delta_y_cf.keys()}
        delta_y_cf_norm = {cid: delta_y_cf[cid] / max(1.0, float(d_vec[int(cid) - 1])) for cid in delta_y_cf.keys()}
        delta_y_base_norm = {cid: delta_y_base[cid] / max(1.0, float(d_vec[int(cid) - 1])) for cid in delta_y_base.keys()}
        delta_y_diff_norm = {cid: delta_y_diff[cid] / max(1.0, float(d_vec[int(cid) - 1])) for cid in delta_y_diff.keys()}
        delta_y_pct_err = {cid: (delta_y_diff[cid] / max(1e-9, abs(delta_y_base[cid]))) * 100.0 for cid in delta_y_diff.keys()}
        delta_y_pct_err_norm = {cid: (delta_y_diff_norm[cid] / max(1e-9, abs(delta_y_base_norm[cid]))) * 100.0 for cid in delta_y_diff_norm.keys()}
        rel_err_y_fed_t = {
            cid: float(
                np.linalg.norm(Y_cf_fed[cid][:, t_eval:t_eval + 1] - Y_blocks_true_cf[cid][:, t_eval:t_eval + 1])
                / max(1e-9, np.linalg.norm(Y_blocks_true_cf[cid][:, t_eval:t_eval + 1]))
            )
            for cid in Y_cf_fed.keys()
        }
        sym_rel_err_y_fed_t = {
            cid: float(
                np.linalg.norm(Y_cf_fed[cid][:, t_eval:t_eval + 1] - Y_blocks_true_cf[cid][:, t_eval:t_eval + 1])
                / max(
                    1e-9,
                    np.linalg.norm(Y_cf_fed[cid][:, t_eval:t_eval + 1]) + np.linalg.norm(Y_blocks_true_cf[cid][:, t_eval:t_eval + 1]),
                )
            )
            for cid in Y_cf_fed.keys()
        }
        if save:
            counterfactual_dir.mkdir(parents=True, exist_ok=True)
            save_arrays_to_csv(f"run_{run_id}_Y_base", Y_blocks_base, counterfactual_dir, 0, T_TEST - 1)
            save_arrays_to_csv(f"run_{run_id}_Y_cf_true", Y_blocks_true_cf, counterfactual_dir, 0, T_TEST - 1)
            save_arrays_to_csv(f"run_{run_id}_Y_cf_fed", Y_cf_fed, counterfactual_dir, 0, T_TEST - 1)
            save_controls_to_csv(f"run_{run_id}_U_base", U_blocks_base, counterfactual_dir)
            save_controls_to_csv(f"run_{run_id}_U_cf", U_blocks_cf, counterfactual_dir)
            save_plot(Y_blocks_base, Y_blocks_true_cf, Y_cf_fed, None, PLOT_CLIENT, counterfactual_dir / f"run_{run_id}_client_{PLOT_CLIENT}.png")

        return {
            "mse_h_fed": mse_h_fed,
            "mse_y_fed": mse_y_fed,
            "effect_mse_fed": effect_mse_fed,
            "effect_mse_fed_win": effect_mse_fed_win,
            "rmse_y_fed": rmse_y_fed,
            "nrmse_effect_fed": nrmse_eff_fed,
            "delta_y_cf": delta_y_cf,
            "delta_y_base": delta_y_base,
            "delta_y_diff": delta_y_diff,
            "delta_y_cf_norm": delta_y_cf_norm,
            "delta_y_base_norm": delta_y_base_norm,
            "delta_y_diff_norm": delta_y_diff_norm,
            "delta_y_pct_err": delta_y_pct_err,
            "delta_y_pct_err_norm": delta_y_pct_err_norm,
            "rel_err_y_fed_t": rel_err_y_fed_t,
            "sym_rel_err_y_fed_t": sym_rel_err_y_fed_t,
        }

    # Monte Carlo aggregation
    runs = []
    for mc_idx in range(MONTE_CARLO):
        seed_override = None if SEED is None else SEED + mc_idx
        print(f"Starting Monte Carlo run {mc_idx + 1}/{MONTE_CARLO} ...")
        runs.append(run_once(seed_override=seed_override, save=(mc_idx == 0), run_id=mc_idx + 1))

    def average_dict(key: str) -> dict:
        cids = runs[0][key].keys()
        return {cid: float(np.mean([r[key][cid] for r in runs])) for cid in cids}

    avg_mse_h_fed = average_dict("mse_h_fed")
    avg_mse_y_fed = average_dict("mse_y_fed")
    avg_rel_err_y_fed_t = average_dict("rel_err_y_fed_t")
    avg_effect_mse_fed = average_dict("effect_mse_fed")
    avg_effect_mse_fed_win = average_dict("effect_mse_fed_win")
    avg_rmse_y_fed = average_dict("rmse_y_fed")
    avg_nrmse_eff_fed = average_dict("nrmse_effect_fed")
    avg_delta_y_cf = average_dict("delta_y_cf")
    avg_delta_y_base = average_dict("delta_y_base")
    avg_delta_y_diff = average_dict("delta_y_diff")
    avg_delta_y_cf_norm = average_dict("delta_y_cf_norm")
    avg_delta_y_base_norm = average_dict("delta_y_base_norm")
    avg_delta_y_diff_norm = average_dict("delta_y_diff_norm")
    avg_delta_y_pct_err = average_dict("delta_y_pct_err")
    avg_delta_y_pct_err_norm = average_dict("delta_y_pct_err_norm")
    avg_rel_err_y_fed_t = average_dict("rel_err_y_fed_t")
    avg_sym_rel_err_y_fed_t = average_dict("sym_rel_err_y_fed_t")

    # Print averages
    print(f"Monte Carlo runs: {MONTE_CARLO}")
    print("Avg Federated MSE_h:", avg_mse_h_fed)
    print("Avg Federated MSE_y:", avg_mse_y_fed)
    print("Avg Federated relative error y at t+1:", avg_rel_err_y_fed_t)
    print("Avg Federated RMSE_y (per-dim):", avg_rmse_y_fed)
    print("Avg Federated effect MSE_y:", avg_effect_mse_fed)
    print("Avg Federated effect MSE_y (window):", avg_effect_mse_fed_win)
    print("Avg Federated NRMSE effect:", avg_nrmse_eff_fed)
    print("Avg Federated relative error y at t+1:", avg_rel_err_y_fed_t)
    print("Avg Federated symmetric relative error y at t+1:", avg_sym_rel_err_y_fed_t)
    print("Avg Delta_y_cf at t+1:", avg_delta_y_cf)
    print("Avg Delta_y_base at t+1:", avg_delta_y_base)
    print("Avg Abs(delta_y_cf - delta_y_base) at t+1:", avg_delta_y_diff)
    print("Avg Delta_y_cf at t+1 (per-dim):", avg_delta_y_cf_norm)
    print("Avg Delta_y_base at t+1 (per-dim):", avg_delta_y_base_norm)
    print("Avg Abs(delta_y_cf - delta_y_base) at t+1 (per-dim):", avg_delta_y_diff_norm)
    print("Avg Percent error at t+1 (%):", avg_delta_y_pct_err)
    print("Avg Percent error at t+1 (per-dim, %):", avg_delta_y_pct_err_norm)


if __name__ == "__main__":  # entry point guard
    main()  # run the experiment
