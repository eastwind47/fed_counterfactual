from __future__ import annotations  # allow forward type annotations (PEP 563 / Py3.11 style)

import argparse                 # command-line argument parsing (we keep only non-training knobs)
import copy                     # deep copies for Monte Carlo aggregation
import csv                      # write metrics CSV
import pickle                   # save Python objects (history snapshots)
from pathlib import Path        # filesystem paths

import numpy as np              # numerical arrays

from data_retriever import RetrieveData            # loads data/config and packs
from global_learner1 import GlobalModel            # server-side model (old framework)
from local_learner1 import LocalModel              # client-side model (old framework)
from spectral_radius1 import SpectralRadiusAnalyzer  # optional spectral diagnostics

# =====================================================================
# Spectral radius diagnostics flag (no CLI): turn on to print diagnostics
# =====================================================================
CHECK_SPECTRAL_RADIUS: bool = True  # set True to compute/print spectral radius before training

# =====================================================================
# Stopping configuration (edit here, no CLI needed)
# =====================================================================
# Which criteria to use (all enabled criteria must be satisfied with patience to stop)
USE_CRITERIA = {
    "max_rounds": True,          # always keep a safety cap by default
    "loss_stagnation": True,     # relative change in smoothed global loss below threshold
    "param_stagnation": True,   # relative change in parameters (A,B,theta,phi) below threshold
    "grad_small": True,         # mean client gradient norm small
    "residual_small": True,     # align/consensus residuals small
}

# Stop as soon as any enabled criterion fires ("any") or only when all do ("all")
STOP_MODE = "all"

# Thresholds and smoothing/patience (tune here)
MAX_ROUNDS = 500                 # safety cap; training stops earlier if other criteria trigger
WINDOW_W = 5                     # window width for loss smoothing (moving average)
PATIENCE_P = 3                   # consecutive rounds that a criterion must hold
EPS_LOSS = 1e-4                  # relative loss change threshold
EPS_PARAM = 5e-5                 # relative parameter change threshold
EPS_GRAD = 1e-8                  # small gradient threshold
EPS_RESID = 1e-3                 # small residual threshold

# Training/control knobs (non-CLI)
SAVE_EVERY: int = 5
SEED: int | None = 0
MONTE_CARLO: int = 3
EPS_DIAG: float = 1e-12

# =========================
# Helpers for stopping logic
# =========================
class StoppingController:
    """Encapsulates early-stopping logic with patience for multiple criteria."""

    def __init__(self):
        # counters for how many consecutive rounds each criterion has held
        self.ok_counts = {
            "loss_stagnation": 0,
            "param_stagnation": 0,
            "grad_small": 0,
            "residual_small": 0,
        }
        # cache of params from the previous round for relative-change checks
        self.last_params = None  # set after first round

    @staticmethod
    def _rel_change(new: float, old: float) -> float:
        """Relative change |new-old|/max(1,|old|) to avoid divide-by-zero."""
        denom = max(1.0, abs(old))
        return abs(new - old) / denom

    def _loss_stagnation(self, loss_hist: list[float]) -> bool:
        """Return True if the moving-average global loss changed little."""
        if len(loss_hist) < WINDOW_W + 1:
            return False  # not enough points to form two windows
        # current and previous moving averages
        curr = sum(loss_hist[-WINDOW_W:]) / WINDOW_W
        prev = sum(loss_hist[-(2*WINDOW_W):-WINDOW_W]) / WINDOW_W
        return self._rel_change(curr, prev) < EPS_LOSS

    def _param_stagnation(self, curr_params: dict) -> bool:
        """Return True if the stacked parameters hardly changed (Frobenius)."""
        if self.last_params is None:
            return False  # need at least one previous snapshot

        def frob_sum(dct: dict) -> float:
            # sum of Frobenius norms for a dict of arrays
            return sum(float((v**2).sum()**0.5) for v in dct.values())

        # Global couplings A
        dA = frob_sum({k: curr_params["A_mn"][k] - self.last_params["A_mn"][k] for k in curr_params["A_mn"]})
        nA = max(1.0, frob_sum(self.last_params["A_mn"]))
        relA = dA / nA

        # Global couplings B
        dB = frob_sum({k: curr_params["B_mn"][k] - self.last_params["B_mn"][k] for k in curr_params["B_mn"]})
        nB = max(1.0, frob_sum(self.last_params["B_mn"]))
        relB = dB / nB

        # Local parameters θ
        dtheta = sum(float(np.linalg.norm(curr_params["theta"][cid] - self.last_params["theta"][cid]))
                     for cid in curr_params["theta"])
        ntheta = max(1.0, sum(float(np.linalg.norm(self.last_params["theta"][cid])) for cid in self.last_params["theta"]))
        relT = dtheta / ntheta

        # Local parameters φ
        dphi = sum(float(np.linalg.norm(curr_params["phi"][cid] - self.last_params["phi"][cid]))
                   for cid in curr_params["phi"])
        nphi = max(1.0, sum(float(np.linalg.norm(self.last_params["phi"][cid])) for cid in self.last_params["phi"]))
        relP = dphi / nphi

        # use the worst relative change as the criterion
        rel = max(relA, relB, relT, relP)
        return rel < EPS_PARAM

    def _grad_small(self, grad_norms: dict[str, float]) -> bool:
        """Return True if mean client gradient norm is small."""
        if not grad_norms:
            return False
        mean_grad = float(sum(grad_norms.values()) / max(1, len(grad_norms)))
        return mean_grad < EPS_GRAD

    def _residual_small(self, align_norm: float, consensus_norm: float) -> bool:
        """Return True if both alignment and consensus residuals are small."""
        return (align_norm < EPS_RESID) and (consensus_norm < EPS_RESID)

    def update_and_should_stop(
        self,
        history: dict,
        curr_params: dict,
        grad_norms: dict[str, float],
        align_norm: float,
        consensus_norm: float,
        round_idx: int,
    ) -> tuple[bool, list[str]]:
        """Update patience counters and decide whether to stop. Returns (stop?, reasons)."""
        satisfied = []  # names of criteria that held this round (with patience)

        # Loss stagnation criterion
        if USE_CRITERIA.get("loss_stagnation", False):
            ok = self._loss_stagnation(history["global_loss"])
            self.ok_counts["loss_stagnation"] = self.ok_counts["loss_stagnation"] + 1 if ok else 0
            if self.ok_counts["loss_stagnation"] >= PATIENCE_P:
                satisfied.append("loss_stagnation")

        # Parameter stagnation criterion
        if USE_CRITERIA.get("param_stagnation", False):
            ok = self._param_stagnation(curr_params)
            self.ok_counts["param_stagnation"] = self.ok_counts["param_stagnation"] + 1 if ok else 0
            if self.ok_counts["param_stagnation"] >= PATIENCE_P:
                satisfied.append("param_stagnation")

        # Small gradient criterion
        if USE_CRITERIA.get("grad_small", False):
            ok = self._grad_small(grad_norms)
            self.ok_counts["grad_small"] = self.ok_counts["grad_small"] + 1 if ok else 0
            if self.ok_counts["grad_small"] >= PATIENCE_P:
                satisfied.append("grad_small")

        # Small residual criterion
        if USE_CRITERIA.get("residual_small", False):
            ok = self._residual_small(align_norm, consensus_norm)
            self.ok_counts["residual_small"] = self.ok_counts["residual_small"] + 1 if ok else 0
            if self.ok_counts["residual_small"] >= PATIENCE_P:
                satisfied.append("residual_small")

        # Save last params for next round comparisons
        self.last_params = curr_params

        # Evaluate enabled criteria (excluding max_rounds) according to STOP_MODE
        enabled = [k for k, v in USE_CRITERIA.items() if v and k != "max_rounds"]
        stop = False
        if enabled:
            if STOP_MODE.lower() == "any":
                stop = bool(satisfied)
            else:  # default back to "all"
                stop = all(c in satisfied for c in enabled)

        # Cap by max rounds if enabled regardless of other criteria
        if USE_CRITERIA.get("max_rounds", True) and round_idx >= MAX_ROUNDS:
            return True, satisfied + (['max_rounds'] if 'max_rounds' not in satisfied else [])
        return stop, satisfied


def parse_args() -> argparse.Namespace:
    """Only keep non-training knobs in CLI; training control lives in-file."""
    parser = argparse.ArgumentParser(
        description="Train the revised counterfactual learning pipeline."
    )
    parser.add_argument(
        "--config",
        default="config.ini",  # default config path
        help="Path to the configuration file produced for this experiment.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY,
        help="Interval (in rounds) for storing parameter snapshots. Use 0 to skip.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,           # optional override for results directory
        help=(
            "Optional override for the directory that will receive logs and "
            "snapshots."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,           # optional random seed for reproducibility
        help="Random seed for reproducible client initialisation.",
    )
    return parser.parse_args()


def prepare_results_directory(config_results: str, override: Path | None) -> tuple[Path, Path]:
    """Create results/snapshots directories (robust to missing permissions)."""
    if override is not None:
        base_dir = override  # explicit override wins
    else:
        base_dir = Path(config_results) / "main1"  # default subdir

    try:
        base_dir.mkdir(parents=True, exist_ok=True)  # ensure base exists
    except OSError:
        # fallback to ./results/main1 if original location not writable
        fallback = Path("results") / "main1"
        fallback.mkdir(parents=True, exist_ok=True)
        base_dir = fallback

    snapshots_dir = base_dir / "snapshots"  # subdir for snapshots
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, snapshots_dir


def build_models(
    data: RetrieveData,
    check_spectral_radius: bool = False,
) -> tuple[
    dict[str, LocalModel],
    GlobalModel,
    dict[str, np.ndarray],
    SpectralRadiusAnalyzer | None,
]:
    """Instantiate all local models, run their DKF, and build the global model."""
    M = data.num_components                  # number of components/clients
    T = data.training_time                   # training horizon (timesteps)

    local_models: dict[str, LocalModel] = {}  # id -> LocalModel
    x_dkf: dict[str, np.ndarray] = {}        # id -> DKF state trajectories
    u_global: dict[str, np.ndarray] = {}     # id -> input trajectories

    for m in range(M):  # loop over components
        comp_id = f"{m + 1}"                                # human-friendly id
        comp_key = f"comp_{m + 1}"                          # key in data pack
        comp_data = data.local_learners_pack[comp_key]       # dict with arrays

        local_model = LocalModel(T, comp_data)
        if "X_dkf" in comp_data:
            local_model.X_dkf = comp_data["X_dkf"].copy()
        if "X_dkf_pred" in comp_data:
            local_model.X_dkf_pred = comp_data["X_dkf_pred"].copy()
        if "X_dkf_resd" in comp_data:
            local_model.X_dkf_resd = comp_data["X_dkf_resd"].copy()


        local_models[comp_id] = local_model                  # register model
        x_dkf[comp_id] = local_model.X_dkf.copy()            # cache DKF states
        u_global[comp_id] = comp_data["U"].copy()            # cache inputs

    # Global model uses only global learners pack + DKF states
    global_model = GlobalModel(M, T, data.global_learners_pack, x_dkf)

    analyzer = None
    if check_spectral_radius:                      # optionally build analyzer
        analyzer = SpectralRadiusAnalyzer(data, x_dkf)
    return local_models, global_model, u_global, analyzer


def _block_norms(blocks: dict[str, np.ndarray]) -> tuple[dict[str, float], float]:
    """Return per-block Frobenius norms and the global stacked Frobenius norm."""
    per_block = {
        key: float(np.linalg.norm(val))
        for key, val in blocks.items()
    }
    total = float(np.sqrt(sum(val * val for val in per_block.values())))
    return per_block, total


def _block_step_metrics(
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    eps: float = EPS_DIAG,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Return per-block and total absolute/relative parameter-step metrics."""
    step = {
        key: float(np.linalg.norm(current[key] - previous[key]))
        for key in current
    }
    rel_step = {
        key: float(step[key] / max(float(np.linalg.norm(previous[key])), eps))
        for key in step
    }
    step_total = float(np.sqrt(sum(val * val for val in step.values())))
    prev_total = float(np.sqrt(sum(float(np.linalg.norm(val)) ** 2 for val in previous.values())))
    rel_total = float(step_total / max(prev_total, eps))
    return step, rel_step, step_total, rel_total


def run_round(
    global_model: GlobalModel,
    local_models: dict[str, LocalModel],
    u_global: dict[str, np.ndarray],
) -> tuple[
    float,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float,
    float,
    dict[str, float | dict[str, float]],
]:
    """One training round: local fwd pass → server step → local updates."""
    h_aug_pred = {}                # client augmented predicted states (to server)
    h_aug_est = {}                 # client augmented estimated states (to server)
    local_losses: dict[str, float] = {}         # per-client local losses
    residual_norms: dict[str, float] = {}       # per-client residual norms

    # ---- Local forward passes ----
    for comp_id, local_model in local_models.items():
        payload = local_model.local_forward_pass()            # run client's fwd pass
        h_aug_pred[comp_id] = payload["h_aug_pred"]          # predicted augmented state
        h_aug_est[comp_id] = payload["h_aug_est"]            # estimated augmented state
        local_losses[comp_id] = float(payload["local_loss"]) # scalar local loss
        residuals = payload["residuals"]                     # residual vector(s)
        residual_norms[comp_id] = float(np.linalg.norm(residuals) / max(global_model.T, 1))

    # ---- Global/server step ----
    server_out = global_model.server_forward_pass(h_aug_pred, h_aug_est, u_global)
    global_model.apply_server_updates()  # gradient step on \hat A, \hat B (old model)

    # ---- Pull gradients for clients and update them ----
    grad_payload, grad_est_payload = global_model.get_gradients_for_clients(include_est=True)
    grad_norms: dict[str, float] = {}
    for comp_id, grad in grad_payload.items():
        local_models[comp_id].gradient_descent_update(grad, grad_est_payload[comp_id])
        grad_norms[comp_id] = float(np.linalg.norm(grad) / max(global_model.T, 1))

    # ---- Aggregate residual diagnostics ----
    align_norm = float(
        sum(np.linalg.norm(val) for val in global_model.align_residuals.values())
        / max(global_model.T, 1)
    )
    consensus_norm = float(
        sum(np.linalg.norm(val) for val in global_model.consensus_history.values())
        / max(global_model.T, 1)
    )
    loss_align = float(server_out.get("loss_align", 0.0))
    loss_consensus = float(server_out.get("loss_consensus", 0.0))
    loss_total = max(loss_align + loss_consensus, EPS_DIAG)
    server_diagnostics: dict[str, float | dict[str, float]] = {
        "loss_align": loss_align,
        "loss_consensus": loss_consensus,
        "loss_align_frac": float(loss_align / loss_total),
        "loss_consensus_frac": float(loss_consensus / loss_total),
        "gradA_norms": {
            key: float(val)
            for key, val in server_out.get("grad_A_norms", {}).items()
        },
        "gradB_norms": {
            key: float(val)
            for key, val in server_out.get("grad_B_norms", {}).items()
        },
        "gradA_norm_total": float(server_out.get("grad_A_total", 0.0)),
        "gradB_norm_total": float(server_out.get("grad_B_total", 0.0)),
    }

    return (
        float(server_out["loss"]),  # global/server loss this round
        local_losses,                # dict of local losses
        grad_norms,                  # dict of local grad norms
        residual_norms,              # dict of local residual norms
        align_norm,                  # alignment residual aggregate
        consensus_norm,              # consensus residual aggregate
        server_diagnostics,          # decomposition + parameter gradient norms
    )


def snapshot_state(
    round_idx: int,
    global_model: GlobalModel,
    local_models: dict[str, LocalModel],
    snapshot_dir: Path,
) -> None:
    """Persist a snapshot of parameters for later analysis/plots."""
    snapshot = {
        "round": round_idx,
        "A_mn": {key: value.copy() for key, value in global_model.A_mn.items()},  # global couplings A
        "B_mn": {key: value.copy() for key, value in global_model.B_mn.items()},  # global couplings B
        "theta": {cid: model.theta.copy() for cid, model in local_models.items()},  # per-client theta
        "phi": {cid: model.phi.copy() for cid, model in local_models.items()},      # per-client phi
    }
    snapshot_path = snapshot_dir / f"round_{round_idx:04d}.pkl"  # destination filename
    with snapshot_path.open("wb") as handle:
        pickle.dump(snapshot, handle)  # write snapshot as pickle


def write_metrics_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    """Write collected per-round metrics into a CSV (one row per round)."""
    if not rows:
        return
    # collect all keys seen across rows; sort placing 'round' first
    headers = sorted({key for row in rows for key in row.keys()}, key=lambda x: ("round" not in x, x))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Entry point: load data, build models, train with early stopping, save artifacts."""
    args = parse_args()  # parse non-training CLI knobs

    if args.seed is not None:
        np.random.seed(args.seed)  # set random seed for reproducibility

    data = RetrieveData(args.config)  # load data/config packs

    # Use the top-level flag for spectral radius diagnostics (no CLI)
    check_spectral_radius = bool(CHECK_SPECTRAL_RADIUS)

    # Build local/global models and optional spectral analyzer for diagnostics
    local_models, global_model, u_global, analyzer = build_models(
        data, check_spectral_radius
    )

    component_ids = sorted(local_models.keys(), key=lambda item: int(item))

    ckf_pack = data.CKF_pack
    A_true_full = ckf_pack["A_complete"]
    B_true_full = ckf_pack["B_complete"]
    p_vec = np.asarray(ckf_pack["p_vec"]).astype(int).reshape(-1)
    s_vec = np.asarray(ckf_pack["s_vec"]).astype(int).reshape(-1)
    input_start_vec = np.asarray(ckf_pack["input_start_vec"]).astype(int).reshape(-1)
    comp_start_vec = ckf_pack.get("comp_start_vec")
    if comp_start_vec is not None:
        state_start_vec = np.asarray(comp_start_vec).astype(int).reshape(-1)
    else:
        state_start_vec = np.zeros_like(p_vec)
        running_state = 0
        for idx, width in enumerate(p_vec):
            state_start_vec[idx] = running_state
            running_state += int(width)

    def _block_key(m_idx: int, n_idx: int) -> str:
        return f"{m_idx}{n_idx}"

    A_true_blocks: dict[str, np.ndarray] = {}
    B_true_blocks: dict[str, np.ndarray] = {}
    A_true_norms: dict[str, float] = {}
    B_true_norms: dict[str, float] = {}
    for i in range(data.num_components):
        row_start = state_start_vec[i]
        row_end = row_start + p_vec[i]
        for j in range(data.num_components):
            if i == j:
                continue
            col_start_a = state_start_vec[j]
            col_end_a = col_start_a + p_vec[j]
            col_start_b = input_start_vec[j]
            col_end_b = col_start_b + s_vec[j]
            key = _block_key(i + 1, j + 1)
            A_block = A_true_full[row_start:row_end, col_start_a:col_end_a].copy()
            B_block = B_true_full[row_start:row_end, col_start_b:col_end_b].copy()
            A_true_blocks[key] = A_block
            B_true_blocks[key] = B_block
            A_true_norms[key] = float(np.linalg.norm(A_block, ord="fro"))
            B_true_norms[key] = float(np.linalg.norm(B_block, ord="fro"))

    offdiag_keys = sorted(global_model.A_mn.keys(), key=lambda k: (len(k), k))

    results_dir, snapshots_dir = prepare_results_directory(
        data.results_location, args.results_dir
    )

    if analyzer is not None:
        print("Spectral radius diagnostics before training:")
        analyzer.print_spectral_summary()

    history = {
        "round": [],
        "global_loss": [],
        "align_norm": [],
        "consensus_norm": [],
        "loss_align": [],
        "loss_consensus": [],
        "loss_align_frac": [],
        "loss_consensus_frac": [],
        "gradA_norm_total": [],
        "gradB_norm_total": [],
        "A_norm_total": [],
        "B_norm_total": [],
        "A_step_total": [],
        "B_step_total": [],
        "A_rel_step_total": [],
        "B_rel_step_total": [],
        "local_loss": {cid: [] for cid in component_ids},
        "grad_norm": {cid: [] for cid in component_ids},
        "residual_norm": {cid: [] for cid in component_ids},
        "theta_norm": {cid: [] for cid in component_ids},
        "phi_norm": {cid: [] for cid in component_ids},
        "phi_consensus_err": {cid: [] for cid in component_ids},
        "gradA_norm": {key: [] for key in offdiag_keys},
        "gradB_norm": {key: [] for key in offdiag_keys},
        "A_norm": {key: [] for key in offdiag_keys},
        "B_norm": {key: [] for key in offdiag_keys},
        "A_step": {key: [] for key in offdiag_keys},
        "B_step": {key: [] for key in offdiag_keys},
        "A_rel_step": {key: [] for key in offdiag_keys},
        "B_rel_step": {key: [] for key in offdiag_keys},
    }
    history["A_abs_err"] = {key: [] for key in offdiag_keys}
    history["B_abs_err"] = {key: [] for key in offdiag_keys}
    history["A_rel_err"] = {key: [] for key in offdiag_keys}
    history["B_rel_err"] = {key: [] for key in offdiag_keys}
    metric_rows: list[dict[str, float]] = []

    last_A_mn: dict[str, np.ndarray] = {key: val.copy() for key, val in global_model.A_mn.items()}
    last_B_mn: dict[str, np.ndarray] = {key: val.copy() for key, val in global_model.B_mn.items()}

    u_mean = {cid: np.asarray(u_global[cid]).mean(axis=1).reshape(-1) for cid in component_ids}

    stopper = StoppingController()

    print(
        f"Starting training with {len(component_ids)} components, horizon {global_model.T}, "
        f"up to {MAX_ROUNDS} rounds (early stopping enabled)."
    )

    last_round_idx = 0

    for round_idx in range(1, MAX_ROUNDS + 1):
        (
            global_loss,
            local_losses,
            grad_norms,
            residual_norms,
            align_norm,
            consensus_norm,
            server_diagnostics,
        ) = run_round(global_model, local_models, u_global)

        A_abs_err: dict[str, float] = {}
        B_abs_err: dict[str, float] = {}
        A_rel_err: dict[str, float] = {}
        B_rel_err: dict[str, float] = {}
        for key in offdiag_keys:
            A_est = global_model.A_mn[key]
            B_est = global_model.B_mn[key]
            A_err = float(np.linalg.norm(A_est - A_true_blocks[key], ord="fro"))
            B_err = float(np.linalg.norm(B_est - B_true_blocks[key], ord="fro"))
            A_denom = A_true_norms[key]
            B_denom = B_true_norms[key]
            history["A_abs_err"][key].append(A_err)
            history["B_abs_err"][key].append(B_err)
            A_abs_err[key] = A_err
            B_abs_err[key] = B_err
            if A_denom > 1e-12:
                A_rel = A_err / A_denom
            else:
                A_rel = float("nan") if A_err > 1e-12 else 0.0
            if B_denom > 1e-12:
                B_rel = B_err / B_denom
            else:
                B_rel = float("nan") if B_err > 1e-12 else 0.0
            history["A_rel_err"][key].append(A_rel)
            history["B_rel_err"][key].append(B_rel)
            A_rel_err[key] = A_rel
            B_rel_err[key] = B_rel

        gradA_norms = {
            key: float(server_diagnostics.get("gradA_norms", {}).get(key, 0.0))
            for key in offdiag_keys
        }
        gradB_norms = {
            key: float(server_diagnostics.get("gradB_norms", {}).get(key, 0.0))
            for key in offdiag_keys
        }
        A_norms, A_norm_total = _block_norms(global_model.A_mn)
        B_norms, B_norm_total = _block_norms(global_model.B_mn)
        A_step, A_rel_step, A_step_total, A_rel_step_total = _block_step_metrics(
            global_model.A_mn,
            last_A_mn,
        )
        B_step, B_rel_step, B_step_total, B_rel_step_total = _block_step_metrics(
            global_model.B_mn,
            last_B_mn,
        )

        history["round"].append(round_idx)
        history["global_loss"].append(global_loss)
        history["align_norm"].append(align_norm)
        history["consensus_norm"].append(consensus_norm)
        history["loss_align"].append(float(server_diagnostics.get("loss_align", 0.0)))
        history["loss_consensus"].append(float(server_diagnostics.get("loss_consensus", 0.0)))
        history["loss_align_frac"].append(float(server_diagnostics.get("loss_align_frac", 0.0)))
        history["loss_consensus_frac"].append(float(server_diagnostics.get("loss_consensus_frac", 0.0)))
        history["gradA_norm_total"].append(float(server_diagnostics.get("gradA_norm_total", 0.0)))
        history["gradB_norm_total"].append(float(server_diagnostics.get("gradB_norm_total", 0.0)))
        history["A_norm_total"].append(A_norm_total)
        history["B_norm_total"].append(B_norm_total)
        history["A_step_total"].append(A_step_total)
        history["B_step_total"].append(B_step_total)
        history["A_rel_step_total"].append(A_rel_step_total)
        history["B_rel_step_total"].append(B_rel_step_total)

        row = {
            "round": round_idx,
            "global_loss": global_loss,
            "align_norm": align_norm,
            "consensus_norm": consensus_norm,
            "loss_align": float(server_diagnostics.get("loss_align", 0.0)),
            "loss_consensus": float(server_diagnostics.get("loss_consensus", 0.0)),
            "loss_align_frac": float(server_diagnostics.get("loss_align_frac", 0.0)),
            "loss_consensus_frac": float(server_diagnostics.get("loss_consensus_frac", 0.0)),
            "gradA_norm_total": float(server_diagnostics.get("gradA_norm_total", 0.0)),
            "gradB_norm_total": float(server_diagnostics.get("gradB_norm_total", 0.0)),
            "A_norm_total": A_norm_total,
            "B_norm_total": B_norm_total,
            "A_step_total": A_step_total,
            "B_step_total": B_step_total,
            "A_rel_step_total": A_rel_step_total,
            "B_rel_step_total": B_rel_step_total,
        }

        for cid in component_ids:
            history["local_loss"][cid].append(local_losses[cid])
            history["grad_norm"][cid].append(grad_norms[cid])
            history["residual_norm"][cid].append(residual_norms[cid])
            theta_norm = float(np.linalg.norm(local_models[cid].theta))
            phi_norm = float(np.linalg.norm(local_models[cid].phi))
            history["theta_norm"][cid].append(theta_norm)
            history["phi_norm"][cid].append(phi_norm)

            phi_vec = local_models[cid].phi.reshape(-1)
            target = np.zeros_like(phi_vec, dtype=float)
            for peer in component_ids:
                if peer == cid:
                    continue
                key_pair = f"{cid}{peer}"
                block = global_model.B_mn.get(key_pair)
                if block is None:
                    continue
                target += np.asarray(block) @ u_mean[peer]
            phi_consensus_err = float(np.linalg.norm(phi_vec - target))
            history["phi_consensus_err"][cid].append(phi_consensus_err)

            row[f"local_loss_{cid}"] = local_losses[cid]
            row[f"grad_norm_{cid}"] = grad_norms[cid]
            row[f"residual_norm_{cid}"] = residual_norms[cid]
            row[f"theta_norm_{cid}"] = theta_norm
            row[f"phi_norm_{cid}"] = phi_norm
            row[f"phi_consensus_err_{cid}"] = phi_consensus_err

        for key in offdiag_keys:
            row[f"A_abs_err_{key}"] = A_abs_err[key]
            row[f"B_abs_err_{key}"] = B_abs_err[key]
            row[f"A_rel_err_{key}"] = A_rel_err[key]
            row[f"B_rel_err_{key}"] = B_rel_err[key]
            row[f"gradA_norm_{key}"] = gradA_norms[key]
            row[f"gradB_norm_{key}"] = gradB_norms[key]
            row[f"A_norm_{key}"] = A_norms[key]
            row[f"B_norm_{key}"] = B_norms[key]
            row[f"A_step_{key}"] = A_step[key]
            row[f"B_step_{key}"] = B_step[key]
            row[f"A_rel_step_{key}"] = A_rel_step[key]
            row[f"B_rel_step_{key}"] = B_rel_step[key]
            history["gradA_norm"][key].append(gradA_norms[key])
            history["gradB_norm"][key].append(gradB_norms[key])
            history["A_norm"][key].append(A_norms[key])
            history["B_norm"][key].append(B_norms[key])
            history["A_step"][key].append(A_step[key])
            history["B_step"][key].append(B_step[key])
            history["A_rel_step"][key].append(A_rel_step[key])
            history["B_rel_step"][key].append(B_rel_step[key])

        metric_rows.append(row)

        last_A_mn = {key: global_model.A_mn[key].copy() for key in offdiag_keys}
        last_B_mn = {key: global_model.B_mn[key].copy() for key in offdiag_keys}

        print(
            f"Round {round_idx:03d} | global loss {global_loss:.6e} | "
            f"align {align_norm:.3e} | consensus {consensus_norm:.3e}"
        )

        curr_params = {
            "A_mn": {k: v.copy() for k, v in global_model.A_mn.items()},
            "B_mn": {k: v.copy() for k, v in global_model.B_mn.items()},
            "theta": {cid: local_models[cid].theta.copy() for cid in component_ids},
            "phi":   {cid: local_models[cid].phi.copy()   for cid in component_ids},
        }

        should_stop, satisfied = stopper.update_and_should_stop(
            history=history,
            curr_params=curr_params,
            grad_norms=grad_norms,
            align_norm=align_norm,
            consensus_norm=consensus_norm,
            round_idx=round_idx,
        )
        last_round_idx = round_idx

        if should_stop:
            why = ", ".join(satisfied) if satisfied else "criteria met"
            print(f"Early stopping at round {round_idx} ({why}).")
            break

        if args.save_every > 0 and round_idx % args.save_every == 0:
            snapshot_state(round_idx, global_model, local_models, snapshots_dir)

    history["final_A_mn"] = {key: last_A_mn[key].copy() for key in offdiag_keys}
    history["final_B_mn"] = {key: last_B_mn[key].copy() for key in offdiag_keys}

    snapshot_state(last_round_idx, global_model, local_models, snapshots_dir)

    history_path = results_dir / "training_history.pkl"
    with history_path.open("wb") as handle:
        pickle.dump(history, handle)

    metrics_csv_path = results_dir / "training_metrics.csv"
    write_metrics_csv(metric_rows, metrics_csv_path)

    couplings_payload = {f"A_{key}": value for key, value in global_model.A_mn.items()}
    couplings_payload.update({f"B_{key}": value for key, value in global_model.B_mn.items()})
    np.savez(results_dir / "couplings_final.npz", **couplings_payload)

    print(f"Run complete. Artifacts saved to {results_dir}.")


def run_single_experiment(
    exp_idx: int,
    args: argparse.Namespace,
    data: RetrieveData,
    check_spectral_radius: bool,
    base_results_dir: Path,
    A_true_norms: dict[str, float],
    B_true_norms: dict[str, float],
) -> dict:
    local_models, global_model, u_global, analyzer = build_models(
        data, check_spectral_radius
    )

    component_ids = sorted(local_models.keys(), key=lambda item: int(item))

    ckf_pack = data.CKF_pack
    A_true_full = ckf_pack["A_complete"]
    B_true_full = ckf_pack["B_complete"]
    p_vec = np.asarray(ckf_pack["p_vec"]).astype(int).reshape(-1)
    s_vec = np.asarray(ckf_pack["s_vec"]).astype(int).reshape(-1)
    input_start_vec = np.asarray(ckf_pack["input_start_vec"]).astype(int).reshape(-1)
    comp_start_vec = ckf_pack.get("comp_start_vec")
    if comp_start_vec is not None:
        state_start_vec = np.asarray(comp_start_vec).astype(int).reshape(-1)
    else:
        state_start_vec = np.zeros_like(p_vec)
        running_state = 0
        for idx, width in enumerate(p_vec):
            state_start_vec[idx] = running_state
            running_state += int(width)

    def _block_key(m_idx: int, n_idx: int) -> str:
        return f"{m_idx}{n_idx}"

    A_true_blocks: dict[str, np.ndarray] = {}
    B_true_blocks: dict[str, np.ndarray] = {}
    local_A_norms: dict[str, float] = {}
    local_B_norms: dict[str, float] = {}
    for i in range(data.num_components):
        row_start = state_start_vec[i]
        row_end = row_start + p_vec[i]
        for j in range(data.num_components):
            if i == j:
                continue
            col_start_a = state_start_vec[j]
            col_end_a = col_start_a + p_vec[j]
            col_start_b = input_start_vec[j]
            col_end_b = col_start_b + s_vec[j]
            key = _block_key(i + 1, j + 1)
            A_block = A_true_full[row_start:row_end, col_start_a:col_end_a].copy()
            B_block = B_true_full[row_start:row_end, col_start_b:col_end_b].copy()
            A_true_blocks[key] = A_block
            B_true_blocks[key] = B_block
            local_A_norms[key] = A_true_norms.get(key, float(np.linalg.norm(A_block, ord="fro")))
            local_B_norms[key] = B_true_norms.get(key, float(np.linalg.norm(B_block, ord="fro")))

    offdiag_keys = sorted(global_model.A_mn.keys(), key=lambda k: (len(k), k))

    run_dir = base_results_dir / f"mc_{exp_idx + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    if analyzer is not None:
        print(f"[MC {exp_idx + 1:03d}] Spectral radius diagnostics before training:")
        analyzer.print_spectral_summary()

    history = {
        "round": [],
        "global_loss": [],
        "align_norm": [],
        "consensus_norm": [],
        "loss_align": [],
        "loss_consensus": [],
        "loss_align_frac": [],
        "loss_consensus_frac": [],
        "gradA_norm_total": [],
        "gradB_norm_total": [],
        "A_norm_total": [],
        "B_norm_total": [],
        "A_step_total": [],
        "B_step_total": [],
        "A_rel_step_total": [],
        "B_rel_step_total": [],
        "local_loss": {cid: [] for cid in component_ids},
        "grad_norm": {cid: [] for cid in component_ids},
        "residual_norm": {cid: [] for cid in component_ids},
        "theta_norm": {cid: [] for cid in component_ids},
        "phi_norm": {cid: [] for cid in component_ids},
        "phi_consensus_err": {cid: [] for cid in component_ids},
        "gradA_norm": {key: [] for key in offdiag_keys},
        "gradB_norm": {key: [] for key in offdiag_keys},
        "A_norm": {key: [] for key in offdiag_keys},
        "B_norm": {key: [] for key in offdiag_keys},
        "A_step": {key: [] for key in offdiag_keys},
        "B_step": {key: [] for key in offdiag_keys},
        "A_rel_step": {key: [] for key in offdiag_keys},
        "B_rel_step": {key: [] for key in offdiag_keys},
    }
    metric_rows: list[dict[str, float]] = []
    history["A_abs_err"] = {key: [] for key in offdiag_keys}
    history["B_abs_err"] = {key: [] for key in offdiag_keys}
    history["A_rel_err"] = {key: [] for key in offdiag_keys}
    history["B_rel_err"] = {key: [] for key in offdiag_keys}

    last_A_mn: dict[str, np.ndarray] = {key: val.copy() for key, val in global_model.A_mn.items()}
    last_B_mn: dict[str, np.ndarray] = {key: val.copy() for key, val in global_model.B_mn.items()}

    u_mean = {cid: np.asarray(u_global[cid]).mean(axis=1).reshape(-1) for cid in component_ids}

    stopper = StoppingController()
    last_round_idx = 0
    stop_reason = ""

    print(
        f"[MC {exp_idx + 1:03d}] Starting training with {len(component_ids)} components, "
        f"horizon {global_model.T}, up to {MAX_ROUNDS} rounds."
    )

    for round_idx in range(1, MAX_ROUNDS + 1):
        (
            global_loss,
            local_losses,
            grad_norms,
            residual_norms,
            align_norm,
            consensus_norm,
            server_diagnostics,
        ) = run_round(global_model, local_models, u_global)

        A_abs_err: dict[str, float] = {}
        B_abs_err: dict[str, float] = {}
        A_rel_err: dict[str, float] = {}
        B_rel_err: dict[str, float] = {}
        for key in offdiag_keys:
            A_est = global_model.A_mn[key]
            B_est = global_model.B_mn[key]
            A_err = float(np.linalg.norm(A_est - A_true_blocks[key], ord="fro"))
            B_err = float(np.linalg.norm(B_est - B_true_blocks[key], ord="fro"))
            A_denom = local_A_norms[key]
            B_denom = local_B_norms[key]
            history["A_abs_err"][key].append(A_err)
            history["B_abs_err"][key].append(B_err)
            A_abs_err[key] = A_err
            B_abs_err[key] = B_err
            if A_denom > 1e-12:
                A_rel = A_err / A_denom
            else:
                A_rel = float("nan") if A_err > 1e-12 else 0.0
            if B_denom > 1e-12:
                B_rel = B_err / B_denom
            else:
                B_rel = float("nan") if B_err > 1e-12 else 0.0
            history["A_rel_err"][key].append(A_rel)
            history["B_rel_err"][key].append(B_rel)
            A_rel_err[key] = A_rel
            B_rel_err[key] = B_rel

        gradA_norms = {
            key: float(server_diagnostics.get("gradA_norms", {}).get(key, 0.0))
            for key in offdiag_keys
        }
        gradB_norms = {
            key: float(server_diagnostics.get("gradB_norms", {}).get(key, 0.0))
            for key in offdiag_keys
        }
        A_norms, A_norm_total = _block_norms(global_model.A_mn)
        B_norms, B_norm_total = _block_norms(global_model.B_mn)
        A_step, A_rel_step, A_step_total, A_rel_step_total = _block_step_metrics(
            global_model.A_mn,
            last_A_mn,
        )
        B_step, B_rel_step, B_step_total, B_rel_step_total = _block_step_metrics(
            global_model.B_mn,
            last_B_mn,
        )

        history["round"].append(round_idx)
        history["global_loss"].append(global_loss)
        history["align_norm"].append(align_norm)
        history["consensus_norm"].append(consensus_norm)
        history["loss_align"].append(float(server_diagnostics.get("loss_align", 0.0)))
        history["loss_consensus"].append(float(server_diagnostics.get("loss_consensus", 0.0)))
        history["loss_align_frac"].append(float(server_diagnostics.get("loss_align_frac", 0.0)))
        history["loss_consensus_frac"].append(float(server_diagnostics.get("loss_consensus_frac", 0.0)))
        history["gradA_norm_total"].append(float(server_diagnostics.get("gradA_norm_total", 0.0)))
        history["gradB_norm_total"].append(float(server_diagnostics.get("gradB_norm_total", 0.0)))
        history["A_norm_total"].append(A_norm_total)
        history["B_norm_total"].append(B_norm_total)
        history["A_step_total"].append(A_step_total)
        history["B_step_total"].append(B_step_total)
        history["A_rel_step_total"].append(A_rel_step_total)
        history["B_rel_step_total"].append(B_rel_step_total)

        row = {
            "round": round_idx,
            "global_loss": global_loss,
            "align_norm": align_norm,
            "consensus_norm": consensus_norm,
            "loss_align": float(server_diagnostics.get("loss_align", 0.0)),
            "loss_consensus": float(server_diagnostics.get("loss_consensus", 0.0)),
            "loss_align_frac": float(server_diagnostics.get("loss_align_frac", 0.0)),
            "loss_consensus_frac": float(server_diagnostics.get("loss_consensus_frac", 0.0)),
            "gradA_norm_total": float(server_diagnostics.get("gradA_norm_total", 0.0)),
            "gradB_norm_total": float(server_diagnostics.get("gradB_norm_total", 0.0)),
            "A_norm_total": A_norm_total,
            "B_norm_total": B_norm_total,
            "A_step_total": A_step_total,
            "B_step_total": B_step_total,
            "A_rel_step_total": A_rel_step_total,
            "B_rel_step_total": B_rel_step_total,
        }

        for cid in component_ids:
            history["local_loss"][cid].append(local_losses[cid])
            history["grad_norm"][cid].append(grad_norms[cid])
            history["residual_norm"][cid].append(residual_norms[cid])
            theta_norm = float(np.linalg.norm(local_models[cid].theta))
            phi_norm = float(np.linalg.norm(local_models[cid].phi))
            history["theta_norm"][cid].append(theta_norm)
            history["phi_norm"][cid].append(phi_norm)

            phi_vec = local_models[cid].phi.reshape(-1)
            target = np.zeros_like(phi_vec, dtype=float)
            for peer in component_ids:
                if peer == cid:
                    continue
                key_pair = f"{cid}{peer}"
                block = global_model.B_mn.get(key_pair)
                if block is None:
                    continue
                target += np.asarray(block) @ u_mean[peer]

            phi_consensus_err = float(np.linalg.norm(phi_vec - target))
            history["phi_consensus_err"][cid].append(phi_consensus_err)

            row[f"local_loss_{cid}"] = local_losses[cid]
            row[f"grad_norm_{cid}"] = grad_norms[cid]
            row[f"residual_norm_{cid}"] = residual_norms[cid]
            row[f"theta_norm_{cid}"] = theta_norm
            row[f"phi_norm_{cid}"] = phi_norm
            row[f"phi_consensus_err_{cid}"] = phi_consensus_err

        for key in offdiag_keys:
            row[f"A_abs_err_{key}"] = A_abs_err[key]
            row[f"B_abs_err_{key}"] = B_abs_err[key]
            row[f"A_rel_err_{key}"] = A_rel_err[key]
            row[f"B_rel_err_{key}"] = B_rel_err[key]
            row[f"gradA_norm_{key}"] = gradA_norms[key]
            row[f"gradB_norm_{key}"] = gradB_norms[key]
            row[f"A_norm_{key}"] = A_norms[key]
            row[f"B_norm_{key}"] = B_norms[key]
            row[f"A_step_{key}"] = A_step[key]
            row[f"B_step_{key}"] = B_step[key]
            row[f"A_rel_step_{key}"] = A_rel_step[key]
            row[f"B_rel_step_{key}"] = B_rel_step[key]
            history["gradA_norm"][key].append(gradA_norms[key])
            history["gradB_norm"][key].append(gradB_norms[key])
            history["A_norm"][key].append(A_norms[key])
            history["B_norm"][key].append(B_norms[key])
            history["A_step"][key].append(A_step[key])
            history["B_step"][key].append(B_step[key])
            history["A_rel_step"][key].append(A_rel_step[key])
            history["B_rel_step"][key].append(B_rel_step[key])

        metric_rows.append(row)

        last_A_mn = {key: global_model.A_mn[key].copy() for key in offdiag_keys}
        last_B_mn = {key: global_model.B_mn[key].copy() for key in offdiag_keys}

        print(
            f"[MC {exp_idx + 1:03d}] Round {round_idx:03d} | global loss {global_loss:.6e} | "
            f"align {align_norm:.3e} | consensus {consensus_norm:.3e} | residual {residual_norms['1']}"
        )

        curr_params = {
            "A_mn": {k: v.copy() for k, v in global_model.A_mn.items()},
            "B_mn": {k: v.copy() for k, v in global_model.B_mn.items()},
            "theta": {cid: local_models[cid].theta.copy() for cid in component_ids},
            "phi":   {cid: local_models[cid].phi.copy()   for cid in component_ids},
        }

        should_stop, satisfied = stopper.update_and_should_stop(
            history=history,
            curr_params=curr_params,
            grad_norms=grad_norms,
            align_norm=align_norm,
            consensus_norm=consensus_norm,
            round_idx=round_idx,
        )
        last_round_idx = round_idx

        if should_stop:
            stop_reason = ", ".join(satisfied) if satisfied else "criteria met"
            print(f"[MC {exp_idx + 1:03d}] Early stopping at round {round_idx} ({stop_reason}).")
            break

        if args.save_every > 0 and round_idx % args.save_every == 0:
            snapshot_state(round_idx, global_model, local_models, snapshots_dir)

    history["final_A_mn"] = {key: last_A_mn[key].copy() for key in offdiag_keys}
    history["final_B_mn"] = {key: last_B_mn[key].copy() for key in offdiag_keys}

    snapshot_state(last_round_idx, global_model, local_models, snapshots_dir)

    history_path = run_dir / "training_history.pkl"
    with history_path.open("wb") as handle:
        pickle.dump(history, handle)

    metrics_csv_path = run_dir / "training_metrics.csv"
    write_metrics_csv(metric_rows, metrics_csv_path)

    couplings_payload = {f"A_{key}": value for key, value in global_model.A_mn.items()}
    couplings_payload.update({f"B_{key}": value for key, value in global_model.B_mn.items()})
    np.savez(run_dir / "couplings_final.npz", **couplings_payload)

    print(f"[MC {exp_idx + 1:03d}] Run complete. Artifacts saved to {run_dir}.")

    return {
        "history": history,
        "metric_rows": metric_rows,
        "stop_reason": stop_reason,
        "rounds_completed": last_round_idx,
        "component_ids": component_ids,
        "final_A_mn": {key: val.copy() for key, val in last_A_mn.items()},
        "final_B_mn": {key: val.copy() for key, val in last_B_mn.items()},
    }


def main() -> None:
    args = parse_args()

    template_data = RetrieveData(args.config)
    if MONTE_CARLO < 1:
        raise ValueError("MONTE_CARLO must be >= 1")
    check_spectral_radius = bool(CHECK_SPECTRAL_RADIUS)

    base_results_dir, _ = prepare_results_directory(
        template_data.results_location, args.results_dir
    )

    base_data = template_data

    # Ground-truth diagonal blocks and index vectors (constant across runs)
    base_ckf = base_data.CKF_pack
    diag_A_blocks = {
        key: value.copy()
        for key, value in base_data.global_learners_pack["A_mm"].items()
    }
    diag_B_blocks = {
        key: value.copy()
        for key, value in base_data.global_learners_pack["B_mm"].items()
    }
    comp_start_vec = (
        np.asarray(base_ckf.get("comp_start_vec", base_data.comp_start_vec)).astype(int).reshape(-1)
    )
    input_start_vec = (
        np.asarray(base_ckf.get("input_start_vec", base_data.input_start_vec)).astype(int).reshape(-1)
    )
    p_vec_ref = np.asarray(base_ckf["p_vec"]).astype(int).reshape(-1)
    s_vec_ref = np.asarray(base_ckf["s_vec"]).astype(int).reshape(-1)

    A_complete = base_ckf["A_complete"]
    B_complete = base_ckf["B_complete"]
    A_total_norm = float(np.linalg.norm(A_complete, ord="fro"))
    B_total_norm = float(np.linalg.norm(B_complete, ord="fro"))
    A_true_norms: dict[str, float] = {}
    B_true_norms: dict[str, float] = {}
    dkf_avg_residual: dict[str, float] = {}
    dkf_avg_sq_residual: dict[str, float] = {}
    for i in range(base_data.num_components):
        row_start = int(comp_start_vec[i])
        row_end = row_start + int(p_vec_ref[i])
        for j in range(base_data.num_components):
            if i == j:
                continue
            col_start_a = int(comp_start_vec[j])
            col_end_a = col_start_a + int(p_vec_ref[j])
            col_start_b = int(input_start_vec[j])
            col_end_b = col_start_b + int(s_vec_ref[j])
            key = f"{i + 1}{j + 1}"
            A_block = A_complete[row_start:row_end, col_start_a:col_end_a]
            B_block = B_complete[row_start:row_end, col_start_b:col_end_b]
            A_true_norms[key] = float(np.linalg.norm(A_block, ord="fro"))
            B_true_norms[key] = float(np.linalg.norm(B_block, ord="fro"))

        comp_key = f"comp_{i + 1}"
        residual_array = base_data.local_learners_pack.get(comp_key, {}).get("X_dkf_resd")
        if residual_array is not None:
            residual_vals = np.asarray(residual_array, dtype=float).reshape(-1)
            dkf_avg_residual[f"{i + 1}"] = float(np.mean(residual_vals)) if residual_vals.size > 0 else float("nan")
            dkf_avg_sq_residual[f"{i + 1}"] = float(np.mean(residual_vals**2)) if residual_vals.size > 0 else float("nan")
        else:
            dkf_avg_residual[f"{i + 1}"] = float("nan")
            dkf_avg_sq_residual[f"{i + 1}"] = float("nan")

    # Centralized Kalman filter baseline (full system)
    from kalman_filter import KalmanFilter

    A_full = np.asarray(base_ckf["A_complete"], dtype=float)
    B_full = np.asarray(base_ckf["B_complete"], dtype=float)
    C_full = np.asarray(base_ckf["C_complete"], dtype=float)
    Q_full = np.asarray(base_ckf["Q"], dtype=float)
    R_full = np.asarray(base_ckf["R"], dtype=float)
    x_full = np.asarray(base_ckf["x0"], dtype=float).copy()
    P_full = np.asarray(base_ckf["P0"], dtype=float).copy()
    Y_full = np.asarray(base_ckf["Y"], dtype=float)
    U_full = np.asarray(base_ckf["U"], dtype=float)
    T_train = base_data.training_time
    Y_train = Y_full[:, :T_train]
    U_train = U_full[:, :T_train]

    ckf = KalmanFilter(A_full, B_full, C_full, Q_full, R_full, P_full.copy(), x_full.copy())
    residual_norms_central: list[float] = []
    ckf_residual_blocks = {f"{i + 1}": [] for i in range(base_data.num_components)}
    for t in range(T_train):
        u_prev = U_train[:, t - 1:t] if t > 0 else np.zeros((U_train.shape[0], 1))
        ckf.predict(u_prev)
        residual = ckf.residual(Y_train[:, t:t + 1])
        residual_norms_central.append(float(np.linalg.norm(residual)))
        for i in range(base_data.num_components):
            d0 = int(base_data.output_start_vec[i, 0]) if base_data.output_start_vec is not None else sum(base_data.output_size_vec[:i, 0])
            d1 = d0 + int(base_data.output_size_vec[i, 0])
            ckf_residual_blocks[f"{i + 1}"].append(float(np.linalg.norm(residual[d0:d1])))
        ckf.update(Y_train[:, t:t + 1])

    ckf_avg_residual = {
        cid: (float(np.mean(vals)) if vals else float("nan"))
        for cid, vals in ckf_residual_blocks.items()
    }

    monte_histories: list[dict] = []
    monte_metric_rows: list[list[dict[str, float]]] = []
    stop_reasons: list[str] = []
    rounds_completed: list[int] = []
    component_ids_ref: list[str] | None = None
    run_seeds: list[int | None] = []
    final_A_mn_across_runs: list[dict[str, np.ndarray]] = []
    final_B_mn_across_runs: list[dict[str, np.ndarray]] = []

    for mc_idx in range(MONTE_CARLO):
        seed_value = args.seed + mc_idx if args.seed is not None else None
        run_seeds.append(seed_value)
        if seed_value is not None:
            np.random.seed(seed_value)

        data = copy.deepcopy(base_data)

        rng = np.random.default_rng(seed_value)
        mean_theta = 0
        mean_phi = 0
        mean_A = 0
        mean_B = 0
        sigma_theta = 0.02
        sigma_phi = 0.02
        sigma_A = 0.02
        sigma_B = 0.02

        for comp_key, comp_data in data.local_learners_pack.items():
            p_m = comp_data['p_m']
            d_m = comp_data['d_m']
            comp_data['theta_init'] = rng.normal(mean_theta, sigma_theta, size=(p_m, d_m))
            comp_data['phi_init'] = rng.normal(mean_phi, sigma_phi, size=(p_m, 1))

        p_vec = data.comp_size_vec[:, 0]
        s_vec = data.input_size_vec[:, 0]
        A_mn_init = {}
        B_mn_init = {}
        for i in range(data.num_components):
            for j in range(data.num_components):
                if i == j:
                    continue
                A_mn_init[f'{i+1}{j+1}'] = rng.normal(
                    mean_A, sigma_A, size=(p_vec[i], p_vec[j])
                )
                B_mn_init[f'{i+1}{j+1}'] = rng.normal(mean_B, sigma_B, size=(p_vec[i], s_vec[j])
                )
        data.global_learners_pack['A_init_offdiag'] = A_mn_init
        data.global_learners_pack['B_init_offdiag'] = B_mn_init

        result = run_single_experiment(
            exp_idx=mc_idx,
            args=args,
            data=data,
            check_spectral_radius=check_spectral_radius,
            base_results_dir=base_results_dir,
            A_true_norms=A_true_norms,
            B_true_norms=B_true_norms,
        )

        monte_histories.append(copy.deepcopy(result["history"]))
        monte_metric_rows.append(copy.deepcopy(result["metric_rows"]))
        stop_reasons.append(result["stop_reason"])
        rounds_completed.append(result["rounds_completed"])
        final_A_mn_across_runs.append({key: value.copy() for key, value in result["final_A_mn"].items()})
        final_B_mn_across_runs.append({key: value.copy() for key, value in result["final_B_mn"].items()})

        if component_ids_ref is None:
            component_ids_ref = result["component_ids"]
        elif component_ids_ref != result["component_ids"]:
            raise ValueError("Component ids differ across Monte Carlo runs.")

    monte_payload = {
        "histories": monte_histories,
        "metric_rows": monte_metric_rows,
        "stop_reasons": stop_reasons,
        "rounds_completed": rounds_completed,
        "component_ids": component_ids_ref,
        "use_criteria": copy.deepcopy(USE_CRITERIA),
        "stop_mode": STOP_MODE,
        "monte_carlo_runs": MONTE_CARLO,
        "seeds": run_seeds,
        "final_A_mn": final_A_mn_across_runs,
        "final_B_mn": final_B_mn_across_runs,
        "diag_A_blocks": diag_A_blocks,
        "diag_B_blocks": diag_B_blocks,
        "comp_start_vec": comp_start_vec,
        "input_start_vec": input_start_vec,
        "p_vec": p_vec_ref,
        "s_vec": s_vec_ref,
        "A_total_norm": A_total_norm,
        "B_total_norm": B_total_norm,
        "ckf_avg_residual": ckf_avg_residual,
        "dkf_avg_residual": dkf_avg_residual,
        "dkf_avg_sq_residual": dkf_avg_sq_residual,
    }

    monte_path = base_results_dir / "monte_carlo_history.pkl"
    with monte_path.open("wb") as handle:
        pickle.dump(monte_payload, handle)

    print(
        f"Monte Carlo complete ({MONTE_CARLO} runs). Aggregated histories saved to {monte_path}."
    )


if __name__ == "__main__":
    main()
