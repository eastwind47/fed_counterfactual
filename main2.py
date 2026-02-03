from __future__ import annotations  # allow forward-referenced type hints

# ---- stdlib imports ----
import argparse                     # minimal CLI: we only expose --rounds
import copy                         # deep-copy histories across Monte Carlo runs
import csv                          # CSV writing of per-round metrics
import pickle                       # serialize snapshots and histories
from pathlib import Path            # filesystem paths (portable)
from typing import Dict, Tuple      # type aliases for readability
from types import SimpleNamespace   # lightweight container for runtime knobs

# ---- third-party ----
import numpy as np                  # numerical backbone

# ---- project modules ----
from data_retriever2 import RetrieveData        # reads config2.ini and prepares data packs
from global_learner2 import GlobalLearner2      # server with Augmented Lagrangian + IMEX
from local_learner2 import LocalLearner2        # client with IMEX updates
from spectral_radius1 import SpectralRadiusAnalyzer  # (optional) spectral diagnostics
from kalman_filter import KalmanFilter

# ======================================================================================
# Global toggles and runtime knobs (kept in-code — no CLI parsing needed)
# ======================================================================================

CHECK_SPECTRAL_RADIUS = 1  # 1 => compute spectral diagnostics before training

# ----- ρ (rho) adaptation toggles for AL (residual balancing heuristic) -----
ADAPT_RHO = False       # enable adaptive rho (False => keep rho fixed)
RHO_MU = 10.0           # imbalance threshold μ; if r_p > μ r_d => increase rho
RHO_TAU_INC = 2.0       # multiplicative factor to increase rho
RHO_TAU_DEC = 2.0       # multiplicative factor to decrease rho
RHO_MIN = 1e-6          # clamp lower bound for rho
RHO_MAX = 1e6           # clamp upper bound for rho
RHO_EVERY = 1           # adapt every k rounds
RHO_USE_A_WEIGHT = True # use ||A_mm^T d̄|| in dual residual definition
LOG_RHO_HISTORY = False # persist rho trajectory (if ADAPT_RHO=True)

# ----- Paths -----
CONFIG_PATH: str = "config2.ini"          # config file to load
RESULTS_DIR_OVERRIDE: Path | None = None  # set to Path(...) to override results dir

# ----- Server (AL & IMEX) -----
USE_AL: bool = True         # True => Augmented Lagrangian (ρ), False => plain ξ-penalty
RHO: float = 10.0           # initial AL penalty
XI: float = 0.0             # used only when USE_AL=False
USE_IMEX_SERVER: bool = True
ALPHA_A: float = 3e-3       # server step size for Â updates
ALPHA_B: float = 3e-3       # server step size for B̂ updates

# ----- Client (IMEX) -----
USE_IMEX_LOCAL: bool = True
ETA1: float = 1e-2          # θ,φ: measurement-loss term
ETA2: float = 5e-3          # θ,φ: server-induced term
GAMMA1: float = 1e-2        # φ counterpart of ETA1
GAMMA2: float = 5e-3        # φ counterpart of ETA2
TAU_THETA: float = 1.0      # IMEX weight for θ (implicit fraction)
TAU_PHI: float = 1.0        # IMEX weight for φ (implicit fraction)

# ----- Training/control -----
SAVE_EVERY: int = 5        # snapshot cadence; 0 => disable
SEED: int | None = 0        # base seed; None => nondeterministic
MONTE_CARLO: int = 5        # number of independent runs

# ======================================================================================
# Stopping configuration (edit here; no CLI required)
# ======================================================================================

# Enable the criteria you care about; STOP_MODE selects “any” vs “all”.
USE_CRITERIA = {
    "max_rounds": True,        # hard cap using MAX_ROUNDS (or --rounds if provided)
    "loss_stagnation": False,  # moving-average relative change < EPS_LOSS for PATIENCE_P windows
    "param_stagnation": False, # relative parameter change < EPS_PARAM for PATIENCE_P rounds
    "grad_small": False,       # mean grad norm < EPS_GRAD for PATIENCE_P rounds
    "residual_small": False,   # both alignment & consensus < EPS_RESID for PATIENCE_P rounds
}

STOP_MODE = "any"  # "any" => stop when any enabled criterion trips; "all" => require all

# Thresholds/patience for the criteria above
MAX_ROUNDS = 500
WINDOW_W = 5
PATIENCE_P = 3
EPS_LOSS = 1e-4
EPS_PARAM = 5e-3
EPS_GRAD = 1e-3
EPS_RESID = 1e-3

# ======================================================================================
# Early-stopping helper
# ======================================================================================

class StoppingController:
    """Manages early-stopping across multiple criteria with patience logic."""

    def __init__(self, max_rounds_limit: int | None = None) -> None:
        # Per-criterion patience counters
        self.ok_counts = {
            "loss_stagnation": 0,
            "param_stagnation": 0,
            "grad_small": 0,
            "residual_small": 0,
        }
        self.last_params: dict | None = None  # last parameters snapshot (for change tests)
        # Hard cap if max_rounds is enabled; else default to MAX_ROUNDS (unused)
        self.max_rounds_limit = max_rounds_limit if max_rounds_limit is not None else MAX_ROUNDS

    @staticmethod
    def _rel_change(new: float, old: float) -> float:
        """Compute safe relative change |new-old|/max(1,|old|)."""
        denom = max(1.0, abs(old))
        return abs(new - old) / denom

    def _loss_stagnation(self, loss_hist: list[float]) -> bool:
        """Check moving-average relative change."""
        if len(loss_hist) < WINDOW_W + 1:
            return False
        curr = sum(loss_hist[-WINDOW_W:]) / WINDOW_W
        prev = sum(loss_hist[-(2 * WINDOW_W):-WINDOW_W]) / WINDOW_W
        return self._rel_change(curr, prev) < EPS_LOSS

    def _param_stagnation(self, curr_params: dict) -> bool:
        """Check relative change in (Â,B̂,θ,φ)."""
        if self.last_params is None:
            return False

        def frob_sum(dct: dict) -> float:
            return sum(float((val ** 2).sum() ** 0.5) for val in dct.values())

        # ΔÂ (Frobenius norms aggregated)
        dA = frob_sum({k: curr_params["A_mn"][k] - self.last_params["A_mn"][k] for k in curr_params["A_mn"]})
        nA = max(1.0, frob_sum(self.last_params["A_mn"]))
        relA = dA / nA

        # ΔB̂
        dB = frob_sum({k: curr_params["B_mn"][k] - self.last_params["B_mn"][k] for k in curr_params["B_mn"]})
        nB = max(1.0, frob_sum(self.last_params["B_mn"]))
        relB = dB / nB

        # Δθ (sum of 2-norms across clients)
        dtheta = sum(
            float(np.linalg.norm(curr_params["theta"][cid] - self.last_params["theta"][cid]))
            for cid in curr_params["theta"]
        )
        ntheta = max(
            1.0,
            sum(float(np.linalg.norm(self.last_params["theta"][cid])) for cid in self.last_params["theta"]),
        )
        relT = dtheta / ntheta

        # Δφ
        dphi = sum(
            float(np.linalg.norm(curr_params["phi"][cid] - self.last_params["phi"][cid]))
            for cid in curr_params["phi"]
        )
        nphi = max(
            1.0,
            sum(float(np.linalg.norm(self.last_params["phi"][cid])) for cid in self.last_params["phi"]),
        )
        relP = dphi / nphi

        # Worst relative change across blocks
        rel = max(relA, relB, relT, relP)
        return rel < EPS_PARAM

    def _grad_small(self, grad_norms: dict[str, float]) -> bool:
        """Mean grad norm below threshold?"""
        if not grad_norms:
            return False
        mean_grad = float(sum(grad_norms.values()) / max(1, len(grad_norms)))
        return mean_grad < EPS_GRAD

    def _residual_small(self, align_norm: float, consensus_norm: float) -> bool:
        """Both residuals below threshold?"""
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
        """Advance patience counters and decide stop; returns (should_stop, reasons)."""
        satisfied: list[str] = []

        if USE_CRITERIA.get("loss_stagnation", False):
            ok = self._loss_stagnation(history["global_loss"])
            self.ok_counts["loss_stagnation"] = self.ok_counts["loss_stagnation"] + 1 if ok else 0
            if self.ok_counts["loss_stagnation"] >= PATIENCE_P:
                satisfied.append("loss_stagnation")

        if USE_CRITERIA.get("param_stagnation", False):
            ok = self._param_stagnation(curr_params)
            self.ok_counts["param_stagnation"] = self.ok_counts["param_stagnation"] + 1 if ok else 0
            if self.ok_counts["param_stagnation"] >= PATIENCE_P:
                satisfied.append("param_stagnation")

        if USE_CRITERIA.get("grad_small", False):
            ok = self._grad_small(grad_norms)
            self.ok_counts["grad_small"] = self.ok_counts["grad_small"] + 1 if ok else 0
            if self.ok_counts["grad_small"] >= PATIENCE_P:
                satisfied.append("grad_small")

        if USE_CRITERIA.get("residual_small", False):
            ok = self._residual_small(align_norm, consensus_norm)
            self.ok_counts["residual_small"] = self.ok_counts["residual_small"] + 1 if ok else 0
            if self.ok_counts["residual_small"] >= PATIENCE_P:
                satisfied.append("residual_small")

        # remember current params for next round
        self.last_params = curr_params

        # combine criteria (“any” or “all”) EXCEPT max_rounds (handled below)
        enabled = [k for k, v in USE_CRITERIA.items() if v and k != "max_rounds"]
        stop = False
        if enabled:
            if STOP_MODE.lower() == "any":
                stop = bool(satisfied)
            else:
                stop = all(name in satisfied for name in enabled)

        # hard cap on rounds (if enabled)
        if USE_CRITERIA.get("max_rounds", True) and round_idx >= self.max_rounds_limit:
            if "max_rounds" not in satisfied:
                satisfied.append("max_rounds")
            return True, satisfied

        return stop, satisfied

# ======================================================================================
# Minimal argparse (only --rounds): used iff USE_CRITERIA['max_rounds'] is True
# ======================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the augmented (AL/IMEX) counterfactual learning pipeline."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=MAX_ROUNDS,
        help="Max rounds (only used when USE_CRITERIA['max_rounds'] is enabled).",
    )
    return parser.parse_args()

# ======================================================================================
# Results directory helpers
# ======================================================================================

def prepare_results_directory(config_results: str, override: Path | None) -> tuple[Path, Path]:
    """Create base results dir and a 'snapshots' subdir. Fallback to ./results/main2 if needed."""
    base_dir = override if override is not None else Path(config_results) / "main2"
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path("results") / "main2"
        fallback.mkdir(parents=True, exist_ok=True)
        base_dir = fallback
    snapshots_dir = base_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, snapshots_dir

# ======================================================================================
# Model construction helpers
# ======================================================================================

def _shift_right_with_zeros(mat: np.ndarray) -> np.ndarray:
    """Compute [0, x[:, :-1]] along time axis; keeps same shape."""
    if mat.ndim != 2:
        raise ValueError("Expected 2D array for time series.")
    d, T = mat.shape
    if T == 0:
        return mat.copy()
    out = np.zeros_like(mat)
    if T > 1:
        out[:, 1:] = mat[:, :-1]
    return out

def _collect_diagonals_from_local(local_pack: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Assemble diagonal blocks A_mm, B_mm, C_mm from each comp_* in local_pack.
    Accepts 'A_mm'/'A', 'B_mm'/'B', 'C_mm'/'C'. If C is missing, builds a d×p
    default [I; 0] using (d,p) inferred from Y and A.
    """
    A_mm: Dict[str, np.ndarray] = {}
    B_mm: Dict[str, np.ndarray] = {}
    C_mm: Dict[str, np.ndarray] = {}

    comp_keys = [k for k in local_pack.keys() if k.startswith("comp_")]
    comp_keys_sorted = sorted(comp_keys, key=lambda k: int(k.split("_")[-1]))

    for k in comp_keys_sorted:
        idx = int(k.split("_")[-1])
        cid = f"{idx}"
        comp = local_pack[k]

        A = comp.get("A_mm", comp.get("A", None))
        B = comp.get("B_mm", comp.get("B", None))
        C = comp.get("C_mm", comp.get("C", None))

        if A is None:
            raise KeyError(f"Local pack '{k}' is missing 'A_mm' (or 'A').")
        if B is None:
            raise KeyError(f"Local pack '{k}' is missing 'B_mm' (or 'B').")

        if C is None:
            p_m = A.shape[0]
            if "Y" in comp and isinstance(comp["Y"], np.ndarray):
                d_m = comp["Y"].shape[0]
            else:
                d_m = p_m
            C = np.zeros((d_m, p_m), dtype=A.dtype)
            t = min(d_m, p_m)
            C[:t, :t] = np.eye(t, dtype=A.dtype)

        A_mm[cid] = np.array(A, copy=True)
        B_mm[cid] = np.array(B, copy=True)
        C_mm[cid] = np.array(C, copy=True)

    return A_mm, B_mm, C_mm

def _extract_offdiagonal_inits(pack: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Fetch initial off-diagonal (Â_mn, B̂_mn) from server pack.
    Accepts 'A_mn_init'/'A_init_offdiag' and 'B_mn_init'/'B_init_offdiag'.
    """
    A_candidates = ("A_mn_init", "A_init_offdiag")
    B_candidates = ("B_mn_init", "B_init_offdiag")

    A_key = next((k for k in A_candidates if k in pack), None)
    B_key = next((k for k in B_candidates if k in pack), None)

    if A_key is None or B_key is None:
        available = ", ".join(sorted(pack.keys()))
        raise KeyError(
            "global_learners_pack must contain off-diagonal init keys. "
            f"Tried {A_candidates} and {B_candidates}. Available keys: {available}"
        )

    return pack[A_key], pack[B_key]

def build_models(
    data: RetrieveData,
    args: argparse.Namespace,
    check_spectral_radius: bool = False,
) -> tuple[
    dict[str, LocalLearner2],
    GlobalLearner2,
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    SpectralRadiusAnalyzer | None,
]:
    """
    Construct local learners (one per client) and a global server, wire streams,
    and optionally create a spectral analyzer.
    """
    M = data.num_components
    T = data.training_time

    # Diagonals for server + initial off-diagonals
    A_mm, B_mm, C_mm = _collect_diagonals_from_local(data.local_learners_pack)
    A_mn_init, B_mn_init = _extract_offdiagonal_inits(data.global_learners_pack)

    # Per-component dims
    p_dims: Dict[str, int] = {m: A_mm[m].shape[0] for m in A_mm.keys()}
    s_dims: Dict[str, int] = {m: B_mm[m].shape[1] for m in B_mm.keys()}
    d_dims: Dict[str, int] = {m: C_mm[m].shape[0] for m in C_mm.keys()}

    # Build local models + stream caches
    local_models: dict[str, LocalLearner2] = {}
    u_global: dict[str, np.ndarray] = {}
    comp_streams: dict[str, dict[str, np.ndarray]] = {}
    x_dkf: dict[str, np.ndarray] = {}

    for m in range(M):
        cid = f"{m + 1}"
        comp_key = f"comp_{m + 1}"
        comp_data = data.local_learners_pack[comp_key]

        # signals
        Y: np.ndarray = comp_data["Y"].copy()    # (d_m, T)
        U: np.ndarray = comp_data["U"].copy()    # (s_m, T)

        Y_prev = _shift_right_with_zeros(Y)

        # local learner
        lm = LocalLearner2(
            A=A_mm[cid], B=B_mm[cid], C=C_mm[cid],
            eta_1=args.eta1, eta_2=args.eta2,
            gamma_1=args.gamma1, gamma_2=args.gamma2,
            use_imex_local=args.use_imex_local,
            tau_theta=args.tau_theta, tau_phi=args.tau_phi,
        )
        theta_init = comp_data.get("theta_init")
        phi_init = comp_data.get("phi_init")
        if theta_init is not None and phi_init is not None:
            lm.set_params(
                np.asarray(theta_init, dtype=float),
                np.asarray(phi_init, dtype=float).reshape(-1),
            )
        Hc_tm1 = comp_data.get("Hc_tm1")
        if Hc_tm1 is None:
            Hc_tm1 = np.zeros((p_dims[cid], T))
        x_dkf[cid] = Hc_tm1.copy()

        local_models[cid] = lm

        # cached streams
        comp_streams[cid] = {"Y": Y, "Y_prev": Y_prev, "U": U, "Hc_tm1": Hc_tm1}
        u_global[cid] = U

    # global server
    gm = GlobalLearner2(
        M=M,
        A_mm=A_mm, B_mm=B_mm, C_mm=C_mm,
        p_dims=p_dims, s_dims=s_dims, d_dims=d_dims,
        T=T,
        xi=args.xi,
        use_aug_lagrangian=bool(args.use_al),
        rho=args.rho,
        use_imex=bool(args.use_imex_server),
        alpha_a=args.alpha_a, alpha_b=args.alpha_b,
    )
    gm.set_offdiag(A_mn_init, B_mn_init)

    analyzer = SpectralRadiusAnalyzer(data, x_dkf) if bool(CHECK_SPECTRAL_RADIUS) else None
    return local_models, gm, u_global, comp_streams, analyzer

# ======================================================================================
# One round of computation (local forward → server step → local gradient step)
# ======================================================================================

def compute_server_loss(global_model: GlobalLearner2) -> Tuple[float, float, float]:
    """
    Compute per-round loss diagnostics from server caches:
      total = (||r||^2 + pen * ||d||^2) / T
      align_norm = ||r|| / T, consensus_norm = ||d|| / T
    pen = rho if AL, else xi.
    """
    T = max(global_model.T, 1)
    loss_r = 0.0
    loss_d = 0.0
    align_sum = 0.0
    cons_sum = 0.0

    pen = global_model.rho if global_model.use_aug_lagrangian else global_model.xi

    for m in range(1, global_model.M + 1):
        key = f"{m}"
        r = global_model._cache["r"][key]   # (p_m, T)
        d = global_model._cache["d"][key]   # (p_m, T)
        loss_r += float(np.sum(r * r))
        loss_d += float(np.sum(d * d))
        align_sum += float(np.linalg.norm(r))
        cons_sum += float(np.linalg.norm(d))

    total = (loss_r + pen * loss_d) / float(T)
    return total, align_sum / float(T), cons_sum / float(T)

def run_round(
    global_model: GlobalLearner2,
    local_models: dict[str, LocalLearner2],
    u_global: dict[str, np.ndarray],
    comp_streams: dict[str, dict[str, np.ndarray]],
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float], float, float]:
    """Run one round; return (global_loss, local_losses, grad_norms, residual_norms, align, consensus)."""

    # local forward passes (produce augmented predictions and residuals)
    h_aug_pred: dict[str, np.ndarray] = {}
    h_aug_est_tm1: dict[str, np.ndarray] = {}
    local_losses: dict[str, float] = {}
    residual_norms: dict[str, float] = {}

    for comp_id, lm in local_models.items():
        streams = comp_streams[comp_id]
        payload = lm.local_forward_pass(
            y_prev=streams["Y_prev"],
            y_curr=streams["Y"],
            h_c_tm1=streams["Hc_tm1"],
            u_tm1=streams["U"],
        )
        h_aug_pred[comp_id] = payload["h_aug_pred"]
        h_aug_est_tm1[comp_id] = payload["h_aug_est_tm1"]

        r_local = payload["r_local"]
        T = max(global_model.T, 1)
        local_losses[comp_id] = float(np.sum(r_local * r_local) / T)
        residual_norms[comp_id] = float(np.linalg.norm(r_local) / T)

    # server forward pass (collect stats) and apply server updates
    _ = global_model.server_forward_pass(
        h_aug_pred=h_aug_pred,
        h_c={cid: streams["Hc_tm1"] for cid, streams in comp_streams.items()},
        u=u_global,
        h_a_prev=h_aug_est_tm1,
        build_stats=True,
    )
    global_model.apply_server_updates()

    # get upstream gradients for clients (true + estimated terms if needed)
    grad_payload, grad_est_payload = global_model.get_gradients_for_clients(include_est=True)

    # local gradient updates + grad norms
    grad_norms: dict[str, float] = {}
    for comp_id, lm in local_models.items():
        grad = grad_payload[comp_id]
        grad_est = grad_est_payload[comp_id]
        lm.gradient_update(grad, grad_est)
        grad_norms[comp_id] = float(np.linalg.norm(grad) / max(global_model.T, 1))

    # server-side loss diagnostics
    total_loss, align_norm, consensus_norm = compute_server_loss(global_model)

    return total_loss, local_losses, grad_norms, residual_norms, align_norm, consensus_norm

# ======================================================================================
# Snapshot & metrics utilities
# ======================================================================================

def snapshot_state(
    round_idx: int,
    global_model: GlobalLearner2,
    local_models: dict[str, LocalLearner2],
    snapshot_dir: Path,
) -> None:
    """Persist model parameters at this round for later analysis."""
    snapshot = {
        "round": round_idx,
        "A_mn": {key: value.copy() for key, value in global_model.A_mn.items()},
        "B_mn": {key: value.copy() for key, value in global_model.B_mn.items()},
        "theta": {cid: model.get_params()[0] for cid, model in local_models.items()},
        "phi": {cid: model.get_params()[1] for cid, model in local_models.items()},
    }
    snapshot_path = snapshot_dir / f"round_{round_idx:04d}.pkl"
    with snapshot_path.open("wb") as handle:
        pickle.dump(snapshot, handle)

def write_metrics_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    """Write a CSV with per-round flattened metrics; header is union of keys across rows."""
    if not rows:
        return
    headers = sorted({key for row in rows for key in row.keys()}, key=lambda x: ("round" not in x, x))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

# ======================================================================================
# Single Monte Carlo experiment
# ======================================================================================

def run_single_experiment(
    exp_idx: int,
    data: RetrieveData,
    args_ns: SimpleNamespace,
    parsed_args: argparse.Namespace,
    base_results_dir: Path,
    check_spectral_radius: bool,
) -> dict:
    """Run one training experiment and return a dict of histories/artifacts."""

    # (Re)build models for this replicate
    local_models, global_model, u_global, comp_streams, analyzer = build_models(
        data, args_ns, check_spectral_radius
    )

    # stable ordering of component ids
    component_ids = sorted(local_models.keys(), key=lambda item: int(item))

    # True global matrices and helper indices from CKF data
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
        """Build the string key used for off-diagonal blocks."""
        return f"{m_idx}{n_idx}"

    # Pre-slice the true off-diagonal blocks for fast retrieval during training
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

    # directories for this replicate
    run_dir = base_results_dir / f"mc_{exp_idx + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # optional spectral diagnostics before training
    if analyzer is not None:
        print(f"[MC {exp_idx + 1:03d}] Spectral radius diagnostics before training:")
        analyzer.print_spectral_summary()

    # round-wise history buffers (global + per-client)
    history = {
        "round": [],
        "global_loss": [],
        "align_norm": [],
        "consensus_norm": [],
        "local_loss": {cid: [] for cid in component_ids},
        "grad_norm": {cid: [] for cid in component_ids},
        "residual_norm": {cid: [] for cid in component_ids},
        "theta_norm": {cid: [] for cid in component_ids},
        "phi_norm": {cid: [] for cid in component_ids},
        "phi_consensus_err": {cid: [] for cid in component_ids},
    }
    offdiag_keys = sorted(global_model.A_mn.keys(), key=lambda k: (len(k), k))
    history["A_abs_err"] = {key: [] for key in offdiag_keys}
    history["B_abs_err"] = {key: [] for key in offdiag_keys}
    history["A_rel_err"] = {key: [] for key in offdiag_keys}
    history["B_rel_err"] = {key: [] for key in offdiag_keys}
    metric_rows: list[dict[str, float]] = []  # flattened rows for CSV

    # Track the most recent server parameters for exporting final values later
    last_A_mn: dict[str, np.ndarray] = {key: val.copy() for key, val in global_model.A_mn.items()}
    last_B_mn: dict[str, np.ndarray] = {key: val.copy() for key, val in global_model.B_mn.items()}

    # Mean input per component (used in phi consensus diagnostics)
    u_mean = {cid: np.asarray(u_global[cid]).mean(axis=1).reshape(-1) for cid in component_ids}

    # ensure at least one stopping criterion is enabled
    enabled_criteria = [key for key, flag in USE_CRITERIA.items() if flag]
    if not enabled_criteria:
        raise RuntimeError("No stopping criteria enabled. Enable at least one entry in USE_CRITERIA.")

    # read optional hard cap from --rounds iff 'max_rounds' is enabled
    max_round_enabled = bool(USE_CRITERIA.get("max_rounds", False))
    max_round_limit = int(parsed_args.rounds if parsed_args.rounds is not None else MAX_ROUNDS) \
        if max_round_enabled else None

    # stopper state
    stopper = StoppingController(max_rounds_limit=max_round_limit)
    last_round_idx = 0
    stop_reason = ""

    # human-readable cap message
    cap_msg = (
        f"up to {max_round_limit} rounds (max_rounds enabled)"
        if max_round_enabled and max_round_limit is not None
        else "until enabled criteria are satisfied"
    )
    print(
        f"[MC {exp_idx + 1:03d}] Starting training (AL={USE_AL}, IMEX_server={USE_IMEX_SERVER}, "
        f"IMEX_local={USE_IMEX_LOCAL}) with {len(component_ids)} components, horizon {global_model.T}, "
        f"{cap_msg}."
    )

    # ---------------- main training loop ----------------
    round_idx = 0
    while True:
        round_idx += 1

        # run one round end-to-end
        (
            global_loss,
            local_losses,
            grad_norms,
            residual_norms,
            align_norm,
            consensus_norm,
        ) = run_round(global_model, local_models, u_global, comp_streams)

        # Frobenius norms of current off-diagonal estimates versus ground truth
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

        # Keep the latest parameter snapshots for final export
        last_A_mn = {key: global_model.A_mn[key].copy() for key in offdiag_keys}
        last_B_mn = {key: global_model.B_mn[key].copy() for key in offdiag_keys}

        # optional rho adaptation
        rep = None
        if bool(USE_AL) and bool(ADAPT_RHO) and (round_idx % int(RHO_EVERY) == 0):
            rep = global_model.adapt_rho(
                mu=float(RHO_MU),
                tau_inc=float(RHO_TAU_INC),
                tau_dec=float(RHO_TAU_DEC),
                rho_min=float(RHO_MIN),
                rho_max=float(RHO_MAX),
                use_A_weight=bool(RHO_USE_A_WEIGHT),
                logger=None,
            )

        # append global metrics
        history["round"].append(round_idx)
        history["global_loss"].append(global_loss)
        history["align_norm"].append(align_norm)
        history["consensus_norm"].append(consensus_norm)

        # flat row for CSV
        row = {
            "round": round_idx,
            "global_loss": global_loss,
            "align_norm": align_norm,
            "consensus_norm": consensus_norm,
        }
        row["rho"] = float(global_model.rho) if USE_AL else 0.0
        row["penalty_scalar"] = float(global_model.rho if USE_AL else XI)
        if rep is not None:
            row["rho_action"] = rep.get("action", "hold")
            row["r_p"] = rep.get("r_p", float("nan"))
            row["r_d"] = rep.get("r_d", float("nan"))

        # per-client metrics and parameter snapshots
        theta_snapshots: dict[str, np.ndarray] = {}
        phi_snapshots: dict[str, np.ndarray] = {}
        for cid in component_ids:
            theta, phi = local_models[cid].get_params()
            theta_copy = theta.copy()
            phi_copy = phi.copy()

            history["local_loss"][cid].append(local_losses[cid])
            history["grad_norm"][cid].append(grad_norms[cid])
            history["residual_norm"][cid].append(residual_norms[cid])
            theta_norm = float(np.linalg.norm(theta_copy))
            phi_norm = float(np.linalg.norm(phi_copy))
            history["theta_norm"][cid].append(theta_norm)
            history["phi_norm"][cid].append(phi_norm)

            phi_vec = phi_copy.reshape(-1)
            target = np.zeros_like(phi_vec, dtype=float)
            for peer in component_ids:
                if peer == cid:
                    continue
                key = f"{cid}{peer}"
                block = global_model.B_mn.get(key)
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

            metrics = local_models[cid]._cache.get("metrics", {})
            row[f"delta_theta_fro_{cid}"] = metrics.get("delta_theta_fro", float("nan"))
            row[f"delta_phi_{cid}"] = metrics.get("delta_phi", float("nan"))
            row[f"norm_gradx_{cid}"] = metrics.get("norm_gradx", float("nan"))
            row[f"norm_gradx_est_{cid}"] = metrics.get("norm_gradx_est", float("nan"))
            row[f"norm_r_local_{cid}"] = metrics.get("norm_r_local", float("nan"))

            theta_snapshots[cid] = theta_copy
            phi_snapshots[cid] = phi_copy

        for key in offdiag_keys:
            row[f"A_abs_err_{key}"] = A_abs_err[key]
            row[f"B_abs_err_{key}"] = B_abs_err[key]
            row[f"A_rel_err_{key}"] = A_rel_err[key]
            row[f"B_rel_err_{key}"] = B_rel_err[key]

        # store this round’s row
        metric_rows.append(row)

        # concise console line
        rho_msg = ""
        if USE_AL:
            rho_msg = f" | rho {global_model.rho:.3e}"
            if rep is not None:
                rho_msg += f" ({rep['action']})"
        print(
            f"[MC {exp_idx + 1:03d}] Round {round_idx:03d} | global loss {global_loss:.6e} | "
            f"align {align_norm:.3e} | consensus {consensus_norm:.3e}{rho_msg}"
        )

        # bundle params for change-based stopping
        curr_params = {
            "A_mn": {key: value.copy() for key, value in global_model.A_mn.items()},
            "B_mn": {key: value.copy() for key, value in global_model.B_mn.items()},
            "theta": theta_snapshots,
            "phi": phi_snapshots,
        }

        # early-stopping check
        should_stop, satisfied = StoppingController.update_and_should_stop(
            stopper,
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

        # periodic snapshots
        if SAVE_EVERY > 0 and round_idx % SAVE_EVERY == 0:
            snapshot_state(round_idx, global_model, local_models, snapshots_dir)

    # final snapshot
    snapshot_state(last_round_idx, global_model, local_models, snapshots_dir)

    # persist full training history
    history_path = run_dir / "training_history.pkl"
    with history_path.open("wb") as handle:
        pickle.dump(history, handle)

    # CSV export
    metrics_csv_path = run_dir / "training_metrics.csv"
    write_metrics_csv(metric_rows, metrics_csv_path)

    # save final couplings
    last_A_mn = {key: value.copy() for key, value in global_model.A_mn.items()}
    last_B_mn = {key: value.copy() for key, value in global_model.B_mn.items()}
    couplings_payload = {f"A_{key}": value for key, value in global_model.A_mn.items()}
    couplings_payload.update({f"B_{key}": value for key, value in global_model.B_mn.items()})
    np.savez(run_dir / "couplings_final.npz", **couplings_payload)

    # optional rho history
    if LOG_RHO_HISTORY and USE_AL:
        rho_hist_path = run_dir / "rho_history.pkl"
        with rho_hist_path.open("wb") as handle:
            pickle.dump(global_model.rho_history, handle)

    print(f"[MC {exp_idx + 1:03d}] Run complete. Artifacts saved to {run_dir}.")

    return {
        "history": history,
        "metric_rows": metric_rows,
        "stop_reason": stop_reason,
        "rounds_completed": last_round_idx,
        "component_ids": component_ids,
        "final_A_mn": last_A_mn,
        "final_B_mn": last_B_mn,
    }

# ======================================================================================
# Entry point
# ======================================================================================

def main() -> None:
    # Only --rounds is parsed; everything else is configured above.
    args = parse_args()

    # Read config once to locate results directory template
    template_data = RetrieveData(CONFIG_PATH)
    if MONTE_CARLO < 1:
        raise ValueError("MONTE_CARLO must be >= 1")
    check_spectral_radius = bool(CHECK_SPECTRAL_RADIUS)

    # Bundle runtime knobs into a SimpleNamespace (no CLI dependence)
    args_ns = SimpleNamespace(
        eta1=ETA1, eta2=ETA2,
        gamma1=GAMMA1, gamma2=GAMMA2,
        use_imex_local=USE_IMEX_LOCAL,
        tau_theta=TAU_THETA, tau_phi=TAU_PHI,
        use_al=USE_AL, rho=RHO, xi=XI,
        use_imex_server=USE_IMEX_SERVER,
        alpha_a=ALPHA_A, alpha_b=ALPHA_B,
    )

    # Prepare results directory (config-controlled unless overridden)
    base_results_dir, _ = prepare_results_directory(
        template_data.results_location, RESULTS_DIR_OVERRIDE
    )

    # Load data once; we'll mutate stochastic inits per Monte Carlo run
    base_data = template_data

    # Constant ground-truth structure (same across runs)
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
    A_total_norm = float(np.linalg.norm(base_ckf["A_complete"], ord="fro"))
    B_total_norm = float(np.linalg.norm(base_ckf["B_complete"], ord="fro"))
    dkf_avg_residual: dict[str, float] = {}
    A_true_norms: dict[str, float] = {}
    B_true_norms: dict[str, float] = {}

    for i in range(base_data.num_components):
        cid = f"{i + 1}"
        comp_key = f"comp_{cid}"
        residual_array = base_data.local_learners_pack.get(comp_key, {}).get("X_dkf_resd")
        if residual_array is not None:
            residual_vals = np.asarray(residual_array, dtype=float).reshape(-1)
            dkf_avg_residual[cid] = float(np.mean(residual_vals)) if residual_vals.size > 0 else float("nan")
        else:
            dkf_avg_residual[cid] = float("nan")
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
            A_block = base_ckf["A_complete"][row_start:row_end, col_start_a:col_end_a]
            B_block = base_ckf["B_complete"][row_start:row_end, col_start_b:col_end_b]
            A_true_norms[key] = float(np.linalg.norm(A_block, ord="fro"))
            B_true_norms[key] = float(np.linalg.norm(B_block, ord="fro"))

    # Centralized Kalman filter baseline (full system)
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
        ckf.predict(U_train[:, t:t + 1])
        residual = ckf.residual(Y_train[:, t:t + 1])
        residual_norms_central.append(float(np.linalg.norm(residual)))
        for i in range(base_data.num_components):
            d0 = int(input_start_vec[i]) if i < len(input_start_vec) else sum(s_vec_ref[:i])
            d1 = d0 + int(s_vec_ref[i])
            ckf_residual_blocks[f"{i + 1}"].append(float(np.linalg.norm(residual[d0:d1])))
        ckf.update(Y_train[:, t:t + 1])

    ckf_avg_residual = {
        cid: (float(np.mean(vals)) if vals else float("nan"))
        for cid, vals in ckf_residual_blocks.items()
    }

    # Monte Carlo aggregators
    monte_histories: list[dict] = []
    monte_metric_rows: list[list[dict[str, float]]] = []
    stop_reasons: list[str] = []
    rounds_completed: list[int] = []
    component_ids_ref: list[str] | None = None
    run_seeds: list[int | None] = []
    final_A_mn_across_runs: list[dict[str, np.ndarray]] = []
    final_B_mn_across_runs: list[dict[str, np.ndarray]] = []

    # Multiple independent runs
    for mc_idx in range(MONTE_CARLO):
        # derive per-run seed
        seed_value = SEED + mc_idx if SEED is not None else None
        run_seeds.append(seed_value)
        if seed_value is not None:
            np.random.seed(seed_value)

        # reload data (deterministic); then sample stochastic inits here
        data = copy.deepcopy(base_data)

        rng = np.random.default_rng(seed_value)
        mean_theta = -0.3
        mean_phi = -0.2
        mean_A = 0.8
        mean_B = 0.8
        sigma_theta = 0.05
        sigma_phi = 0.1
        sigma_A = 0.1
        sigma_B = 0.1

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
                B_mn_init[f'{i+1}{j+1}'] = rng.normal(
                    mean_B, sigma_B, size=(p_vec[i], s_vec[j])
                )
        data.global_learners_pack['A_init_offdiag'] = A_mn_init
        data.global_learners_pack['B_init_offdiag'] = B_mn_init

        # execute one experiment
        result = run_single_experiment(
            exp_idx=mc_idx,
            data=data,
            args_ns=args_ns,
            parsed_args=args,
            base_results_dir=base_results_dir,
            check_spectral_radius=check_spectral_radius,
        )

        # aggregate outputs
        monte_histories.append(copy.deepcopy(result["history"]))
        monte_metric_rows.append(copy.deepcopy(result["metric_rows"]))
        stop_reasons.append(result["stop_reason"])
        rounds_completed.append(result["rounds_completed"])
        final_A_mn_across_runs.append({key: value.copy() for key, value in result["final_A_mn"].items()})
        final_B_mn_across_runs.append({key: value.copy() for key, value in result["final_B_mn"].items()})

        # enforce consistent component ordering
        if component_ids_ref is None:
            component_ids_ref = result["component_ids"]
        elif component_ids_ref != result["component_ids"]:
            raise ValueError("Component ids differ across Monte Carlo runs.")

    # bundle and persist Monte Carlo summary
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
    }
    monte_path = base_results_dir / "monte_carlo_history.pkl"
    with monte_path.open("wb") as handle:
        pickle.dump(monte_payload, handle)

    print(f"Monte Carlo complete ({MONTE_CARLO} runs). Aggregated histories saved to {monte_path}.")

if __name__ == "__main__":
    main()
