"""Analyze off-diagonal B block errors and their effect on random inputs."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from data_retriever import RetrieveData

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = "config.ini"  # config file to load data packs
RESULTS_SUBDIR = "main1"  # default subfolder under results_location for v1 artifacts
TRAINING_RESULTS_DIR = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/Components_2/rebuttal/main1/mc_001" 
SNAPSHOTS_DIR = None  # set to override snapshots dir; else uses <results>/snapshots
PLOT_DIR = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/Components_2/rebuttal/count_exp_results/plots"  # set to where you want plots saved; if None uses <results>/b_block_plots
T_INPUT = 300  # length of random input sequence for the time-varying error plot
SEED = None  # RNG seed; set an int for reproducibility, None for fresh draws


def load_latest_snapshot(snapshots_dir: Path) -> Dict[str, np.ndarray]:
    """Load the latest snapshot and return the off-diagonal B_mn estimates."""
    snap_files = sorted(snapshots_dir.glob("round_*.pkl"))
    if not snap_files:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
    latest = snap_files[-1]
    with latest.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["B_mn"]


def build_true_offdiag_B(ckf: dict, p_starts: np.ndarray, s_starts: np.ndarray, p_vec: np.ndarray, s_vec: np.ndarray, M: int) -> Dict[str, np.ndarray]:
    """Slice the true B_complete into off-diagonal blocks keyed by 'mn'."""
    B_full = ckf["B_complete"]
    blocks: Dict[str, np.ndarray] = {}
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            row_start = int(p_starts[i])
            row_end = row_start + int(p_vec[i])
            col_start = int(s_starts[j])
            col_end = col_start + int(s_vec[j])
            key = f"{i+1}{j+1}"
            blocks[key] = B_full[row_start:row_end, col_start:col_end]
    return blocks


def main() -> None:
    data = RetrieveData(CONFIG_PATH)
    results_dir = Path(TRAINING_RESULTS_DIR) if TRAINING_RESULTS_DIR else Path(data.results_location) / RESULTS_SUBDIR
    snapshots_dir = Path(SNAPSHOTS_DIR) if SNAPSHOTS_DIR else results_dir / "snapshots"
    plot_dir = Path(PLOT_DIR) if PLOT_DIR else results_dir / "b_block_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Load estimated and true off-diagonal B blocks
    B_est = load_latest_snapshot(snapshots_dir)
    p_vec = data.comp_size_vec[:, 0]
    s_vec = data.input_size_vec[:, 0]
    p_starts = data.comp_start_vec[:, 0]
    s_starts = data.input_start_vec[:, 0]
    M = data.num_components
    B_true = build_true_offdiag_B(data.CKF_pack, p_starts, s_starts, p_vec, s_vec, M)

    # Frobenius norm of block errors
    block_errors = {key: float(np.linalg.norm(B_est[key] - B_true[key], ord="fro")) for key in B_true.keys()}

    # Time-varying error for random inputs
    u_blocks: Dict[str, np.ndarray] = {}
    for j in range(M):
        cid = f"{j+1}"
        u_blocks[cid] = rng.normal(0.0, 1.0, size=(int(s_vec[j]), T_INPUT))

    time_vec = np.arange(T_INPUT)
    time_errors: Dict[str, np.ndarray] = {}
    time_rel_errors: Dict[str, np.ndarray] = {}
    for key, B_true_block in B_true.items():
        m = int(key[0]) - 1
        n = int(key[1]) - 1
        u_n = u_blocks[f"{n+1}"]
        B_est_block = B_est[key]
        err_series = np.zeros(T_INPUT)
        rel_series = np.zeros(T_INPUT)
        for t in range(T_INPUT):
            u_t = u_n[:, t:t+1]
            diff = B_est_block @ u_t - B_true_block @ u_t
            err_series[t] = float(np.linalg.norm(diff))
            denom = float(np.linalg.norm(B_true_block @ u_t))
            rel_series[t] = err_series[t] / max(1e-9, denom)
        time_errors[key] = err_series
        time_rel_errors[key] = rel_series
        plt.figure(figsize=(8, 3))
        plt.plot(time_vec, err_series)
        plt.xlabel("Time")
        plt.ylabel(r"$\|B_{mn}^{est} u_n - B_{mn}^{true} u_n\|_2$")
        plt.title(f"Input-induced error for block B_{key}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"B_{key}_input_error.png")
        plt.close()

        plt.figure(figsize=(8, 3))
        plt.plot(time_vec, rel_series)
        plt.xlabel("Time")
        plt.ylabel(r"$\|B_{mn}^{est} u_n - B_{mn}^{true} u_n\|_2 / \|B_{mn}^{true} u_n\|_2$")
        plt.title(f"Relative input-induced error for block B_{key}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"B_{key}_input_rel_error.png")
        plt.close()

    # Print summary
    print("Frobenius norm errors ||B_est - B_true||_F (off-diagonal):")
    for key, val in block_errors.items():
        denom = float(np.linalg.norm(B_true[key], ord="fro"))
        rel = val / denom if denom > 1e-9 else float("nan")
        print(f"B_{key}: abs={val}, rel={rel}")

    print("\nMean relative input-induced error over time:")
    for key, series in time_rel_errors.items():
        print(f"B_{key}: mean_rel_err={float(np.mean(series))}")
    print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
