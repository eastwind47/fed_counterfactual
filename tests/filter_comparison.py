"""
filter_comparison.py
--------------------
Compares the one-step-ahead PREDICTIONS of the available state estimators against the
true state X_complete, on either the train or test split (set SPLIT below). Prints
per-component RMSE tables and saves a trajectory/error plot for one chosen component.

Predictions x_pred(t) = A x_est(t-1) + B u(t-1) are scored, not filtered posteriors: the
posterior absorbs y(t) via the measurement update, which makes the coupling-blind DKF look
deceptively close to the oracle. The prediction never sees y(t), so it honestly exercises
the cross-component coupling.

Estimators (more will be added once training exists):
  1. CKF — Centralized Kalman Filter (oracle): full A/B/C/Q/R on the full Y, U.
           Upper bound on achievable performance.
  2. DKF — Distributed Kalman Filter: each client uses only its diagonal blocks
           (A_mm, B_mm, C_mm, Q_mm, R_mm) on its own Y_m, U_m, ignoring coupling.
           This is the cooperative baseline the VFL correction will be added to.

Why it is needed:
    Quantifies how much performance the DKF loses by ignoring the inter-component
    coupling A_mn, B_mn — the gap the federated learner is meant to close.

Inputs:
    data/datasets/linear_N6/{SPLIT}/  — global A/B/C/Q/R, X_complete, Y_complete,
    U_complete, and per-component C{m+1}/ (A_mm, B_mm, C_mm, Q_mm, R_mm, Y, U).

Outputs:
    tests/filter_comparison_{SPLIT}_C{m+1}_dim{d}.png — trajectory, error-norm and
        innovation-residual panels for the chosen component/dimension.
    State-prediction RMSE and innovation-residual RMSE tables printed to stdout.

Run with:
    python tests/filter_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend — save PNG without a display
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.KalmanFilter import KalmanFilter

# ---------------------------------------------------------------------------
# Config — edit to switch split / plotted component
# ---------------------------------------------------------------------------
SPLIT     = "test"      # "train" or "test"
COMPONENT = 2           # 0-based component index for the trajectory plot
DIMENSION = 0           # state dimension within that component (0-based)
T_PLOT    = 3000        # timesteps shown in the trajectory plots
DATA_PATH = PROJECT_ROOT / "data" / "datasets" / "linear_N6"
SAVE_DIR  = Path(__file__).parent


def _load(path):
    """Load a CSV as a 2-D float array (delimiter=',')."""
    return np.atleast_2d(np.loadtxt(path, delimiter=","))


def predictions_ckf(A, B, C, Q, R, Y, U):
    """
    Centralized KF one-step-ahead predictions on the full system.

    Parameters
    ----------
    A, B, C, Q, R : ndarray
        Full stacked system matrices.
    Y : ndarray, shape (T, D_total)
        Full observation trajectory (row-major).
    U : ndarray, shape (T, S_total)
        Full control trajectory (row-major).

    Returns
    -------
    X_pred : ndarray, shape (T, P_total)
        One-step-ahead predicted states (a-priori), row-major.
    """
    kf = KalmanFilter(A, B, C, Q, R)
    _, X_pred = kf.run(Y.T, U.T)          # (P_total, T); x0=zeros, P0=I
    return X_pred.T                       # -> (T, P_total)


def predictions_dkf(A_mm, B_mm, C_mm, Q_mm, R_mm, Y_m, U_m):
    """
    Distributed (local) KF one-step-ahead predictions for a single component.

    Uses only the component's own diagonal blocks and its own Y_m, U_m — the coupling
    A_mn, B_mn is ignored.

    Parameters
    ----------
    A_mm, B_mm, C_mm, Q_mm, R_mm : ndarray
        The component's local system blocks.
    Y_m : ndarray, shape (T, D_m)
        The component's observation trajectory (row-major).
    U_m : ndarray, shape (T, S_m)
        The component's control trajectory (row-major).

    Returns
    -------
    X_pred : ndarray, shape (T, P_m)
        One-step-ahead predicted states (a-priori), row-major.
    """
    kf = KalmanFilter(A_mm, B_mm, C_mm, Q_mm, R_mm)
    _, X_pred = kf.run(Y_m.T, U_m.T)      # (P_m, T)
    return X_pred.T                       # -> (T, P_m)


def rmse(X_hat, X_true):
    """State-prediction RMSE between predicted and true state trajectories."""
    return float(np.sqrt(np.mean((X_hat - X_true) ** 2)))


def resid_rmse(X_pred, C_m, Y_m):
    """Innovation-residual RMSE sqrt(mean ||y(t) - C_m x_pred(t)||^2)."""
    r = Y_m - X_pred @ C_m.T
    return float(np.sqrt(np.mean(r ** 2)))


def resid_rmse_masked(X_pred, C_m, Y_m, mask):
    """Innovation-residual RMSE over the timesteps selected by a boolean mask."""
    r = Y_m[mask] - X_pred[mask] @ C_m.T
    return float(np.sqrt(np.mean(r ** 2)))


def shade_windows(ax, labels):
    """
    Shade contiguous mean-shift windows on an axis from a 0/1 label vector.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to shade.
    labels : ndarray (bool) or None
        1 inside a mean-shift window; no-op if None.
    """
    if labels is None:
        return
    in_win, t0, first = False, 0, True
    for t in range(len(labels) + 1):
        active = bool(labels[t]) if t < len(labels) else False
        if active and not in_win:
            t0, in_win = t, True
        elif not active and in_win:
            ax.axvspan(t0, t, color="#e08214", alpha=0.15, lw=0,
                       label="mean-shift window" if first else "_")
            in_win, first = False, False


def main():
    split_dir = DATA_PATH / SPLIT
    comp_dirs = sorted(
        [p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("C") and p.name[1:].isdigit()],
        key=lambda p: int(p.name[1:]),
    )
    N = len(comp_dirs)

    # Per-component local blocks and trajectories
    A_mm = [_load(d / "A.csv") for d in comp_dirs]
    B_mm = [_load(d / "B.csv") for d in comp_dirs]
    C_mm = [_load(d / "C.csv") for d in comp_dirs]
    Q_mm = [_load(d / "Q.csv") for d in comp_dirs]
    R_mm = [_load(d / "R.csv") for d in comp_dirs]
    Y_m  = [_load(d / "Y.csv") for d in comp_dirs]
    U_m  = [_load(d / "U.csv") for d in comp_dirs]
    state_dims = [a.shape[0] for a in A_mm]
    P_starts = [sum(state_dims[:m]) for m in range(N)]

    # Global matrices and full trajectories
    A = _load(split_dir / "A_complete.csv"); B = _load(split_dir / "B_complete.csv")
    C = _load(split_dir / "C_complete.csv"); Q = _load(split_dir / "Q_complete.csv")
    R = _load(split_dir / "R_complete.csv")
    X_complete = _load(split_dir / "X_complete.csv")   # (T, P_total)
    Y_complete = _load(split_dir / "Y_complete.csv")   # (T, D_total)
    U_complete = _load(split_dir / "U_complete.csv")   # (T, S_total)
    T = X_complete.shape[0]
    print(f"Split={SPLIT} | N={N} | T={T} | state_dims={state_dims}")

    # Mean-shift window mask (present for mean_shift test data): shading + readout.
    shift_labels = None
    labels_path = split_dir / "shift_labels.csv"
    if labels_path.exists():
        raw = np.loadtxt(labels_path)
        if raw.any():
            shift_labels = raw.astype(bool)

    # True per-component states
    X_true = [X_complete[:, P_starts[m]:P_starts[m] + state_dims[m]] for m in range(N)]

    # CKF (oracle) predictions, sliced per component
    print("Running CKF (oracle) ...")
    X_ckf_full = predictions_ckf(A, B, C, Q, R, Y_complete, U_complete)
    X_ckf = [X_ckf_full[:, P_starts[m]:P_starts[m] + state_dims[m]] for m in range(N)]

    # DKF (local) predictions per component
    print("Running DKF (local) ...")
    X_dkf = [predictions_dkf(A_mm[m], B_mm[m], C_mm[m], Q_mm[m], R_mm[m], Y_m[m], U_m[m])
             for m in range(N)]

    # --- RMSE tables ---
    print(f"\n=== State-prediction RMSE vs X_complete — split={SPLIT} ===")
    print(f"{'Comp':<8}{'CKF':>12}{'DKF':>12}{'DKF/CKF':>10}")
    print("-" * 42)
    for m in range(N):
        r_ckf, r_dkf = rmse(X_ckf[m], X_true[m]), rmse(X_dkf[m], X_true[m])
        print(f"  C{m + 1:<5}{r_ckf:>12.6f}{r_dkf:>12.6f}{r_dkf / max(r_ckf, 1e-12):>10.2f}")

    print(f"\n=== Innovation-residual RMSE ||y - C x_pred|| — split={SPLIT} ===")
    print(f"{'Comp':<8}{'CKF':>12}{'DKF':>12}")
    print("-" * 32)
    for m in range(N):
        print(f"  C{m + 1:<5}{resid_rmse(X_ckf[m], C_mm[m], Y_m[m]):>12.6f}"
              f"{resid_rmse(X_dkf[m], C_mm[m], Y_m[m]):>12.6f}")

    # In-window vs out-of-window residual RMSE (mean_shift split only): the residual
    # should rise inside windows, most sharply for the coupling-blind DKF.
    if shift_labels is not None:
        in_w, out_w = shift_labels, ~shift_labels
        print(f"\n=== Innovation-residual RMSE in/out mean-shift windows — split={SPLIT} ===")
        print(f"{'Comp':<8}{'CKF in':>10}{'CKF out':>10}{'DKF in':>10}{'DKF out':>10}")
        print("-" * 48)
        for m in range(N):
            print(f"  C{m + 1:<5}"
                  f"{resid_rmse_masked(X_ckf[m], C_mm[m], Y_m[m], in_w):>10.4f}"
                  f"{resid_rmse_masked(X_ckf[m], C_mm[m], Y_m[m], out_w):>10.4f}"
                  f"{resid_rmse_masked(X_dkf[m], C_mm[m], Y_m[m], in_w):>10.4f}"
                  f"{resid_rmse_masked(X_dkf[m], C_mm[m], Y_m[m], out_w):>10.4f}")

    # --- Plot for the chosen component/dimension ---
    m, d = COMPONENT, DIMENSION
    assert m < N and d < state_dims[m], "COMPONENT/DIMENSION out of range"
    T_p = min(T_PLOT, T)
    t = np.arange(T_p)
    lab = None if shift_labels is None else shift_labels[:T_p]   # window mask over plotted range

    err_ckf = np.linalg.norm(X_true[m] - X_ckf[m], axis=1)   # per-step error norm
    err_dkf = np.linalg.norm(X_true[m] - X_dkf[m], axis=1)
    res_ckf = np.linalg.norm(Y_m[m] - X_ckf[m] @ C_mm[m].T, axis=1)
    res_dkf = np.linalg.norm(Y_m[m] - X_dkf[m] @ C_mm[m].T, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"Filter comparison (one-step predictions) — {SPLIT.upper()} | "
                 f"Component {m + 1}, state dim {d}", fontsize=13)

    axes[0].plot(t, X_true[m][:T_p, d], color="black", lw=1.8, label="X_true")
    axes[0].plot(t, X_ckf[m][:T_p, d], color="#2166ac", lw=1.0, label="CKF (oracle)")
    axes[0].plot(t, X_dkf[m][:T_p, d], color="#d6604d", lw=1.0, ls="--", label="DKF (local)")
    shade_windows(axes[0], lab)
    axes[0].set(ylabel="state value", title=f"State one-step predictions vs truth (first {T_p})")
    axes[0].legend(loc="upper right", fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, err_ckf[:T_p], color="#2166ac", lw=1.0, label="CKF")
    axes[1].plot(t, err_dkf[:T_p], color="#d6604d", lw=1.0, ls="--", label="DKF")
    shade_windows(axes[1], lab)
    axes[1].set(ylabel=f"||X_true - X_pred|| ({state_dims[m]} dims)", title="Prediction error norm")
    axes[1].legend(loc="upper right", fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, res_ckf[:T_p], color="#2166ac", lw=1.0, label="CKF")
    axes[2].plot(t, res_dkf[:T_p], color="#d6604d", lw=1.0, ls="--", label="DKF")
    shade_windows(axes[2], lab)
    axes[2].set(xlabel="timestep", ylabel=f"||y - C x_pred|| ({C_mm[m].shape[0]} dims)",
                title="Innovation residual norm")
    axes[2].legend(loc="upper right", fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = SAVE_DIR / f"filter_comparison_{SPLIT}_C{m + 1}_dim{d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {save_path.resolve()}")


if __name__ == "__main__":
    main()
