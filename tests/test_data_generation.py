"""
test_data_generation.py
-----------------------
Validates data/synthetic_lti.py end to end. A dataset is generated once (module-scoped
fixture) into a temp directory from the real configs/data_linear.yaml, then checked for:

  1. Correct global and per-component matrix/trajectory shapes (both splits).
  2. A is stable (spectral radius == target) and has the declared block structure
     (diagonal blocks present; off-diagonal A_mn present exactly on the dependency edges).
  3. B has the configured coupling structure (dense => every off-diagonal block present).
  4. C is block-diagonal; Q, R are block-diagonal with the configured variances.
  5. Controls are persistently exciting on both splits.
  6. Per-component files equal the corresponding slices of the global arrays.
  7. The precomputed DKF trajectories reproduce a fresh KalmanFilter run.
  8. The saved state trajectory is consistent with A, B, U (one-step residual ~ N(0, Q)).
  9. Train and test share the same true matrices A, B, C, Q, R.

Run with:
    python -m pytest tests/test_data_generation.py -v
"""

import numpy as np
import pytest
from scipy.linalg import eigvals

from core.KalmanFilter import KalmanFilter
from data import synthetic_lti as sl

# --- Config-derived constants (single source of truth for the expected structure) ---
CFG = sl._load_config()
N = CFG["system"]["num_components"]
STATE = CFG["system"]["state_dims"]
OBS = CFG["system"]["obs_dims"]
INP = CFG["system"]["input_dims"]
DEPS = [tuple(e) for e in CFG["system"]["dependencies"]]
RHO_TARGET = CFG["matrices"]["A"]["spectral_radius_target"]
QVAR = CFG["matrices"]["Q"]
RVAR = CFG["matrices"]["R"]

P_starts = sl._block_starts(STATE)
D_starts = sl._block_starts(OBS)
S_starts = sl._block_starts(INP)
P_TOTAL, D_TOTAL, S_TOTAL = sum(STATE), sum(OBS), sum(INP)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Generate the dataset once into a temp dir; shared by all tests in this module."""
    out = tmp_path_factory.mktemp("linear_N6")
    sl.generate_and_save(str(out))
    return out


def _load(path):
    """Load a CSV as a float ndarray (delimiter=',')."""
    return np.loadtxt(path, delimiter=",")


def _block(M, r_starts, r_dims, c_starts, c_dims, i, j):
    """Return the (i, j) block of a stacked matrix M (row-block i, col-block j)."""
    return M[r_starts[i]:r_starts[i] + r_dims[i], c_starts[j]:c_starts[j] + c_dims[j]]


# ------------------------------------------------------------------ shapes

@pytest.mark.parametrize("split", ["train", "test"])
def test_global_and_component_shapes(dataset, split):
    T = CFG[split]["total_time"]
    d = dataset / split
    assert _load(d / "A_complete.csv").shape == (P_TOTAL, P_TOTAL)
    assert _load(d / "B_complete.csv").shape == (P_TOTAL, S_TOTAL)
    assert _load(d / "C_complete.csv").shape == (D_TOTAL, P_TOTAL)
    assert _load(d / "Q_complete.csv").shape == (P_TOTAL, P_TOTAL)
    assert _load(d / "R_complete.csv").shape == (D_TOTAL, D_TOTAL)
    assert _load(d / "X_complete.csv").shape == (T, P_TOTAL)
    assert _load(d / "Y_complete.csv").shape == (T, D_TOTAL)
    assert _load(d / "U_complete.csv").shape == (T, S_TOTAL)
    for m in range(N):
        c = d / f"C{m + 1}"
        assert _load(c / "X.csv").reshape(T, STATE[m]).shape == (T, STATE[m])
        assert _load(c / "Y.csv").shape == (T, OBS[m])
        assert _load(c / "U.csv").reshape(T, INP[m]).shape == (T, INP[m])
        assert _load(c / "X_dkf.csv").reshape(T, STATE[m]).shape == (T, STATE[m])
        assert _load(c / "X_dkf_pred.csv").reshape(T, STATE[m]).shape == (T, STATE[m])


# ------------------------------------------------------------------ A structure

def test_A_stable_and_block_structure(dataset):
    A = _load(dataset / "train" / "A_complete.csv")
    rho = np.max(np.abs(eigvals(A)))
    assert abs(rho - RHO_TARGET) < 1e-6                    # stability scaling hit the target
    for i in range(N):
        for j in range(N):
            blk = _block(A, P_starts, STATE, P_starts, STATE, i, j)
            # Block (i, j) is on the row-block dst=i, col-block src=j => edge (j, i).
            expect_nonzero = (i == j) or ((j, i) in DEPS)
            if expect_nonzero:
                assert np.any(blk != 0.0), f"A block ({i},{j}) should be nonzero"
            else:
                assert np.all(blk == 0.0), f"A block ({i},{j}) should be zero"


# ------------------------------------------------------------------ B structure

def test_B_dense_coupling(dataset):
    # configs/data_linear.yaml uses coupling: dense => every block (incl. off-diagonal) nonzero
    assert CFG["matrices"]["B"]["coupling"] == "dense"
    B = _load(dataset / "train" / "B_complete.csv")
    for i in range(N):
        for j in range(N):
            blk = _block(B, P_starts, STATE, S_starts, INP, i, j)
            assert np.any(blk != 0.0), f"B block ({i},{j}) should be nonzero (dense)"


# ------------------------------------------------------------------ C, Q, R

def test_C_block_diagonal(dataset):
    C = _load(dataset / "train" / "C_complete.csv")
    for i in range(N):
        for j in range(N):
            blk = _block(C, D_starts, OBS, P_starts, STATE, i, j)
            if i == j:
                assert np.any(blk != 0.0)
            else:
                assert np.all(blk == 0.0), f"C off-diagonal block ({i},{j}) must be zero"


def test_Q_R_values(dataset):
    Q = _load(dataset / "train" / "Q_complete.csv")
    R = _load(dataset / "train" / "R_complete.csv")
    # Q_m = QVAR[m] I_{P_m}, R_m = RVAR[m] I_{D_m}; all variances equal here.
    assert np.allclose(Q, QVAR[0] * np.eye(P_TOTAL))
    assert np.allclose(R, RVAR[0] * np.eye(D_TOTAL))


# ------------------------------------------------------------------ excitation

@pytest.mark.parametrize("split", ["train", "test"])
def test_persistence_of_excitation(dataset, split):
    U = _load(dataset / split / "U_complete.csv")
    is_pe, eigs = sl.check_persistence_of_excitation(U)
    assert is_pe and eigs[0] > 0.0


# ------------------------------------------------------------------ per-component consistency

def test_per_component_matches_global(dataset):
    d = dataset / "train"
    A = _load(d / "A_complete.csv"); B = _load(d / "B_complete.csv"); C = _load(d / "C_complete.csv")
    X = _load(d / "X_complete.csv"); Y = _load(d / "Y_complete.csv"); U = _load(d / "U_complete.csv")
    for m in range(N):
        c = d / f"C{m + 1}"
        p0, p1 = P_starts[m], P_starts[m] + STATE[m]
        d0, d1 = D_starts[m], D_starts[m] + OBS[m]
        s0, s1 = S_starts[m], S_starts[m] + INP[m]
        assert np.allclose(_load(c / "X.csv").reshape(X.shape[0], STATE[m]), X[:, p0:p1])
        assert np.allclose(_load(c / "Y.csv"), Y[:, d0:d1])
        assert np.allclose(_load(c / "U.csv").reshape(U.shape[0], INP[m]), U[:, s0:s1])
        assert np.allclose(_load(c / "A.csv"), A[p0:p1, p0:p1])   # A_mm only
        assert np.allclose(_load(c / "B.csv"), B[p0:p1, s0:s1])   # B_mm only
        assert np.allclose(_load(c / "C.csv"), C[d0:d1, p0:p1])   # C_mm only


# ------------------------------------------------------------------ DKF precompute

def test_dkf_precompute_reproducible(dataset):
    d = dataset / "train"
    for m in range(N):
        c = d / f"C{m + 1}"
        A_mm = _load(c / "A.csv"); B_mm = _load(c / "B.csv").reshape(STATE[m], INP[m])
        C_mm = _load(c / "C.csv"); Q_mm = _load(c / "Q.csv").reshape(STATE[m], STATE[m])
        R_mm = _load(c / "R.csv")
        Y = _load(c / "Y.csv").T                                  # (D_m, T)
        U = _load(c / "U.csv").reshape(-1, INP[m]).T              # (S_m, T)
        kf = KalmanFilter(A_mm, B_mm, C_mm, Q_mm, R_mm)
        X_dkf, X_dkf_pred = kf.run(Y, U)                          # x0=zeros, P0=I
        assert np.allclose(X_dkf.T, _load(c / "X_dkf.csv").reshape(-1, STATE[m]), atol=1e-9)
        assert np.allclose(X_dkf_pred.T, _load(c / "X_dkf_pred.csv").reshape(-1, STATE[m]), atol=1e-9)


# ------------------------------------------------------------------ dynamics consistency

def test_state_trajectory_matches_dynamics(dataset):
    d = dataset / "train"
    A = _load(d / "A_complete.csv"); B = _load(d / "B_complete.csv")
    X = _load(d / "X_complete.csv"); U = _load(d / "U_complete.csv")
    # r[t] = x[t+1] - A x[t] - B u[t] equals the process noise w[t] ~ N(0, Q).
    r = X[1:] - X[:-1] @ A.T - U[:-1] @ B.T                       # shape (T-1, P_total)
    assert np.allclose(r.mean(axis=0), 0.0, atol=5e-4)           # zero-mean noise
    assert np.allclose(np.cov(r, rowvar=False), QVAR[0] * np.eye(P_TOTAL), atol=5e-4)


# ------------------------------------------------------------------ shared matrices

def test_train_test_share_true_matrices(dataset):
    for name in ("A_complete", "B_complete", "C_complete", "Q_complete", "R_complete"):
        assert np.allclose(_load(dataset / "train" / f"{name}.csv"),
                           _load(dataset / "test" / f"{name}.csv"))
