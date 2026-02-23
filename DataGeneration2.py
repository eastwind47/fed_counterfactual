from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, eigvals
import matplotlib.pyplot as plt
import os

# ---------------------------
# Helpers for matrix blocks
# ---------------------------
def make_positive_block(P, rng, eps=1e-3):
    """Return a P×P matrix with strictly positive entries in (eps, 1]."""
    A = rng.rand(P, P)
    return A * (1 - eps) + eps

def make_full_block(P, rng, scale=1.0):
    """Full (signed) dense block with controlled scale."""
    return scale * rng.randn(P, P)

# ----------------------------------------
# Persistently exciting, correlated inputs
# ----------------------------------------
def generate_correlated_pe_inputs(
    T: int,
    n_u: int,
    rng: np.random.RandomState,
    latent_rank: int | None = None,
    kind: str = "multisine_ar",
    ar_coeff: float = 0.7,
    multisine_K: int = 8,
):
    """
    Build u(t) that is persistently exciting AND correlated across channels.
    We create a low-rank latent driver Z (shared across clients), then mix it
    into n_u channels with a random dense matrix W.
    """
    r = latent_rank if latent_rank is not None else max(3, n_u // 2)
    r = min(r, n_u)
    Z = np.zeros((T, r))

    if "multisine" in kind:
        t = np.arange(T)
        freqs = rng.uniform(0.002, 0.25, size=(r, multisine_K))
        phases = rng.uniform(0, 2 * np.pi, size=(r, multisine_K))
        amps = rng.uniform(0.5, 1.0, size=(r, multisine_K))
        for i in range(r):
            for k in range(multisine_K):
                Z[:, i] += amps[i, k] * np.sin(2 * np.pi * freqs[i, k] * t + phases[i, k])

    if "ar" in kind:
        eps = rng.randn(T, r)
        for i in range(1, T):
            Z[i] = ar_coeff * Z[i - 1] + eps[i]

    # Normalize each latent to unit variance
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-8)

    # Random dense mixing to n_u channels (creates cross-channel correlation)
    W = rng.randn(n_u, r)
    U = Z @ W.T

    # Channelwise standardization for numeric balance
    U = (U - U.mean(axis=0, keepdims=True)) / (U.std(axis=0, keepdims=True) + 1e-8)
    return U  # shape (T, n_u)

# ----------------------------------------
# LTI data generator with strong cross-peer excitation, but diagonal Q
# ----------------------------------------
def generate_lti_data(
    N: int,                # number of components
    D: int,                # measurement dim per component
    P: int,                # state dim per component
    S: int,                # input dim per component
    T: int,                # horizon
    R,                     # measurement noise (list or scalar/array)
    Q=None,                # process noise (list or scalar/array) -- FORCED DIAGONAL
    u=None,                # optional inputs (T, N*S)
    B=None,                # optional B (N*P, N*S)
    seed=None,
    dependencies=None,     # list of (src, dst) for off-diag A
    # ---- Cross-peer excitation knobs (unchanged) ----
    offdiag_A_gain: float = 1.5,     # strength of A off-diag blocks before stabilization
    offdiag_B_density: float = 0.4,  # probability that an off-diag B block entry is nonzero
    offdiag_B_gain: float = 0.5,     # scale of off-diag B entries
    Q_cross_gain: float = 0.0,       # IGNORED (Q is forced diagonal)
    stability_margin: float = 1.1,   # A := A / (stability_margin * rho(A)) for stability
):
    """
    Generate LTI state-space data with explicit cross-peer excitation in A/B/u,
    but with DIAGONAL process noise Q (no cross-component correlations).

      x_{t+1} = A x_t + B u_t + w_t,   y_t = C x_t + v_t

    Cross-peer levers still active:
      - A off-diagonals (dependencies + offdiag_A_gain)
      - B off-diagonals (offdiag_B_density/offdiag_B_gain)

    Note: Q_cross_gain is ignored because Q is enforced diagonal.
    """
    rng = np.random.RandomState(seed)
    n_x, n_y, n_u = N * P, N * D, N * S

    # --- Build diagonal A blocks ---
    A_blocks = [make_full_block(P, rng, scale=1.0) for _ in range(N)]
    A = block_diag(*A_blocks)

    # --- Inject A off-diagonals per dependencies (stronger than diag) ---
    if dependencies:
        for src, dst in dependencies:
            assert 0 <= src < N and 0 <= dst < N and src != dst
            A[dst * P:(dst + 1) * P, src * P:(src + 1) * P] = make_full_block(P, rng, scale=offdiag_A_gain)

    # --- Stabilize A by spectral radius (keeps relative off/diag mix) ---
    rho = max(abs(eigvals(A)))
    A /= (stability_margin * rho)

    # --- Build C as block-diagonal Gaussian (per-component) ---
    C = block_diag(*[rng.randn(D, P) for _ in range(N)])

    # --- Process-noise covariance Q (FORCED DIAGONAL) ---
    if Q is None:
        # Default small diagonal noise per-state
        q0 = 0.05
        Qmat = q0 * np.eye(n_x)
    elif isinstance(Q, (list, tuple)):
        # Per-component entries; each Qi coerced to diagonal
        Q_blocks = []
        for Qi in Q:
            Qi = np.asarray(Qi)
            if Qi.ndim == 0:
                Q_blocks.append(float(Qi) * np.eye(P))
            elif Qi.ndim == 1:
                # length-P vector -> diag
                Q_blocks.append(np.diag(Qi))
            else:
                # matrix -> keep only diagonal
                Q_blocks.append(np.diag(np.diag(Qi)))
        Qmat = block_diag(*Q_blocks)
    else:
        Q = np.asarray(Q)
        if Q.ndim == 0:
            Qmat = float(Q) * np.eye(n_x)
        elif Q.ndim == 1:
            # length-n_x vector -> diag
            Qmat = np.diag(Q)
        else:
            # matrix -> keep only diagonal
            Qmat = np.diag(np.diag(Q))

    # Ensure exact diagonal numeric form
    Qmat = np.diag(np.diag(Qmat))

    # --- Measurement-noise covariance R (per-component or global) ---
    if isinstance(R, (list, tuple)):
        R_blocks = []
        for Ri in R:
            Ri = np.array(Ri)
            if Ri.ndim == 0:
                R_blocks.append(Ri * np.eye(D))
            elif Ri.ndim == 1:
                R_blocks.append(np.diag(Ri))
            else:
                R_blocks.append(Ri)
        Rmat = block_diag(*R_blocks)
    else:
        R = np.array(R)
        if R.ndim == 0:
            Rmat = R * np.eye(n_y)
        elif R.ndim == 1:
            Rmat = np.diag(R)
        else:
            Rmat = R
    Rmat = 0.5 * (Rmat + Rmat.T) + 1e-9 * np.eye(n_y)

    # --- Inputs & B ---
    if u is None:
        u = rng.normal(0, 0.1, size=(T, n_u))
    else:
        u = np.array(u, dtype=float)

    # Build B with both diagonal and off-diagonal structure
    if B is None:
        B = np.random.uniform(low=0.05, high=0.5, size=(n_x, n_u))
        # Prepare mask: True -> zero it
        mask = np.zeros_like(B, dtype=bool)
        for i in range(N):
            for j in range(N):
                r = slice(i * P, (i + 1) * P)
                c = slice(j * S, (j + 1) * S)
                if i == j:
                    # keep diagonal block as is
                    continue
                else:
                    # random sparsity on off-diagonal blocks
                    rand_keep = (np.random.rand(P, S) < offdiag_B_density)
                    mask[r, c] = ~rand_keep
                    B[r, c] *= offdiag_B_gain
        B[mask] = 0.0
    else:
        B = np.array(B, dtype=float)

    # --- Allocate and simulate ---
    x = np.zeros((T, n_x))
    y = np.zeros((T, n_y))
    w = np.random.multivariate_normal(np.zeros(n_x), Qmat, size=T)
    v = np.random.multivariate_normal(np.zeros(n_y), Rmat, size=T)

    for t in range(T - 1):
        x[t + 1] = A @ x[t] + B @ u[t] + w[t]
        y[t] = C @ x[t] + v[t]
    y[-1] = C @ x[-1] + v[-1]

    return x, y, A, B, C, Qmat, Rmat, u

# ----------------------------------------
# Diagnostics: persistence & cross-Gramians
# ----------------------------------------
def check_persistence_of_excitation(u, eps=1e-6):
    """Basic PE check via Gramian eigenvalues."""
    G_u = u.T @ u
    evals = np.linalg.eigvals(G_u)
    is_pe = np.all(evals.real > eps)
    return bool(is_pe), evals

def summarize_cross_stats(x, u, N, P, S):
    """
    Print norms and condition numbers of cross-peer statistics:
    Σ_{x_p x_n}, Σ_{u_p u_n}, Σ_{u_p x_n}.
    """
    T = x.shape[0]
    def gram(A, B):  # time average
        return (A.T @ B) / float(T)

    for p in range(N):
        for n in range(N):
            if p == n:
                continue
            Xp = x[:, p * P:(p + 1) * P]
            Xn = x[:, n * P:(n + 1) * P]
            Up = u[:, p * S:(p + 1) * S]
            Un = u[:, n * S:(n + 1) * S]

            Sxh = gram(Xp, Xn)
            Suu = gram(Up, Un)
            Suh = gram(Up, Xn)

            def info(M):
                sval = np.linalg.svd(M, compute_uv=False)
                cond = (sval[0] / max(sval[-1], 1e-12)) if sval.size > 1 else 1.0
                return np.linalg.norm(M, 2), cond

            nxh, cxh = info(Sxh)
            nuu, cuu = info(Suu)
            nuh, cuh = info(Suh)
            print(f"[p={p} -> n={n}] ||Σ_xpxn||2={nxh:.3e} (cond~{cxh:.2f}) | "
                  f"||Σ_upun||2={nuu:.3e} (cond~{cuu:.2f}) | "
                  f"||Σ_upxn||2={nuh:.3e} (cond~{cuh:.2f})")

# ========================================
# Demo & file export (same interface)
# ========================================
if __name__ == "__main__":
    # Parameters
    N = 2    # number of clients
    D = 8    # measurement dim per client
    P = 2    # state dim per client
    S = 2    # input dim per client
    T = 30000

    obs_noise = (0.0001 * np.ones(N)).tolist()
    proc_noise = (0.001 * np.ones(N)).tolist()

    # Strong bidirectional coupling in A
    dependencies = [(0, 1)]

    # Per-component diagonal process noise (diagonal blocks)
    Q_list = [proc * np.ones(P) for proc in proc_noise]  # vector -> diag per component
    R_list = [var * np.eye(D) for var in obs_noise]

    # Generate data with strong cross-peer excitation (Q is diagonal)
    x, y, A, B, C, Qmat, Rmat, u = generate_lti_data(
        N=N, D=D, P=P, S=S, T=T,
        R=R_list, Q=Q_list,      # diagonal Q
        u=None, B=None,          # let generator create PE & off-diag B
        seed=42,
        dependencies=dependencies,
        offdiag_A_gain=1.5,
        offdiag_B_density=0.6,
        offdiag_B_gain=0.5,
        Q_cross_gain=0.0,        # ignored
        stability_margin=1.1,
    )

    # Quick checks
    np.set_printoptions(precision=3, suppress=True)
    print("A spectral radius:", float(max(abs(eigvals(A)))))
    is_pe, evals = check_persistence_of_excitation(u)
    print("Inputs PE? ->", is_pe, "| min eig(Σ_u):", float(np.min(np.real(evals))))

    print("\nCross-peer Gramian summary (time-averaged):")
    summarize_cross_stats(x, u, N=N, P=P, S=S)

    # ---------------- Save exactly as before ----------------
    base_path = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/uai2026"
    os.makedirs(base_path, exist_ok=True)
    base_dir = os.path.join(base_path, f"Components_{N}")
    os.makedirs(base_dir, exist_ok=True)

    np.savetxt(os.path.join(base_dir, "A_complete.csv"), A, delimiter=",")
    np.savetxt(os.path.join(base_dir, "C_complete.csv"), C, delimiter=",")
    np.savetxt(os.path.join(base_dir, "B_complete.csv"), B, delimiter=",")
    np.savetxt(os.path.join(base_dir, "Q_complete.csv"), Qmat, delimiter=",")
    np.savetxt(os.path.join(base_dir, "R_complete.csv"), Rmat, delimiter=",")
    np.savetxt(os.path.join(base_dir, "x0_complete.csv"), x[0:1, :], delimiter=",")
    np.savetxt(os.path.join(base_dir, "U_complete.csv"), u, delimiter=",")

    for i in range(N):
        comp_dir = os.path.join(base_dir, f"C{i+1}")
        os.makedirs(comp_dir, exist_ok=True)
        xi = x[:, i * P:(i + 1) * P]
        yi = y[:, i * D:(i + 1) * D]
        ui = u[:, i * S:(i + 1) * S]
        Ai = A[i * P:(i + 1) * P, i * P:(i + 1) * P]
        Ci = C[i * D:(i + 1) * D, i * P:(i + 1) * P]
        Bi = B[i * P:(i + 1) * P, i * S:(i + 1) * S]  # diag block for per-client file
        Qi = Qmat[i * P:(i + 1) * P, i * P:(i + 1) * P]
        Ri = Rmat[i * D:(i + 1) * D, i * D:(i + 1) * D]
        x0i = x[0:1, i * P:(i + 1) * P]

        np.savetxt(os.path.join(comp_dir, "X.csv"), xi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "Y.csv"), yi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "U.csv"), ui, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "A.csv"), Ai, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "C.csv"), Ci, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "B.csv"), Bi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "Q.csv"), Qi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "R.csv"), Ri, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "x0.csv"), x0i, delimiter=",")
