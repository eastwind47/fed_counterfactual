'''This data generation file is created to work with v1 pipeline, but dkf observes data generated only by diagonal blocks.'''
from __future__ import annotations

import os

import numpy as np
from scipy.linalg import block_diag, eigvals

from kalman_filter import KalmanFilter

# ---------------------------------------------------------------------------
# Experiment configuration (single source of truth)
# ---------------------------------------------------------------------------
# System size
N_COMP = 2
D_DIM = 8
P_DIM = 2
S_DIM = 2
TOTAL_TIME = 30000

# Random seeds
SEED = 42
LOCAL_DKF_SEED_OFFSET = 100

# Noise levels (per component)
OBS_NOISE_VAR = 1e-4
PROC_NOISE_VAR = 1e-3

# Coupling/stability knobs
DEPENDENCIES: list[tuple[int, int]] = [(0, 1)]
OFFDIAG_A_GAIN = 2
OFFDIAG_B_DENSITY = 1.0
OFFDIAG_B_GAIN = 2
STABILITY_MARGIN = 1.05

# C-matrix generation knobs
# C_GENERATION options: "random", "constant"
C_GENERATION = "constant"
# Used only when C_GENERATION == "constant": one scalar per diagonal block/client.
C_BLOCK_CONSTANTS: list[float] = [0.8, 0.8]

# Input generation knobs for federated/full-system stream (U_fed)
# FED_INPUT_MODE options: "normal", "correlated", "mean_shift"
FED_INPUT_MODE = "mean_shift"
FED_NORMAL_INPUT_MEAN = 0.25
FED_NORMAL_INPUT_STD = 0.1
FED_INPUT_LATENT_RANK: int | None = None
FED_INPUT_KIND = "multisine_ar"
FED_INPUT_AR_COEFF = 0.7
FED_INPUT_MULTISINE_K = 8
FED_MEAN_SHIFT_BASE_MEAN = 0.0
FED_MEAN_SHIFT_BASE_VAR = 0.01
# Each entry: (start_idx, end_idx_exclusive, mean, variance)
FED_MEAN_SHIFT_WINDOWS: list[tuple[int, int, float, float]] = [[500, 5000, 0.2, 0.1], [5000, 10000, 0.5, 0.1], [10000, 15000, 0.3, 0.1], [15000, 30000, 0.5, 0.1]]

# Input generation knobs for DKF/local-diagonal stream (U_dkf)
# DKF_INPUT_MODE options: "normal", "correlated", "mean_shift"
DKF_INPUT_MODE = "normal"
DKF_NORMAL_INPUT_MEAN = 2
DKF_NORMAL_INPUT_STD = 0.1
DKF_INPUT_LATENT_RANK: int | None = None
DKF_INPUT_KIND = "multisine_ar"
DKF_INPUT_AR_COEFF = 0.7
DKF_INPUT_MULTISINE_K = 8
DKF_MEAN_SHIFT_BASE_MEAN = 0.0
DKF_MEAN_SHIFT_BASE_VAR = 0.01
# Each entry: (start_idx, end_idx_exclusive, mean, variance)
DKF_MEAN_SHIFT_WINDOWS: list[tuple[int, int, float, float]] = []

# Seed offsets for input generators (so fed/dkf streams can be independent)
FED_INPUT_SEED_OFFSET = 0
DKF_INPUT_SEED_OFFSET = 10_000

# Output location
BASE_PATH = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/uai2026/data/dissimilar-dkf"


def make_positive_block(p_dim: int, rng: np.random.RandomState, eps: float = 1e-3) -> np.ndarray:
    """Return a p_dim x p_dim block with positive entries in (eps, 1]."""
    mat = rng.rand(p_dim, p_dim)
    return mat * (1.0 - eps) + eps


def make_full_block(p_dim: int, rng: np.random.RandomState, scale: float = 1.0) -> np.ndarray:
    """Return a dense signed p_dim x p_dim block."""
    return scale * rng.randn(p_dim, p_dim)


def generate_correlated_pe_inputs(
    total_time: int,
    n_u: int,
    rng: np.random.RandomState,
    latent_rank: int | None = None,
    kind: str = "multisine_ar",
    ar_coeff: float = 0.7,
    multisine_k: int = 8,
) -> np.ndarray:
    """
    Generate persistently exciting and correlated inputs.

    Output shape is (total_time, n_u).
    """
    rank = latent_rank if latent_rank is not None else max(3, n_u // 2)
    rank = min(rank, n_u)
    z_latent = np.zeros((total_time, rank))

    if "multisine" in kind:
        time_idx = np.arange(total_time)
        freqs = rng.uniform(0.002, 0.25, size=(rank, multisine_k))
        phases = rng.uniform(0, 2 * np.pi, size=(rank, multisine_k))
        amps = rng.uniform(0.5, 1.0, size=(rank, multisine_k))
        for i in range(rank):
            for k in range(multisine_k):
                z_latent[:, i] += amps[i, k] * np.sin(
                    2.0 * np.pi * freqs[i, k] * time_idx + phases[i, k]
                )

    if "ar" in kind:
        eps = rng.randn(total_time, rank)
        for t in range(1, total_time):
            z_latent[t] = ar_coeff * z_latent[t - 1] + eps[t]

    z_latent = (z_latent - z_latent.mean(axis=0, keepdims=True)) / (
        z_latent.std(axis=0, keepdims=True) + 1e-8
    )

    mixing = rng.randn(n_u, rank)
    u_full = z_latent @ mixing.T
    u_full = (u_full - u_full.mean(axis=0, keepdims=True)) / (
        u_full.std(axis=0, keepdims=True) + 1e-8
    )
    return u_full


def build_c_matrix(
    n_comp: int,
    d_dim: int,
    p_dim: int,
    rng: np.random.RandomState,
    mode: str,
    block_constants: list[float] | None,
) -> np.ndarray:
    """
    Build block-diagonal observation matrix C.

    - random: each block sampled from N(0,1)
    - constant: block m is filled with block_constants[m]
    """
    if mode == "random":
        return block_diag(*[rng.randn(d_dim, p_dim) for _ in range(n_comp)])

    if mode == "constant":
        if block_constants is None:
            raise ValueError(
                "C_GENERATION='constant' requires block_constants (one per component)."
            )
        if len(block_constants) != n_comp:
            raise ValueError(
                f"C constant mode requires {n_comp} block constants, "
                f"got {len(block_constants)}."
            )
        c_blocks = [
            np.full((d_dim, p_dim), float(block_constants[m]), dtype=float)
            for m in range(n_comp)
        ]
        return block_diag(*c_blocks)

    raise ValueError(f"Unsupported C generation mode: {mode}. Use 'random' or 'constant'.")


def generate_mean_shift_inputs(
    total_time: int,
    n_u: int,
    rng: np.random.RandomState,
    base_mean: float,
    base_var: float,
    windows: list[tuple[int, int, float, float]],
    stream_name: str,
) -> np.ndarray:
    """
    Generate piecewise Gaussian inputs with interval-wise mean/variance shifts.

    Windows use [start, end) indexing and apply to all input channels.
    Later windows overwrite earlier windows on overlap.
    """
    if base_var < 0:
        raise ValueError(f"{stream_name}: base variance must be >= 0, got {base_var}.")

    u_full = rng.normal(base_mean, np.sqrt(base_var), size=(total_time, n_u))
    for idx, window in enumerate(windows):
        if len(window) != 4:
            raise ValueError(
                f"{stream_name}: window {idx} must have 4 values "
                "(start, end, mean, variance)."
            )
        t0, t1, mean_i, var_i = window
        t0 = int(t0)
        t1 = int(t1)
        mean_i = float(mean_i)
        var_i = float(var_i)
        if t0 < 0 or t1 > total_time or t1 <= t0:
            raise ValueError(
                f"{stream_name}: invalid window {idx} = ({t0}, {t1}, {mean_i}, {var_i}) "
                f"for total_time={total_time}."
            )
        if var_i < 0:
            raise ValueError(
                f"{stream_name}: variance in window {idx} must be >= 0, got {var_i}."
            )
        u_full[t0:t1, :] = rng.normal(mean_i, np.sqrt(var_i), size=(t1 - t0, n_u))
    return u_full


def build_inputs(
    total_time: int,
    n_u: int,
    seed: int,
    mode: str,
    normal_mean: float,
    normal_std: float,
    latent_rank: int | None,
    kind: str,
    ar_coeff: float,
    multisine_k: int,
    mean_shift_base_mean: float,
    mean_shift_base_var: float,
    mean_shift_windows: list[tuple[int, int, float, float]],
    stream_name: str,
) -> np.ndarray:
    """
    Build control inputs for one stream from config knobs.

    Returns shape (total_time, n_u).
    """
    rng = np.random.RandomState(seed)
    if mode == "normal":
        if normal_std < 0:
            raise ValueError(f"{stream_name}: normal std must be >= 0, got {normal_std}.")
        return rng.normal(normal_mean, normal_std, size=(total_time, n_u))
    if mode == "correlated":
        if latent_rank is None:
            latent_rank = max(3, n_u // 2)
        return generate_correlated_pe_inputs(
            total_time=total_time,
            n_u=n_u,
            rng=rng,
            latent_rank=latent_rank,
            kind=kind,
            ar_coeff=ar_coeff,
            multisine_k=multisine_k,
        )
    if mode == "mean_shift":
        return generate_mean_shift_inputs(
            total_time=total_time,
            n_u=n_u,
            rng=rng,
            base_mean=mean_shift_base_mean,
            base_var=mean_shift_base_var,
            windows=mean_shift_windows,
            stream_name=stream_name,
        )
    raise ValueError(
        f"{stream_name}: unsupported mode '{mode}'. "
        "Use 'normal', 'correlated', or 'mean_shift'."
    )


def generate_lti_data(
    n_comp: int,
    d_dim: int,
    p_dim: int,
    s_dim: int,
    total_time: int,
    r_cov,
    q_cov=None,
    u_full=None,
    b_full=None,
    seed=None,
    dependencies=None,
    offdiag_a_gain: float = 100,
    offdiag_b_density: float = 1,
    offdiag_b_gain: float = 100,
    stability_margin: float = 1.05,
    c_generation: str = "random",
    c_block_constants: list[float] | None = None,
):
    """
    Generate full-system LTI trajectories:
      x_{t+1} = A x_t + B u_t + w_t
      y_t     = C x_t + v_t

    Returns:
      x, y, A, B, C, Q, R, u
    """
    rng = np.random.RandomState(seed)
    n_x = n_comp * p_dim
    n_y = n_comp * d_dim
    n_u = n_comp * s_dim

    # Build A with diagonal blocks + optional off-diagonal dependencies.
    a_blocks = [make_full_block(p_dim, rng, scale=1.0) for _ in range(n_comp)]
    a_mat = block_diag(*a_blocks)
    if dependencies:
        for src, dst in dependencies:
            assert 0 <= src < n_comp and 0 <= dst < n_comp and src != dst
            r0 = dst * p_dim
            r1 = (dst + 1) * p_dim
            c0 = src * p_dim
            c1 = (src + 1) * p_dim
            a_mat[r0:r1, c0:c1] = make_full_block(p_dim, rng, scale=offdiag_a_gain)
    rho = max(abs(eigvals(a_mat)))
    a_mat /= (stability_margin * rho)

    # Block-diagonal C.
    c_mat = build_c_matrix(
        n_comp=n_comp,
        d_dim=d_dim,
        p_dim=p_dim,
        rng=rng,
        mode=c_generation,
        block_constants=c_block_constants,
    )

    # Q (forced diagonal by block diagonal construction).
    if q_cov is None:
        q_mat = 0.05 * np.eye(n_x)
    elif isinstance(q_cov, (list, tuple)):
        q_blocks = []
        for q_i in q_cov:
            q_i = np.asarray(q_i)
            if q_i.ndim == 0:
                q_blocks.append(float(q_i) * np.eye(p_dim))
            elif q_i.ndim == 1:
                q_blocks.append(np.diag(q_i))
            else:
                q_blocks.append(np.diag(np.diag(q_i)))
        q_mat = block_diag(*q_blocks)
    else:
        q_cov = np.asarray(q_cov)
        if q_cov.ndim == 0:
            q_mat = float(q_cov) * np.eye(n_x)
        elif q_cov.ndim == 1:
            q_mat = np.diag(q_cov)
        else:
            q_mat = np.diag(np.diag(q_cov))
    q_mat = np.diag(np.diag(q_mat))

    # R (per-component or global).
    if isinstance(r_cov, (list, tuple)):
        r_blocks = []
        for r_i in r_cov:
            r_i = np.asarray(r_i)
            if r_i.ndim == 0:
                r_blocks.append(float(r_i) * np.eye(d_dim))
            elif r_i.ndim == 1:
                r_blocks.append(np.diag(r_i))
            else:
                r_blocks.append(r_i)
        r_mat = block_diag(*r_blocks)
    else:
        r_cov = np.asarray(r_cov)
        if r_cov.ndim == 0:
            r_mat = float(r_cov) * np.eye(n_y)
        elif r_cov.ndim == 1:
            r_mat = np.diag(r_cov)
        else:
            r_mat = r_cov
    r_mat = 0.5 * (r_mat + r_mat.T) + 1e-9 * np.eye(n_y)

    # Inputs + B.
    if u_full is None:
        u_full = rng.normal(0.0, 0.1, size=(total_time, n_u))
    else:
        u_full = np.asarray(u_full, dtype=float)

    if b_full is None:
        b_full = rng.uniform(0.05, 0.5, size=(n_x, n_u))
        mask = np.zeros_like(b_full, dtype=bool)
        for i in range(n_comp):
            for j in range(n_comp):
                r0 = i * p_dim
                r1 = (i + 1) * p_dim
                c0 = j * s_dim
                c1 = (j + 1) * s_dim
                if i == j:
                    continue
                keep = rng.rand(p_dim, s_dim) < offdiag_b_density
                mask[r0:r1, c0:c1] = ~keep
                b_full[r0:r1, c0:c1] *= offdiag_b_gain
        b_full[mask] = 0.0
    else:
        b_full = np.asarray(b_full, dtype=float)

    # Simulate full system.
    x_full = np.zeros((total_time, n_x))
    y_full = np.zeros((total_time, n_y))
    w_full = rng.multivariate_normal(np.zeros(n_x), q_mat, size=total_time)
    v_full = rng.multivariate_normal(np.zeros(n_y), r_mat, size=total_time)

    for t in range(total_time - 1):
        x_full[t + 1] = a_mat @ x_full[t] + b_full @ u_full[t] + w_full[t]
        y_full[t] = c_mat @ x_full[t] + v_full[t]
    y_full[-1] = c_mat @ x_full[-1] + v_full[-1]

    return x_full, y_full, a_mat, b_full, c_mat, q_mat, r_mat, u_full


def run_local_diag_dkf(
    a_loc: np.ndarray,
    b_loc: np.ndarray,
    c_loc: np.ndarray,
    q_loc: np.ndarray,
    r_loc: np.ndarray,
    x0_loc: np.ndarray,
    u_loc: np.ndarray,
    local_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build local diagonal-system observations and run local DKF on that stream.

    Returns:
      x_dkf_pred (total_time, p_m),
      x_dkf_est (total_time, p_m),
      y_diag (total_time, d_m)
    """
    rng = np.random.RandomState(local_seed)
    total_time = u_loc.shape[0]
    p_m = a_loc.shape[0]
    d_m = c_loc.shape[0]
    s_m = b_loc.shape[1]

    x_diag = np.zeros((total_time, p_m))
    y_diag = np.zeros((total_time, d_m))
    x_diag[0] = x0_loc.reshape(-1)

    w_loc = rng.multivariate_normal(np.zeros(p_m), q_loc, size=total_time)
    v_loc = rng.multivariate_normal(np.zeros(d_m), r_loc, size=total_time)

    for t in range(total_time - 1):
        y_diag[t] = c_loc @ x_diag[t] + v_loc[t]
        x_diag[t + 1] = a_loc @ x_diag[t] + b_loc @ u_loc[t] + w_loc[t]
    y_diag[-1] = c_loc @ x_diag[-1] + v_loc[-1]

    dkf = KalmanFilter(
        a_loc,
        b_loc,
        c_loc,
        q_loc,
        r_loc,
        q_loc.copy(),
        x0_loc.reshape(-1, 1).copy(),
    )

    x_dkf_pred = np.zeros((total_time, p_m))
    x_dkf_est = np.zeros((total_time, p_m))

    for t in range(total_time):
        u_prev = u_loc[t - 1:t].T if t > 0 else np.zeros((s_m, 1))
        dkf.predict(u_prev)
        x_dkf_pred[t] = dkf.get_state().reshape(-1)
        dkf.update(y_diag[t:t + 1].T)
        x_dkf_est[t] = dkf.get_state().reshape(-1)

    return x_dkf_pred, x_dkf_est, y_diag


def check_persistence_of_excitation(u_full: np.ndarray, eps: float = 1e-6) -> tuple[bool, np.ndarray]:
    """Basic PE check via Gramian eigenvalues."""
    gram = u_full.T @ u_full
    eig = np.linalg.eigvals(gram)
    return bool(np.all(eig.real > eps)), eig


if __name__ == "__main__":
    # Build per-component noise blocks.
    obs_noise = (OBS_NOISE_VAR * np.ones(N_COMP)).tolist()
    proc_noise = (PROC_NOISE_VAR * np.ones(N_COMP)).tolist()
    q_list = [proc * np.ones(P_DIM) for proc in proc_noise]
    r_list = [var * np.eye(D_DIM) for var in obs_noise]

    # Build two independent input streams:
    # - u_fed: used for full-system (federated) data generation and saved as U.csv.
    # - u_dkf: used only for local diagonal DKF simulations and saved as U_dkf.csv.
    u_fed = build_inputs(
        total_time=TOTAL_TIME,
        n_u=N_COMP * S_DIM,
        seed=SEED + FED_INPUT_SEED_OFFSET,
        mode=FED_INPUT_MODE,
        normal_mean=FED_NORMAL_INPUT_MEAN,
        normal_std=FED_NORMAL_INPUT_STD,
        latent_rank=FED_INPUT_LATENT_RANK,
        kind=FED_INPUT_KIND,
        ar_coeff=FED_INPUT_AR_COEFF,
        multisine_k=FED_INPUT_MULTISINE_K,
        mean_shift_base_mean=FED_MEAN_SHIFT_BASE_MEAN,
        mean_shift_base_var=FED_MEAN_SHIFT_BASE_VAR,
        mean_shift_windows=FED_MEAN_SHIFT_WINDOWS,
        stream_name="FED",
    )
    u_dkf = build_inputs(
        total_time=TOTAL_TIME,
        n_u=N_COMP * S_DIM,
        seed=SEED + DKF_INPUT_SEED_OFFSET,
        mode=DKF_INPUT_MODE,
        normal_mean=DKF_NORMAL_INPUT_MEAN,
        normal_std=DKF_NORMAL_INPUT_STD,
        latent_rank=DKF_INPUT_LATENT_RANK,
        kind=DKF_INPUT_KIND,
        ar_coeff=DKF_INPUT_AR_COEFF,
        multisine_k=DKF_INPUT_MULTISINE_K,
        mean_shift_base_mean=DKF_MEAN_SHIFT_BASE_MEAN,
        mean_shift_base_var=DKF_MEAN_SHIFT_BASE_VAR,
        mean_shift_windows=DKF_MEAN_SHIFT_WINDOWS,
        stream_name="DKF",
    )

    # Print effective config so there is no ambiguity about what is used.
    print("DataGeneration4 effective config:")
    print(f"  N={N_COMP}, D={D_DIM}, P={P_DIM}, S={S_DIM}, T={TOTAL_TIME}")
    print(
        "  coupling:",
        f"dependencies={DEPENDENCIES}, offdiag_a_gain={OFFDIAG_A_GAIN},",
        f"offdiag_b_density={OFFDIAG_B_DENSITY}, offdiag_b_gain={OFFDIAG_B_GAIN},",
        f"stability_margin={STABILITY_MARGIN}",
    )
    print(
        "  C generation:",
        f"mode={C_GENERATION}, block_constants={C_BLOCK_CONSTANTS}",
    )
    print(
        "  fed input:",
        f"mode={FED_INPUT_MODE}, normal_mean={FED_NORMAL_INPUT_MEAN}, normal_std={FED_NORMAL_INPUT_STD},",
        f"latent_rank={FED_INPUT_LATENT_RANK}, kind={FED_INPUT_KIND},",
        f"ar_coeff={FED_INPUT_AR_COEFF}, multisine_k={FED_INPUT_MULTISINE_K},",
        f"mean_shift_base=({FED_MEAN_SHIFT_BASE_MEAN}, {FED_MEAN_SHIFT_BASE_VAR}),",
        f"mean_shift_windows={FED_MEAN_SHIFT_WINDOWS}",
    )
    print(
        "  dkf input:",
        f"mode={DKF_INPUT_MODE}, normal_mean={DKF_NORMAL_INPUT_MEAN}, normal_std={DKF_NORMAL_INPUT_STD},",
        f"latent_rank={DKF_INPUT_LATENT_RANK}, kind={DKF_INPUT_KIND},",
        f"ar_coeff={DKF_INPUT_AR_COEFF}, multisine_k={DKF_INPUT_MULTISINE_K},",
        f"mean_shift_base=({DKF_MEAN_SHIFT_BASE_MEAN}, {DKF_MEAN_SHIFT_BASE_VAR}),",
        f"mean_shift_windows={DKF_MEAN_SHIFT_WINDOWS}",
    )
    print(
        "  seeds:",
        f"base={SEED}, fed_input_offset={FED_INPUT_SEED_OFFSET},",
        f"dkf_input_offset={DKF_INPUT_SEED_OFFSET}, local_dkf_seed_offset={LOCAL_DKF_SEED_OFFSET}",
    )
    print(f"  base_path={BASE_PATH}")

    x, y, A, B, C, Qmat, Rmat, u = generate_lti_data(
        n_comp=N_COMP,
        d_dim=D_DIM,
        p_dim=P_DIM,
        s_dim=S_DIM,
        total_time=TOTAL_TIME,
        r_cov=r_list,
        q_cov=q_list,
        u_full=u_fed,
        b_full=None,
        seed=SEED,
        dependencies=DEPENDENCIES,
        offdiag_a_gain=OFFDIAG_A_GAIN,
        offdiag_b_density=OFFDIAG_B_DENSITY,
        offdiag_b_gain=OFFDIAG_B_GAIN,
        stability_margin=STABILITY_MARGIN,
        c_generation=C_GENERATION,
        c_block_constants=C_BLOCK_CONSTANTS,
    )

    is_pe, eigvals_u = check_persistence_of_excitation(u)
    print("A spectral radius:", float(max(abs(eigvals(A)))))
    print("Inputs PE? ->", is_pe, "| min eig(U^T U):", float(np.min(np.real(eigvals_u))))

    # Save dataset in the same structure used by data_retriever.py.
    os.makedirs(BASE_PATH, exist_ok=True)
    base_dir = os.path.join(BASE_PATH, f"Components_{N_COMP}")
    os.makedirs(base_dir, exist_ok=True)

    np.savetxt(os.path.join(base_dir, "A_complete.csv"), A, delimiter=",")
    np.savetxt(os.path.join(base_dir, "C_complete.csv"), C, delimiter=",")
    np.savetxt(os.path.join(base_dir, "B_complete.csv"), B, delimiter=",")
    np.savetxt(os.path.join(base_dir, "Q_complete.csv"), Qmat, delimiter=",")
    np.savetxt(os.path.join(base_dir, "R_complete.csv"), Rmat, delimiter=",")
    np.savetxt(os.path.join(base_dir, "x0_complete.csv"), x[0:1, :], delimiter=",")
    np.savetxt(os.path.join(base_dir, "U_complete.csv"), u, delimiter=",")

    for i in range(N_COMP):
        comp_dir = os.path.join(base_dir, f"C{i + 1}")
        os.makedirs(comp_dir, exist_ok=True)

        x_i = x[:, i * P_DIM:(i + 1) * P_DIM]
        y_i = y[:, i * D_DIM:(i + 1) * D_DIM]
        u_i = u[:, i * S_DIM:(i + 1) * S_DIM]
        u_dkf_i = u_dkf[:, i * S_DIM:(i + 1) * S_DIM]

        a_i = A[i * P_DIM:(i + 1) * P_DIM, i * P_DIM:(i + 1) * P_DIM]
        c_i = C[i * D_DIM:(i + 1) * D_DIM, i * P_DIM:(i + 1) * P_DIM]
        b_i = B[i * P_DIM:(i + 1) * P_DIM, i * S_DIM:(i + 1) * S_DIM]
        q_i = Qmat[i * P_DIM:(i + 1) * P_DIM, i * P_DIM:(i + 1) * P_DIM]
        r_i = Rmat[i * D_DIM:(i + 1) * D_DIM, i * D_DIM:(i + 1) * D_DIM]
        x0_i = x[0:1, i * P_DIM:(i + 1) * P_DIM]

        # Save training/validation trajectories as before (full-system Y/U targets).
        np.savetxt(os.path.join(comp_dir, "X.csv"), x_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "Y.csv"), y_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "U.csv"), u_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "U_dkf.csv"), u_dkf_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "A.csv"), a_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "C.csv"), c_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "B.csv"), b_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "Q.csv"), q_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "R.csv"), r_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "x0.csv"), x0_i, delimiter=",")

        # Build local diagonal DKF states from diagonal local simulations.
        x_dkf_pred_i, x_dkf_est_i, y_dkf_i = run_local_diag_dkf(
            a_loc=a_i,
            b_loc=b_i,
            c_loc=c_i,
            q_loc=q_i,
            r_loc=r_i,
            x0_loc=x0_i.reshape(-1, 1),
            u_loc=u_dkf_i,
            local_seed=SEED + LOCAL_DKF_SEED_OFFSET + i,
        )
        np.savetxt(os.path.join(comp_dir, "Y_dkf.csv"), y_dkf_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "X_dkf_pred.csv"), x_dkf_pred_i, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "X_dkf_est.csv"), x_dkf_est_i, delimiter=",")
