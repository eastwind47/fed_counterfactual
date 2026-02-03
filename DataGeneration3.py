'''This script generates the data that is similar to DataGeneration.py but with normalized outputs and inputs.'''

import numpy as np
from scipy.linalg import block_diag, eigvals
import matplotlib.pyplot as plt
import os

def make_positive_block(P, rng, eps=1e-3):
    """
    Return a P×P matrix with strictly positive entries in (eps, 1].
    """
    A = rng.rand(P, P)
    return A * (1 - eps) + eps

def generate_lti_data(
    N: int,
    D: int,
    P: int,
    S: int,
    T: int,
    R,
    Q=None,
    u=None,
    B=None,
    seed=None,
    dependencies=None
):
    """
    Generate LTI state-space data:
        x_{t+1} = A x_t + B u_t + w_t
        y_t     = C x_t     + v_t

    Parameters
    ----------
    N : int
      number of components
    D : int
      measurement dimension per component
    P : int
      state dimension per component
    S : int
      input dimension per component
    T : int
      time horizon
    R : scalar or list of length N or array
      measurement-noise for each component. If list, each entry can be:
        - scalar (isotropic),
        - length-D vector (diagonal),
        - D×D matrix (full)
    Q : scalar or array, optional
      process-noise covariance (same rules as R but size N*P)
    u : array (T, N*S), optional
      inputs
    B : array (N*P, N*S), optional
      input gain
    dependencies : list of (src, dst) tuples, optional
      directed edges between components (0-based indices). For each (i,j),
      the block A[j,i] will be nonzero positive.

    Returns
    -------
    x : (T, N*P) state trajectory
    y : (T, N*D) measurements
    A,C,B : system matrices
    """
    rng = np.random.RandomState(seed)
    n_x, n_y, n_u = N*P, N*D, N*S

    # --- Build diagonal blocks of A ---
    A_blocks = [make_positive_block(P, rng) for _ in range(N)]
    A = block_diag(*A_blocks)

    # --- Inject inter-component dependencies ---
    if dependencies:
        for src, dst in dependencies:
            assert 0 <= src < N and 0 <= dst < N and src != dst
            A[dst*P:(dst+1)*P, src*P:(src+1)*P] = make_positive_block(P, rng)

    # --- Scale A to be stable ---
    rho = max(abs(eigvals(A)))
    A /= (1.1 * rho)

    # --- Build C as block-diagonal Gaussian ---
    C = block_diag(*[rng.randn(D, P) for _ in range(N)])

    # --- Process-noise covariance Q ---
    # if Q is None:
    #     Qmat = np.zeros((n_x, n_x))
    # else:
    #     Q = np.array(Q)
    #     if Q.ndim == 0:
    #         Qmat = Q * np.eye(n_x)
    #     elif Q.ndim == 1:
    #         Qmat = np.diag(Q)
    #     else:
    #         Qmat = Q

    # --- Process-noise covariance Q ---
    if Q is None:
        Qmat = np.zeros((n_x, n_x))
    elif isinstance(Q, (list, tuple)):
        Q_blocks = []
        for Qi in Q:
            Qi = np.array(Qi)
            if Qi.ndim == 0:
                Q_blocks.append(Qi * np.eye(P))
            elif Qi.ndim == 1:
                Q_blocks.append(np.diag(Qi))
            else:
                Q_blocks.append(Qi)
        Qmat = block_diag(*Q_blocks)
    else:
        Q = np.array(Q)
        if Q.ndim == 0:
            Qmat = Q * np.eye(n_x)
        elif Q.ndim == 1:
            Qmat = np.diag(Q)
        else:
            Qmat = Q

    # --- Measurement-noise covariance R ---
    # allow per-component R
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

    # --- Inputs & B ---
    if u is None:
        u = np.zeros((T, n_u))
    if B is None:
        B = np.zeros((n_x, u.shape[1]))

    # --- Allocate & draw noise ---
    x = np.zeros((T, n_x))

    # Set initial state
    # x[0, :] = 3

    y = np.zeros((T, n_y))
    w = rng.multivariate_normal(np.zeros(n_x), Qmat, size=T)
    v = rng.multivariate_normal(np.zeros(n_y), Rmat, size=T)

    # --- Simulate ---
    for t in range(T-1):
        x[t+1] = A @ x[t] + B @ u[t] + w[t]
        y[t]   = C @ x[t] + v[t]
    y[-1] = C @ x[-1] + v[-1]

    return x, y, A, B, C, Qmat, Rmat

# ------------------------------
# N: Number of clients
# D: Dim. of data per client
# P: Dim. of states per client
# T: Total time horizon
# R: Noise covariance for data. Array of dimension ND. Here you can control the noise covariance for each client
# Q: Noise covariance for states.
# ------------------------------


def check_persistence_of_excitation(u):
    """
    Check if the input signal u satisfies the Persistence of Excitation (PE) condition.

    Parameters
    ----------
    u : ndarray
        Input signal of shape (T, m), where T is the time horizon and m is the input dimension.

    Returns
    -------
    is_pe : bool
        True if the PE condition is satisfied, False otherwise.
    eigenvalues : ndarray
        Eigenvalues of the input Gramian matrix.
    """
    # Compute the input Gramian
    G_u = u.T @ u  # Equivalent to summing u(t) @ u(t).T over time

    # Compute eigenvalues of the Gramian
    eigenvalues = np.linalg.eigvals(G_u)

    # Check if all eigenvalues are positive
    is_pe = np.all(eigenvalues > 1e-6)  # Use a small threshold to account for numerical precision

    return is_pe, eigenvalues


if __name__ == "__main__":
    # Define parameters
    N = 2  # Number of clients
    D = 8  # Dimension of data per client
    P = 2  # Dimension of states per client
    S = 2  # Dimension of inputs per client
    T = 30000  # Total time horizon

    # Define variances for each client
    obs_noise = (0.0001 * np.ones(N)).tolist()  # Variances for client 0 and client 1

    # Define process noises for each client
    proc_noise = (0.001 * np.ones(N)).tolist()  # Process noise for client 0 and client 1

    # Define directed dependencies (e.g., 0→1)
    dependencies = [(0, 1)]

    # Process noise covariance (same across all clients) (Choose None for no process noise)
    Q = [proc * np.eye(P) for proc in proc_noise]

    # Generate R blocks automatically based on variances
    R = [var * np.eye(D) for var in obs_noise]

    # Define windows of mean-shifts (optional)
    windows = [
        (1000, 2000,  5.0),
        (3000, 5000, -3.0),
        (7000, 9000,  2.0),
        (12000, 18000, 1.0),
        (25000, 26000, -2.0)
    ]

    # Build input u
    n_x = N * P
    n_u = N * S

    # # Option 1: Window-based inputs (commented out)
    # u = np.zeros((T, n_u))
    # for t0, t1, mag in windows:
    #     u[t0:t1, :] = mag

    # Option 2: Sample u from normal distribution
    u = np.random.normal(0, 1, size=(T, n_u))

    # Generate B matrix with sparsity pattern
    B = np.random.uniform(low=0.05, high=0.5, size=(n_x, n_u))
    B[np.random.rand(*B.shape) > 1] = 0  # 30% sparsity

    # Generate LTI data
    x, y, A, B, C, Qmat, Rmat = generate_lti_data(
        N=N, D=D, P=P, S=S, T=T,
        R=R, Q=Q,
        u=u, B=B,
        seed=42,
        dependencies=dependencies
    )

    # print("x_complete shape = ", x.shape)
    # print("y_complete shape = ", y.shape)
    # print("B matrix:\n", B_used)
    # print A
    np.set_printoptions(precision=3, suppress=True)
    # print("A matrix:\n", A)

    # print('initial Q shape = ', Qmat.shape)
    ## plot
    time = np.arange(T)
    # time = np.arange(10)
    fig, ax = plt.subplots(2,1, figsize=(12,6), sharex=True)
    
    # for i in range(n_x):
    #     ax[0].plot(time[:100], x[:100,i], label=f'x[{i}]')
    # # for s,e,_ in windows:
    # #     ax[0].axvspan(s, e, color='gray', alpha=0.2)
    # ax[0].set_title("State trajectories")
    # ax[0].legend(fontsize='small', ncol=3)
    
    # # for j in range(N*D):
    # #     ax[1].plot(time, y[:,j], label=f'y[{j}]')
    # # for s,e,_ in windows:
    # #     ax[1].axvspan(s, e, color='gray', alpha=0.2)
    # # ax[1].set_title("Measurement trajectories")
    # # ax[1].legend(fontsize='small', ncol=3)
    
    # plt.xlabel("Time step")
    # plt.tight_layout()
    # plt.show()

    # --- Normalize inputs/outputs for training ---
    eps = 1e-8  # prevent division by zero
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + eps
    u_mean = u.mean(axis=0)
    u_std = u.std(axis=0) + eps

    y_norm = (y - y_mean) / y_std
    u_norm = (u - u_mean) / u_std

    # Normalize measurement covariance for use with normalized outputs
    S_inv = np.diag(1.0 / y_std)
    Rmat_normalized = S_inv @ Rmat @ S_inv

    # Custom base path for saving files
    base_path = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/norm_comp_2"  # Replace with your desired path
    os.makedirs(base_path, exist_ok=True)  # Ensure the base path exists

    # Directory structure
    base_dir = f"Components_{N}"
    base_dir = os.path.join(base_path, base_dir)

    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Save full matrices
    np.savetxt(os.path.join(base_dir, "A_complete.csv"), A, delimiter=",")
    np.savetxt(os.path.join(base_dir, "C_complete.csv"), C, delimiter=",")
    np.savetxt(os.path.join(base_dir, "B_complete.csv"), B, delimiter=",")
    np.savetxt(os.path.join(base_dir, "Q_complete.csv"), Qmat, delimiter=",")
    # Save normalized versions under pipeline-friendly names and keep raw with _raw suffix
    np.savetxt(os.path.join(base_dir, "R_complete.csv"), Rmat_normalized, delimiter=",")
    np.savetxt(os.path.join(base_dir, "R_complete_raw.csv"), Rmat, delimiter=",")
    np.savetxt(os.path.join(base_dir, "x0_complete.csv"), x[0:1,:], delimiter=",")
    np.savetxt(os.path.join(base_dir, "U_complete.csv"), u_norm, delimiter=",")
    np.savetxt(os.path.join(base_dir, "U_complete_raw.csv"), u, delimiter=",")
    np.savetxt(os.path.join(base_dir, "Y_complete.csv"), y_norm, delimiter=",")
    np.savetxt(os.path.join(base_dir, "Y_complete_raw.csv"), y, delimiter=",")
    np.savetxt(os.path.join(base_dir, "y_mean.csv"), y_mean[None, :], delimiter=",")
    np.savetxt(os.path.join(base_dir, "y_std.csv"), y_std[None, :], delimiter=",")
    np.savetxt(os.path.join(base_dir, "u_mean.csv"), u_mean[None, :], delimiter=",")
    np.savetxt(os.path.join(base_dir, "u_std.csv"), u_std[None, :], delimiter=",")

    # Per-component files
    for i in range(N):
        comp_dir = os.path.join(base_dir, f"C{i+1}")
        os.makedirs(comp_dir, exist_ok=True)

        # Slice data for the current component
        xi = x[:, i*P:(i+1)*P]
        yi = y[:, i*D:(i+1)*D]
        ui = u[:, i*S:(i+1)*S]
        yi_norm = y_norm[:, i*D:(i+1)*D]
        ui_norm = u_norm[:, i*S:(i+1)*S]
        # print("xi shape = ", xi.shape)
        # print("yi shape =  ", yi.shape)
        # print("ui shape = ", ui.shape)
        Ai = A[i*P:(i+1)*P, i*P:(i+1)*P]
        Ci = C[i*D:(i+1)*D, i*P:(i+1)*P]
        Bi = B[i*P:(i+1)*P, i*S:(i+1)*S]  # Per-client B matrix
        # print("Ci shape = ", Ci.shape)
        # print("Bi shape = ", Bi.shape)
        # Qi = Q[i*P:(i+1)*P, i*P:(i+1)*P]
        # Ri = R[i*D:(i+1)*D, i*D:(i+1)*D]
        Qi = Qmat[i*P:(i+1)*P, i*P:(i+1)*P]
        Ri = Rmat[i*D:(i+1)*D, i*D:(i+1)*D]
        Ri_norm = Rmat_normalized[i*D:(i+1)*D, i*D:(i+1)*D]
        x0i = x[0:1, i*P:(i+1)*P]
        # print("x0i shape = ", x0i.shape)

        # Save CSVs
        np.savetxt(os.path.join(comp_dir, "X.csv"), xi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "Y.csv"), yi_norm, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "Y_raw.csv"), yi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "U.csv"), ui_norm, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "U_raw.csv"), ui, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "A.csv"), Ai, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "C.csv"), Ci, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "B.csv"), Bi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "Q.csv"), Qi, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "R.csv"), Ri_norm, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "R_raw.csv"), Ri, delimiter=",")
        np.savetxt(os.path.join(comp_dir, "x0.csv"), x0i, delimiter=",")

    # Print A matrix
    np.set_printoptions(precision=3, suppress=True)
    print("A matrix:\n", A)
    print("spectral radius of A:", max(abs(eigvals(A))))

    # # Plot state trajectories
    # time = np.arange(T)
    # fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # for i in range(n_x):
    #     ax[0].plot(time, x[:, i], label=f'x[{i}]')
    # ax[0].set_title("State trajectories")
    # ax[0].legend(fontsize='small', ncol=3)

    # plt.xlabel("Time step")
    # plt.tight_layout()
    # plt.show()

    # # Example usage
    # # Assuming `u` is the input signal of shape (T, n_x)
    # is_pe, eigenvalues = check_persistence_of_excitation(u)

    # print("Is the system satisfying the PE condition?", is_pe)
    # print("Eigenvalues of the input Gramian:", eigenvalues)


