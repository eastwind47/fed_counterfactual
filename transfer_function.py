"""Compute discrete-time transfer functions for true vs estimated v1 models.

G(z) = C (z I - A)^{-1} B, with z = exp(j * omega).

Outputs (default under the couplings file directory / transfer_function):
- transfer_function_data.npz (G_true, G_est, omega grid, errors, singular values)
- transfer_function_error.csv (omega, Frobenius error)
- transfer_function_poles.csv (poles of A_true and A_est)
- transfer_function_zeros.json (per-SISO zeros for true and estimated)
- transfer_function_coefficients.json (per-entry numerator/denominator coefficients)
- plots/*.pdf (error curve, sigma_max curve)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from data_retriever import RetrieveData

# ------------------------------------------------------------------
# Optional override: set this to the full path of couplings_final.npz
# Example:
# est_data_loc = "/path/to/results/main1/mc_001/couplings_final.npz"
# ------------------------------------------------------------------
est_data_loc = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/norm_comp_2/Components_2/rebuttal/main1/mc_002/couplings_final.npz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute discrete-time transfer functions for v1 pipeline (true vs estimated)."
    )
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to config.ini used for training (for true matrices and results paths).",
    )
    parser.add_argument(
        "--couplings-path",
        type=Path,
        default=None,
        help=(
            "Path to couplings_final.npz. If omitted, will look under "
            "<results_location>/main1/couplings_final.npz or latest mc_* subfolder."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Override results root directory (the folder that contains 'main1'). "
            "If provided, couplings are searched under <results-dir>/main1."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save transfer-function outputs. Default: <couplings_dir>/transfer_function",
    )
    parser.add_argument(
        "--num-freq",
        type=int,
        default=200,
        help="Number of frequency points in [0, pi].",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=np.pi,
        help="Maximum omega for the frequency grid (default pi).",
    )
    parser.add_argument(
        "--tf-variable",
        default="s",
        help="Symbol for transfer polynomials in outputs (e.g., s for continuous-time, z for discrete-time).",
    )
    return parser.parse_args()


def _resolve_results_root(config_path: str, override: Path | None) -> Path:
    if override is not None:
        return override
    data = RetrieveData(config_path)
    return Path(data.results_location).expanduser()


def _find_couplings_file(
    config_path: str,
    couplings_path: Path | None,
    results_root: Path,
) -> Path:
    if couplings_path is not None:
        if not couplings_path.exists():
            raise FileNotFoundError(f"Couplings file not found: {couplings_path}")
        return couplings_path

    # Default: <results_root>/main1/couplings_final.npz
    base = results_root / "main1"
    candidate = base / "couplings_final.npz"
    if candidate.exists():
        return candidate

    # If not found, search in mc_* subfolders (pick latest lexicographically)
    mc_runs = sorted([p for p in base.glob("mc_*") if p.is_dir()])
    if mc_runs:
        latest = mc_runs[-1]
        alt = latest / "couplings_final.npz"
        if alt.exists():
            return alt

    raise FileNotFoundError(
        "Could not find couplings_final.npz. "
        "Provide --couplings-path or verify results directory."
    )


def _assemble_A_matrix(
    diag_blocks: Dict[str, np.ndarray],
    offdiag_blocks: Dict[str, np.ndarray],
    comp_start_vec: np.ndarray,
    p_vec: np.ndarray,
) -> np.ndarray:
    total_p = int(np.sum(p_vec))
    full = np.zeros((total_p, total_p), dtype=float)
    M = len(p_vec)
    for m in range(M):
        r0 = int(comp_start_vec[m])
        r1 = r0 + int(p_vec[m])
        diag_key = f"{m + 1}{m + 1}"
        if diag_key in diag_blocks:
            full[r0:r1, r0:r1] = np.asarray(diag_blocks[diag_key])
        for n in range(M):
            if m == n:
                continue
            key = f"{m + 1}{n + 1}"
            block = offdiag_blocks.get(key)
            if block is None:
                continue
            c0 = int(comp_start_vec[n])
            c1 = c0 + int(p_vec[n])
            full[r0:r1, c0:c1] = np.asarray(block)
    return full


def _assemble_B_matrix(
    diag_blocks: Dict[str, np.ndarray],
    offdiag_blocks: Dict[str, np.ndarray],
    comp_start_vec: np.ndarray,
    input_start_vec: np.ndarray,
    p_vec: np.ndarray,
    s_vec: np.ndarray,
) -> np.ndarray:
    total_p = int(np.sum(p_vec))
    total_s = int(np.sum(s_vec))
    full = np.zeros((total_p, total_s), dtype=float)
    M = len(p_vec)
    for m in range(M):
        r0 = int(comp_start_vec[m])
        r1 = r0 + int(p_vec[m])
        diag_key = f"{m + 1}{m + 1}"
        if diag_key in diag_blocks:
            c0 = int(input_start_vec[m])
            c1 = c0 + int(s_vec[m])
            full[r0:r1, c0:c1] = np.asarray(diag_blocks[diag_key])
        for n in range(M):
            if m == n:
                continue
            key = f"{m + 1}{n + 1}"
            block = offdiag_blocks.get(key)
            if block is None:
                continue
            c0 = int(input_start_vec[n])
            c1 = c0 + int(s_vec[n])
            full[r0:r1, c0:c1] = np.asarray(block)
    return full


def _frequency_response(A: np.ndarray, B: np.ndarray, C: np.ndarray, omega: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    d = C.shape[0]
    m = B.shape[1]
    I = np.eye(n, dtype=complex)
    G = np.zeros((omega.size, d, m), dtype=complex)
    for idx, w in enumerate(omega):
        z = np.exp(1j * w)
        X = np.linalg.solve(z * I - A, B)
        G[idx] = C @ X
    return G


def _singular_values(G: np.ndarray) -> np.ndarray:
    # G: (N, d, m) -> return (N, min(d,m)) singular values
    N = G.shape[0]
    r = min(G.shape[1], G.shape[2])
    svals = np.zeros((N, r), dtype=float)
    for i in range(N):
        svals[i] = np.linalg.svd(G[i], compute_uv=False)
    return svals


def _compute_siso_zeros(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> Dict[str, list[list[float]]]:
    """Compute per-SISO zeros using ss2tf.

    Returns dict: key "y{out}_u{in}" -> list of [real, imag] pairs.
    """
    d = C.shape[0]
    m = B.shape[1]
    D = np.zeros((d, m), dtype=float)
    zeros: Dict[str, list[list[float]]] = {}

    for j in range(m):
        num, den = signal.ss2tf(A, B, C, D, input=j)
        for i in range(d):
            num_ij = np.asarray(num[i], dtype=float)
            key = f"y{i + 1}_u{j + 1}"
            if np.allclose(num_ij, 0.0):
                zeros[key] = []
                continue
            roots = np.roots(num_ij)
            zeros[key] = [[float(r.real), float(r.imag)] for r in roots]
    return zeros


def _trim_leading_zeros(coeffs: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    coeffs = np.asarray(coeffs).reshape(-1)
    if coeffs.size == 0:
        return np.array([0.0], dtype=float)
    nz_idx = np.where(np.abs(coeffs) > atol)[0]
    if nz_idx.size == 0:
        return np.array([0.0], dtype=float)
    trimmed = coeffs[int(nz_idx[0]):]
    return np.real_if_close(trimmed, tol=1000).astype(float)


def _poly_to_string(coeffs: np.ndarray, var: str) -> str:
    c = _trim_leading_zeros(coeffs)
    deg = c.size - 1
    terms: list[str] = []
    for k, val in enumerate(c):
        if abs(val) < 1e-12:
            continue
        power = deg - k
        coef = float(val)
        coef_str = f"{coef:.10g}"
        if power == 0:
            term = coef_str
        elif power == 1:
            term = f"{var}" if np.isclose(coef, 1.0) else (f"-{var}" if np.isclose(coef, -1.0) else f"{coef_str}*{var}")
        else:
            term = f"{var}^{power}" if np.isclose(coef, 1.0) else (f"-{var}^{power}" if np.isclose(coef, -1.0) else f"{coef_str}*{var}^{power}")
        terms.append(term)
    if not terms:
        return "0"
    return " + ".join(terms).replace("+ -", "- ")


def _compute_transfer_polynomials_scipy(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    tf_variable: str,
    *,
    fallback_reason: str | None = None,
) -> dict:
    d = C.shape[0]
    m = B.shape[1]
    D = np.zeros((d, m), dtype=float)
    entries: Dict[str, dict] = {}

    for j in range(m):
        num, den = signal.ss2tf(A, B, C, D, input=j)
        den = _trim_leading_zeros(np.asarray(den, dtype=float))
        for i in range(d):
            key = f"y{i + 1}_u{j + 1}"
            num_ij = _trim_leading_zeros(np.asarray(num[i], dtype=float))
            num_expr = _poly_to_string(num_ij, tf_variable)
            den_expr = _poly_to_string(den, tf_variable)
            entries[key] = {
                "numerator_coeffs": [float(v) for v in num_ij],
                "denominator_coeffs": [float(v) for v in den],
                "expression": f"({num_expr}) / ({den_expr})",
            }

    matrix_expression = [
        [entries[f"y{i + 1}_u{j + 1}"]["expression"] for j in range(m)] for i in range(d)
    ]
    payload = {
        "method": "scipy.signal.ss2tf",
        "shape": {"outputs": int(d), "inputs": int(m)},
        "entries": entries,
        "matrix_expression": matrix_expression,
    }
    if fallback_reason is not None:
        payload["fallback_reason"] = fallback_reason
    return payload


def _compute_transfer_polynomials_control(A: np.ndarray, B: np.ndarray, C: np.ndarray, tf_variable: str) -> dict:
    import control as ct  # type: ignore

    d = C.shape[0]
    m = B.shape[1]
    D = np.zeros((d, m), dtype=float)
    tf = ct.ss2tf(ct.ss(A, B, C, D))

    entries: Dict[str, dict] = {}
    for i in range(d):
        for j in range(m):
            key = f"y{i + 1}_u{j + 1}"
            num_ij = _trim_leading_zeros(np.asarray(tf.num[i][j], dtype=float))
            den_ij = _trim_leading_zeros(np.asarray(tf.den[i][j], dtype=float))
            num_expr = _poly_to_string(num_ij, tf_variable)
            den_expr = _poly_to_string(den_ij, tf_variable)
            entries[key] = {
                "numerator_coeffs": [float(v) for v in num_ij],
                "denominator_coeffs": [float(v) for v in den_ij],
                "expression": f"({num_expr}) / ({den_expr})",
            }

    matrix_expression = [
        [entries[f"y{i + 1}_u{j + 1}"]["expression"] for j in range(m)] for i in range(d)
    ]
    return {
        "method": "python-control.ss2tf",
        "shape": {"outputs": int(d), "inputs": int(m)},
        "entries": entries,
        "matrix_expression": matrix_expression,
    }


def _compute_transfer_polynomials(A: np.ndarray, B: np.ndarray, C: np.ndarray, tf_variable: str) -> dict:
    try:
        return _compute_transfer_polynomials_control(A, B, C, tf_variable)
    except ModuleNotFoundError as exc:
        return _compute_transfer_polynomials_scipy(A, B, C, tf_variable, fallback_reason=str(exc))
    except Exception as exc:
        return _compute_transfer_polynomials_scipy(A, B, C, tf_variable, fallback_reason=f"python-control failed: {exc}")


def _entry_sort_key(entry_key: str) -> tuple[int, int]:
    # key format: y{i}_u{j}
    y_part, u_part = entry_key.split("_")
    return int(y_part[1:]), int(u_part[1:])


def _print_transfer_coefficients(label: str, tf_payload: dict, tf_variable: str) -> None:
    print("")
    print(f"=== {label} transfer function coefficients ===")
    print(f"Method: {tf_payload.get('method', 'unknown')}")
    if "fallback_reason" in tf_payload:
        print(f"Fallback reason: {tf_payload['fallback_reason']}")
    shape = tf_payload.get("shape", {})
    print(f"Shape: outputs={shape.get('outputs', '?')}, inputs={shape.get('inputs', '?')}, variable={tf_variable}")
    print("")

    entries = tf_payload.get("entries", {})
    for key in sorted(entries.keys(), key=_entry_sort_key):
        entry = entries[key]
        print(f"{key}:")
        print(f"  numerator_coeffs: {entry.get('numerator_coeffs', [])}")
        print(f"  denominator_coeffs: {entry.get('denominator_coeffs', [])}")
        print(f"  expression: {entry.get('expression', '')}")
        print("")


def _save_poles_csv(path: Path, poles_true: np.ndarray, poles_est: np.ndarray) -> None:
    lines = ["system,real,imag"]
    for val in poles_true:
        lines.append(f"true,{val.real},{val.imag}")
    for val in poles_est:
        lines.append(f"est,{val.real},{val.imag}")
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()

    # Load true matrices from config
    data = RetrieveData(args.config)
    ckf = data.CKF_pack
    A_true = np.asarray(ckf["A_complete"], dtype=float)
    B_true = np.asarray(ckf["B_complete"], dtype=float)
    C = np.asarray(ckf["C_complete"], dtype=float)

    # Vectors and starts for assembly
    p_vec = np.asarray(ckf["p_vec"]).astype(int).reshape(-1)
    s_vec = np.asarray(ckf["s_vec"]).astype(int).reshape(-1)
    comp_start_vec = np.asarray(ckf["comp_start_vec"]).astype(int).reshape(-1)
    input_start_vec = np.asarray(ckf["input_start_vec"]).astype(int).reshape(-1)

    # Locate couplings_final.npz (est_data_loc overrides CLI)
    results_root = _resolve_results_root(args.config, args.results_dir)
    couplings_override = Path(est_data_loc).expanduser() if est_data_loc else args.couplings_path
    couplings_path = _find_couplings_file(args.config, couplings_override, results_root)

    # Build estimated A_hat and B_hat
    diag_A = {k: np.asarray(v) for k, v in data.global_learners_pack["A_mm"].items()}
    diag_B = {k: np.asarray(v) for k, v in data.global_learners_pack["B_mm"].items()}

    with np.load(couplings_path, allow_pickle=False) as couplings:
        offdiag_A = {k.replace("A_", ""): np.asarray(v) for k, v in couplings.items() if k.startswith("A_")}
        offdiag_B = {k.replace("B_", ""): np.asarray(v) for k, v in couplings.items() if k.startswith("B_")}

    A_est = _assemble_A_matrix(diag_A, offdiag_A, comp_start_vec, p_vec)
    B_est = _assemble_B_matrix(diag_B, offdiag_B, comp_start_vec, input_start_vec, p_vec, s_vec)

    # Frequency grid
    omega = np.linspace(0.0, float(args.omega_max), int(args.num_freq))

    # Frequency response
    G_true = _frequency_response(A_true, B_true, C, omega)
    G_est = _frequency_response(A_est, B_est, C, omega)

    G_err = np.linalg.norm(G_true - G_est, axis=(1, 2))

    # Singular values
    sigma_true = _singular_values(G_true)
    sigma_est = _singular_values(G_est)

    # Poles and zeros
    poles_true = np.linalg.eigvals(A_true)
    poles_est = np.linalg.eigvals(A_est)
    zeros_true = _compute_siso_zeros(A_true, B_true, C)
    zeros_est = _compute_siso_zeros(A_est, B_est, C)

    # Output directory
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = couplings_path.parent / "transfer_function"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save core data
    np.savez(
        out_dir / "transfer_function_data.npz",
        omega=omega,
        z=np.exp(1j * omega),
        G_true=G_true,
        G_est=G_est,
        G_err_fro=G_err,
        sigma_true=sigma_true,
        sigma_est=sigma_est,
        poles_true=poles_true,
        poles_est=poles_est,
    )

    # Error CSV
    err_csv_lines = ["omega,err_fro"]
    for w, e in zip(omega, G_err):
        err_csv_lines.append(f"{w},{e}")
    (out_dir / "transfer_function_error.csv").write_text("\n".join(err_csv_lines))

    # Poles CSV
    _save_poles_csv(out_dir / "transfer_function_poles.csv", poles_true, poles_est)

    # Zeros JSON (per SISO transfer)
    zeros_payload = {
        "method": "siso_ss2tf",
        "zeros_true": zeros_true,
        "zeros_est": zeros_est,
    }
    (out_dir / "transfer_function_zeros.json").write_text(json.dumps(zeros_payload, indent=2))

    # Transfer-function matrix coefficients
    coeff_payload = {
        "variable": str(args.tf_variable),
        "true": _compute_transfer_polynomials(A_true, B_true, C, args.tf_variable),
        "est": _compute_transfer_polynomials(A_est, B_est, C, args.tf_variable),
    }
    (out_dir / "transfer_function_coefficients.json").write_text(json.dumps(coeff_payload, indent=2))
    _print_transfer_coefficients("True", coeff_payload["true"], args.tf_variable)
    _print_transfer_coefficients("Estimated", coeff_payload["est"], args.tf_variable)

    # Plots
    plt.figure(figsize=(8, 5))
    plt.plot(omega, G_err, color="#377eb8")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\|G_{true}(e^{j\omega}) - G_{est}(e^{j\omega})\|_F$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "fro_error_vs_omega.pdf", format="pdf", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(omega, sigma_true[:, 0], label="sigma_max true", color="#4daf4a")
    plt.plot(omega, sigma_est[:, 0], label="sigma_max est", color="#e41a1c")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\sigma_{max}(G(e^{j\omega}))$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "sigma_max_vs_omega.pdf", format="pdf", dpi=200)
    plt.close()

    print(f"Saved transfer-function outputs to {out_dir}")


if __name__ == "__main__":
    main()
