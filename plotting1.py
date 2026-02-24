"""Generate Monte Carlo summary plots for main1 experiments."""

from __future__ import annotations

import csv
import pickle
import re
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

colorblind_colors = [
    "#f781bf",
    "#377eb8",
    "#984ea3",
    "#ff7f00",
    "#4daf4a",
    "#a65628",
    "#ffff33",
    "#e41a1c",
]


def _try_float(value) -> float | None:
    """Return float(value) if possible, else None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_history_metrics(histories: List[dict]) -> Dict[str, List[np.ndarray]]:
    """Flatten the nested history dict into per-metric numpy series."""
    metrics: Dict[str, List[np.ndarray]] = {}
    for entry in histories:
        for key, value in entry.items():
            if key == "round":
                continue
            if isinstance(value, dict):
                for sub_key, sub_series in value.items():
                    arr = np.atleast_1d(np.asarray(sub_series, dtype=float))
                    if arr.size == 0:
                        continue
                    metrics.setdefault(f"{key}_{sub_key}", []).append(arr)
            else:
                arr = np.atleast_1d(np.asarray(value, dtype=float))
                if arr.size == 0:
                    continue
                metrics.setdefault(key, []).append(arr)
    return metrics


def _collect_metric_row_metrics(metric_rows: List[List[dict]]) -> Dict[str, List[np.ndarray]]:
    """Extract column-wise numeric series from the metric_rows tables."""
    metrics: Dict[str, List[np.ndarray]] = {}
    for rows in metric_rows:
        if not rows:
            continue
        keys = {key for row in rows for key in row.keys() if key != "round"}
        for key in keys:
            column: List[float] = []
            valid = True
            for row in rows:
                val = _try_float(row.get(key))
                if val is None:
                    valid = False
                    break
                column.append(val)
            if valid:
                metrics.setdefault(key, []).append(np.asarray(column, dtype=float))
    return metrics


def _nanmean_safe(values: List[float]) -> float:
    """Return the finite mean of a list, allowing NaNs but ignoring them."""
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _write_average_final_metrics(
    final_losses: List[float],
    final_consensus: List[float],
    csv_path: Path,
) -> None:
    """Store Monte Carlo-average final metrics in a CSV file."""
    if not final_losses and not final_consensus:
        return
    avg_loss = _nanmean_safe(final_losses)
    avg_consensus = _nanmean_safe(final_consensus)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["average_final_server_loss", "average_final_consensus_norm"])
        writer.writerow([avg_loss, avg_consensus])


def _stack_runs(series_list: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-length runs with NaNs and compute mean/std envelopes."""
    if not series_list:
        raise ValueError("Empty series list")
    max_len = max(len(arr) for arr in series_list)
    if max_len == 0:
        raise ValueError("All series are empty")
    data = np.full((len(series_list), max_len), np.nan, dtype=float)
    for idx, arr in enumerate(series_list):
        if arr.size:
            data[idx, : arr.size] = arr
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    return data, mean, std


def _sanitize_name(name: str) -> str:
    """Produce a filesystem-safe version of the metric name."""
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
    return safe or "metric"


def _pretty_label(name: str) -> str:
    """Make a human-friendly label for plot titles/axes."""
    return name.replace("_", " ").title()


def _is_summary_metric(metric_name: str) -> bool:
    """Return True only for publication-style summary metrics."""
    if metric_name in {
        "global_loss",
        "A_global_abs_err",
        "A_global_rel_err",
        "B_global_abs_err",
        "B_global_rel_err",
    }:
        return True
    return metric_name.startswith(
        (
            "A_abs_err_",
            "A_rel_err_",
            "B_abs_err_",
            "B_rel_err_",
        )
    )


def _average_blocks(block_runs: List[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Average matching blocks across runs."""
    if not block_runs:
        return {}
    common_keys = set(block_runs[0].keys())
    for run in block_runs[1:]:
        common_keys &= set(run.keys())
    averaged: dict[str, np.ndarray] = {}
    for key in sorted(common_keys):
        stack = np.stack([np.asarray(run[key]) for run in block_runs], axis=0)
        averaged[key] = stack.mean(axis=0)
    return averaged


def _assemble_A_matrix(
    diag_blocks: dict[str, np.ndarray],
    offdiag_blocks: dict[str, np.ndarray],
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
            full[r0:r1, c0:c1] = block
    return full


def _assemble_B_matrix(
    diag_blocks: dict[str, np.ndarray],
    offdiag_blocks: dict[str, np.ndarray],
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
            full[r0:r1, c0:c1] = block
    return full


def main() -> None:
    config = ConfigParser()
    config.read("config.ini")

    location_cfg = config["LOCATION"]
    results_root = Path(location_cfg["results_location"]).expanduser()
    plots_root = Path(location_cfg["plots_location"]).expanduser()

    results_dir = results_root / "main1"
    aggregate_path = results_dir / "monte_carlo_history.pkl"
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"Aggregated Monte Carlo payload not found: {aggregate_path}"
        )

    with aggregate_path.open("rb") as handle:
        payload = pickle.load(handle)

    histories = payload.get("histories", [])
    metric_rows = payload.get("metric_rows", [])
    if not histories:
        raise ValueError("No history time-series available for plotting.")

    final_global_losses: List[float] = []
    final_consensus_norms: List[float] = []
    for history in histories:
        series = history.get("global_loss")
        if series is None:
            continue
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            continue
        final_val = float(arr[-1])
        if np.isfinite(final_val):
            final_global_losses.append(final_val)

        consensus_series = history.get("consensus_norm")
        if consensus_series is not None:
            cons_arr = np.asarray(consensus_series, dtype=float)
            if cons_arr.size:
                final_cons = float(cons_arr[-1])
                if np.isfinite(final_cons):
                    final_consensus_norms.append(final_cons)

    avg_loss_csv = results_dir / "average_final_server_loss.csv"
    _write_average_final_metrics(final_global_losses, final_consensus_norms, avg_loss_csv)

    diag_A_blocks = {
        key: np.asarray(val)
        for key, val in payload.get("diag_A_blocks", {}).items()
    }
    diag_B_blocks = {
        key: np.asarray(val)
        for key, val in payload.get("diag_B_blocks", {}).items()
    }
    comp_start_vec = np.asarray(payload.get("comp_start_vec", []), dtype=int)
    input_start_vec = np.asarray(payload.get("input_start_vec", []), dtype=int)
    p_vec = np.asarray(payload.get("p_vec", []), dtype=int)
    s_vec = np.asarray(payload.get("s_vec", []), dtype=int)
    final_A_runs = payload.get("final_A_mn", [])
    final_B_runs = payload.get("final_B_mn", [])
    A_total_norm = float(payload.get("A_total_norm", float("nan")))
    B_total_norm = float(payload.get("B_total_norm", float("nan")))
    plots_root.mkdir(parents=True, exist_ok=True)

    history_metrics = _collect_history_metrics(histories)
    row_metrics = _collect_metric_row_metrics(metric_rows)

    all_metrics: Dict[str, List[np.ndarray]] = {**history_metrics}
    for key, series in row_metrics.items():
        if key in all_metrics:
            continue
        all_metrics[key] = series

    A_abs_runs: List[np.ndarray] = []
    B_abs_runs: List[np.ndarray] = []
    for history in histories:
        A_abs_dict = history.get("A_abs_err", {})
        if A_abs_dict:
            max_len = max((len(vals) for vals in A_abs_dict.values()), default=0)
            if max_len > 0:
                agg = np.zeros(max_len, dtype=float)
                for vals in A_abs_dict.values():
                    arr = np.asarray(vals, dtype=float)
                    agg[: arr.size] += arr ** 2
                A_abs_runs.append(np.sqrt(agg))
        B_abs_dict = history.get("B_abs_err", {})
        if B_abs_dict:
            max_len = max((len(vals) for vals in B_abs_dict.values()), default=0)
            if max_len > 0:
                agg = np.zeros(max_len, dtype=float)
                for vals in B_abs_dict.values():
                    arr = np.asarray(vals, dtype=float)
                    agg[: arr.size] += arr ** 2
                B_abs_runs.append(np.sqrt(agg))

    if A_abs_runs:
        all_metrics["A_global_abs_err"] = A_abs_runs
        if np.isfinite(A_total_norm) and A_total_norm > 1e-12:
            all_metrics["A_global_rel_err"] = [run / A_total_norm for run in A_abs_runs]

    if B_abs_runs:
        all_metrics["B_global_abs_err"] = B_abs_runs
        if np.isfinite(B_total_norm) and B_total_norm > 1e-12:
            all_metrics["B_global_rel_err"] = [run / B_total_norm for run in B_abs_runs]

    for metric_name, series_list in sorted(all_metrics.items()):
        if not _is_summary_metric(metric_name):
            continue
        try:
            _data, mean, std = _stack_runs(series_list)
        except ValueError:
            continue

        rounds = np.arange(1, mean.size + 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(rounds, mean, color=colorblind_colors[1])
        ax.fill_between(
            rounds,
            mean - std,
            mean + std,
            color=colorblind_colors[1],
            alpha=0.25,
            label="±1 std",
        )
        ax.set_xlabel("Number of iterations ($k$)", fontsize=25)
        if metric_name == "global_loss":
            ylabel = r"Server Loss ($L_s^k$)"
        elif metric_name == "A_global_rel_err":
            ylabel = r"Relative error of $\hat{A}$"
        elif metric_name == "B_global_rel_err":
            ylabel = r"Relative error of $\hat{B}$"
        elif "consensus" in metric_name:
            ylabel = ""
        elif metric_name.startswith("local_loss_"):
            ylabel = "Client residual"
        else:
            ylabel = _pretty_label(metric_name)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = plots_root / f"{_sanitize_name(metric_name)}.pdf"
        fig.savefig(out_path, format="pdf", dpi=800, bbox_inches="tight")
        plt.close(fig)

    if len(final_A_runs) and len(diag_A_blocks) and comp_start_vec.size and p_vec.size:
        avg_A_blocks = _average_blocks(final_A_runs)
        avg_B_blocks = _average_blocks(final_B_runs)
        assembled_A = _assemble_A_matrix(diag_A_blocks, avg_A_blocks, comp_start_vec, p_vec)
        assembled_B = _assemble_B_matrix(
            diag_B_blocks,
            avg_B_blocks,
            comp_start_vec,
            input_start_vec,
            p_vec,
            s_vec,
        )
        np.savetxt(results_dir / "estimated_A_complete.csv", assembled_A, delimiter=",")
        np.savetxt(results_dir / "estimated_B_complete.csv", assembled_B, delimiter=",")
        print("Saved assembled matrices to estimated_A_complete.csv and estimated_B_complete.csv")

    print(f"Plots saved under {plots_root}")


if __name__ == "__main__":
    main()
