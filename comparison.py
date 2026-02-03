"""Generate side-by-side comparisons of v1 (main1) and v2 (main2) metrics."""

from __future__ import annotations

import pickle
import re
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# Match plotting style with training plots.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

# Keep plot styling consistent with plotting1.py / plotting2.py.
FIG_SIZE = (8, 6)
SAVE_KWARGS = {"format": "pdf", "dpi": 800, "bbox_inches": "tight"}
LABEL_FONT_SIZE = 25
LEGEND_FONT_SIZE = 20

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


# Set this before running to control where comparison plots are written.
comparison_location = "/Users/home/Documents/naz/research_codes/counterfactual_reasoning/synthetic_exp/Components_2/comparison_plots" # user supplied at runtime


def _try_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_history_metrics(histories: List[dict]) -> Dict[str, List[np.ndarray]]:
    metrics: Dict[str, List[np.ndarray]] = {}
    for entry in histories:
        for key, value in entry.items():
            if key == "round":
                continue
            if isinstance(value, dict):
                for sub_key, sub_series in value.items():
                    arr = np.asarray(sub_series, dtype=float)
                    metrics.setdefault(f"{key}_{sub_key}", []).append(arr)
            else:
                arr = np.asarray(value, dtype=float)
                metrics.setdefault(key, []).append(arr)
    return metrics


def _collect_metric_row_metrics(metric_rows: List[List[dict]]) -> Dict[str, List[np.ndarray]]:
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


def _stack_runs(series_list: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
    return safe or "metric"


def _pretty_label(name: str) -> str:
    if name == "global_loss":
        return r"Server Loss ($L_s^k$)"
    return name.replace("_", " ").title()


def _load_payload(
    config_path: str, subdir: str
) -> tuple[Dict[str, List[np.ndarray]], dict[str, float], dict[str, float]]:
    cfg = ConfigParser()
    cfg.read(config_path)
    results_root = Path(cfg["LOCATION"]["results_location"]).expanduser()
    aggregate_path = results_root / subdir / "monte_carlo_history.pkl"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing {aggregate_path}")
    payload = pickle.loads(aggregate_path.read_bytes())
    histories = payload.get("histories", [])
    metric_rows = payload.get("metric_rows", [])
    if not histories:
        raise ValueError(f"No histories found in {aggregate_path}")

    history_metrics = _collect_history_metrics(histories)
    row_metrics = _collect_metric_row_metrics(metric_rows)

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

    all_metrics = {**history_metrics}
    for key, series in row_metrics.items():
        all_metrics.setdefault(key, series)

    total_norm = payload.get("A_total_norm")
    if A_abs_runs:
        all_metrics["A_global_abs_err"] = A_abs_runs
        if isinstance(total_norm, (float, int)) and total_norm > 1e-12:
            all_metrics["A_global_rel_err"] = [run / total_norm for run in A_abs_runs]
    total_norm = payload.get("B_total_norm")
    if B_abs_runs:
        all_metrics["B_global_abs_err"] = B_abs_runs
        if isinstance(total_norm, (float, int)) and total_norm > 1e-12:
            all_metrics["B_global_rel_err"] = [run / total_norm for run in B_abs_runs]

    ckf_ref_raw = payload.get("ckf_avg_residual", {}) or {}
    dkf_ref_raw = payload.get("dkf_avg_residual", {}) or {}
    ckf_ref: dict[str, float] = {}
    for cid, val in ckf_ref_raw.items():
        numeric = _try_float(val)
        if numeric is not None:
            ckf_ref[str(cid)] = numeric
    dkf_ref: dict[str, float] = {}
    for cid, val in dkf_ref_raw.items():
        numeric = _try_float(val)
        if numeric is not None:
            dkf_ref[str(cid)] = numeric

    return all_metrics, ckf_ref, dkf_ref


def main() -> None:
    if comparison_location is None:
        raise ValueError("comparison_location must be set to a writable directory before running.")
    comparison_dir = Path(comparison_location)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    metrics_v1, ckf_v1, dkf_v1 = _load_payload("config.ini", "main1")
    metrics_v2, ckf_v2, dkf_v2 = _load_payload("config2.ini", "main2")

    ckf_ref = {**ckf_v2, **ckf_v1}
    dkf_ref = {**dkf_v2, **dkf_v1}

    shared_keys = sorted(set(metrics_v1.keys()) & set(metrics_v2.keys()))

    for metric_name in shared_keys:
        series_v1 = metrics_v1[metric_name]
        series_v2 = metrics_v2[metric_name]
        try:
            _, mean_v1, std_v1 = _stack_runs(series_v1)
            _, mean_v2, std_v2 = _stack_runs(series_v2)
        except ValueError:
            continue

        rounds = np.arange(1, max(len(mean_v1), len(mean_v2)) + 1)
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        line_pm, = ax.plot(
            rounds[: len(mean_v1)],
            mean_v1,
            label="PM",
            color=colorblind_colors[1],
        )
        ax.fill_between(
            rounds[: len(mean_v1)],
            mean_v1 - std_v1,
            mean_v1 + std_v1,
            color=colorblind_colors[1],
            alpha=0.25,
            label="_nolegend_",
        )

        line_al, = ax.plot(
            rounds[: len(mean_v2)],
            mean_v2,
            label="AL",
            color=colorblind_colors[3],
        )
        ax.fill_between(
            rounds[: len(mean_v2)],
            mean_v2 - std_v2,
            mean_v2 + std_v2,
            color=colorblind_colors[3],
            alpha=0.25,
            label="_nolegend_",
        )
        ax.set_xlabel(r"Number of iterations ($k$)", fontsize=LABEL_FONT_SIZE)
        if metric_name == "global_loss":
            ylabel = r"Server Loss ($L_s^k$)"
        elif metric_name == "consensus_norm":
            ylabel = r"$\mathcal{D}$"
        elif metric_name.startswith("phi_consensus_err_"):
            cid = metric_name.split("_")[-1]
            ylabel = rf"$\delta d_{{{cid}}}$"
        elif metric_name.startswith("local_loss_"):
            ylabel = "Client residual"
        else:
            ylabel = _pretty_label(metric_name)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        legend_handles = [line_pm, line_al]
        legend_labels = ["PM", "AL"]

        if metric_name.startswith("local_loss_"):
            cid = metric_name.split("_")[-1]
            dkf_val = dkf_ref.get(cid)
            if dkf_val is not None and np.isfinite(dkf_val):
                dkf_line = ax.axhline(
                    dkf_val,
                    color=colorblind_colors[0],
                    linestyle="--",
                    linewidth=2,
                )
                legend_handles.append(dkf_line)
                legend_labels.append(
                    rf"$\frac{{1}}{{T}} \sum_{{t = 1}}^T \|r_{{{cid},c}}^t\|_2^2$"
                )
            # CKF baseline temporarily disabled; uncomment when centralized reference is available.
            # ckf_val = ckf_ref.get(cid)
            # if ckf_val is not None and np.isfinite(ckf_val):
            #     ckf_line = ax.axhline(
            #         ckf_val,
            #         color=colorblind_colors[3],
            #         linestyle="-.",
            #         linewidth=2,
            #     )
            #     legend_handles.append(ckf_line)
            #     legend_labels.append(
            #         rf"$\frac{{1}}{{T}} \sum_{{t = 1}}^T \|r_{{{cid},o}}^t\|_2^2$"
            #     )

        ax.legend(legend_handles, legend_labels, fontsize=LEGEND_FONT_SIZE)
        fig.tight_layout()

        out_path = comparison_dir / f"comparison_{_sanitize_name(metric_name)}.pdf"
        fig.savefig(out_path, **SAVE_KWARGS)
        plt.close(fig)

    print(f"Comparison plots saved under {comparison_dir}")


if __name__ == "__main__":
    main()
