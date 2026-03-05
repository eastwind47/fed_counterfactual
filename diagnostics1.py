"""Generate diagnostics plots for main1 experiments."""

from __future__ import annotations

import pickle
import re
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

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

DIAGNOSTIC_EXACT = {
    "align_norm",
    "consensus_norm",
    "loss_align",
    "loss_consensus",
    "loss_align_frac",
    "loss_consensus_frac",
    "gradA_norm_total",
    "gradB_norm_total",
    "A_norm_total",
    "B_norm_total",
    "A_step_total",
    "B_step_total",
    "A_rel_step_total",
    "B_rel_step_total",
}

DIAGNOSTIC_PREFIXES = (
    "local_loss_",
    "grad_norm_",
    "residual_norm_",
    "theta_norm_",
    "phi_norm_",
    "phi_consensus_err_",
    "gradA_norm_",
    "gradB_norm_",
    "A_norm_",
    "B_norm_",
    "A_step_",
    "B_step_",
    "A_rel_step_",
    "B_rel_step_",
)


def _try_float(value) -> float | None:
    """Return float(value) if possible, else None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_history_metrics(histories: List[dict]) -> Dict[str, List[np.ndarray]]:
    """Flatten nested history dicts into per-metric arrays."""
    metrics: Dict[str, List[np.ndarray]] = {}
    for entry in histories:
        for key, value in entry.items():
            if key == "round":
                continue
            if isinstance(value, dict):
                for sub_key, sub_series in value.items():
                    arr = np.atleast_1d(np.asarray(sub_series, dtype=float))
                    if arr.size:
                        metrics.setdefault(f"{key}_{sub_key}", []).append(arr)
            else:
                arr = np.atleast_1d(np.asarray(value, dtype=float))
                if arr.size:
                    metrics.setdefault(key, []).append(arr)
    return metrics


def _collect_metric_row_metrics(metric_rows: List[List[dict]]) -> Dict[str, List[np.ndarray]]:
    """Extract numeric columns from metric_rows tables."""
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
    """Pad variable-length runs with NaNs, then compute mean/std envelopes."""
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
    """Produce a filesystem-safe filename stem."""
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
    return safe or "metric"


def _pretty_label(name: str) -> str:
    """Human-readable axis label."""
    return name.replace("_", " ").title()


def _is_diagnostic_metric(metric_name: str) -> bool:
    """Select metrics intended for debugging/training diagnostics."""
    if metric_name in DIAGNOSTIC_EXACT:
        return True
    return metric_name.startswith(DIAGNOSTIC_PREFIXES)


def _ylabel(metric_name: str) -> str:
    """Map metric keys to cleaner y-axis labels."""
    if metric_name == "align_norm":
        return r"Alignment norm"
    if metric_name == "consensus_norm":
        return r"Consensus norm"
    if metric_name == "loss_align":
        return r"Alignment loss term"
    if metric_name == "loss_consensus":
        return r"Consensus loss term"
    if metric_name == "loss_align_frac":
        return r"Alignment fraction"
    if metric_name == "loss_consensus_frac":
        return r"Consensus fraction"
    if metric_name.startswith("local_loss_"):
        return r"Client residual"
    if metric_name.startswith("grad_norm_"):
        return r"Client gradient norm"
    if metric_name.startswith("residual_norm_"):
        return r"Client residual norm"
    if metric_name.startswith("theta_norm_"):
        return r"$\|\theta\|_F$"
    if metric_name.startswith("phi_norm_"):
        return r"$\|\phi\|_2$"
    if metric_name.startswith("phi_consensus_err_"):
        return r"$\|\phi - \sum B_{mn}\bar{u}_n\|_2$"
    if metric_name.startswith("gradA_norm_"):
        return r"$\|\nabla A\|_F$"
    if metric_name.startswith("gradB_norm_"):
        return r"$\|\nabla B\|_F$"
    if metric_name.startswith("A_norm_"):
        return r"$\|A\|_F$"
    if metric_name.startswith("B_norm_"):
        return r"$\|B\|_F$"
    if metric_name.startswith("A_step_"):
        return r"$\|\Delta A\|_F$"
    if metric_name.startswith("B_step_"):
        return r"$\|\Delta B\|_F$"
    if metric_name.startswith("A_rel_step_"):
        return r"$\|\Delta A\|/\|A\|$"
    if metric_name.startswith("B_rel_step_"):
        return r"$\|\Delta B\|/\|B\|$"
    return _pretty_label(metric_name)


def _plot_server_loss_decomposition(
    all_metrics: Dict[str, List[np.ndarray]],
    out_dir: Path,
) -> None:
    """Plot global loss with alignment/consensus components on one figure."""
    needed = ["global_loss", "loss_align", "loss_consensus"]
    if any(name not in all_metrics for name in needed):
        return

    series = {}
    for name in needed:
        try:
            _, mean, std = _stack_runs(all_metrics[name])
        except ValueError:
            return
        series[name] = (mean, std)

    rounds = np.arange(1, len(series["global_loss"][0]) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(rounds, series["global_loss"][0], color=colorblind_colors[1], label=r"$L_s^k$")
    ax.fill_between(
        rounds,
        series["global_loss"][0] - series["global_loss"][1],
        series["global_loss"][0] + series["global_loss"][1],
        color=colorblind_colors[1],
        alpha=0.2,
    )
    ax.plot(rounds, series["loss_align"][0], color=colorblind_colors[3], label=r"$L_{\mathrm{align}}^k$")
    ax.fill_between(
        rounds,
        series["loss_align"][0] - series["loss_align"][1],
        series["loss_align"][0] + series["loss_align"][1],
        color=colorblind_colors[3],
        alpha=0.2,
    )
    ax.plot(rounds, series["loss_consensus"][0], color=colorblind_colors[4], label=r"$L_{\mathrm{cons}}^k$")
    ax.fill_between(
        rounds,
        series["loss_consensus"][0] - series["loss_consensus"][1],
        series["loss_consensus"][0] + series["loss_consensus"][1],
        color=colorblind_colors[4],
        alpha=0.2,
    )

    ax.set_xlabel("Number of iterations ($k$)", fontsize=25)
    ax.set_ylabel("Server loss terms", fontsize=25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    fig.tight_layout()
    fig.savefig(out_dir / "server_loss_decomposition.pdf", format="pdf", dpi=800, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = ConfigParser()
    config.read("config.ini")

    location_cfg = config["LOCATION"]
    results_root = Path(location_cfg["results_location"]).expanduser()
    plots_root = Path(location_cfg["plots_location"]).expanduser()

    results_dir = results_root / "main1"
    aggregate_path = results_dir / "monte_carlo_history.pkl"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Aggregated Monte Carlo payload not found: {aggregate_path}")

    with aggregate_path.open("rb") as handle:
        payload = pickle.load(handle)

    histories = payload.get("histories", [])
    metric_rows = payload.get("metric_rows", [])
    if not histories:
        raise ValueError("No history time-series available for diagnostics.")

    diagnostics_dir = plots_root / "diagnostics1"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    dkf_ref = payload.get("dkf_avg_sq_residual", payload.get("dkf_avg_residual", {}))

    history_metrics = _collect_history_metrics(histories)
    row_metrics = _collect_metric_row_metrics(metric_rows)

    all_metrics: Dict[str, List[np.ndarray]] = {**history_metrics}
    for key, series in row_metrics.items():
        if key in all_metrics:
            continue
        all_metrics[key] = series

    for metric_name, series_list in sorted(all_metrics.items()):
        if not _is_diagnostic_metric(metric_name):
            continue
        try:
            _data, mean, std = _stack_runs(series_list)
        except ValueError:
            continue

        rounds = np.arange(1, mean.size + 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        line_label = None
        if metric_name.startswith("local_loss_"):
            cid = metric_name.split("_")[-1]
            line_label = rf"$L_{{{cid},a}}^k$"
        ax.plot(rounds, mean, color=colorblind_colors[1], label=line_label)
        ax.fill_between(
            rounds,
            mean - std,
            mean + std,
            color=colorblind_colors[1],
            alpha=0.25,
        )

        ax.set_xlabel("Number of iterations ($k$)", fontsize=25)
        if metric_name == "consensus_norm" or metric_name.startswith("phi_consensus_err_"):
            ax.set_ylabel("")
        else:
            ax.set_ylabel(_ylabel(metric_name), fontsize=25)
        ax.grid(True, alpha=0.3)

        if metric_name.startswith("local_loss_"):
            cid = metric_name.split("_")[-1]
            if cid in dkf_ref and np.isfinite(dkf_ref[cid]):
                ax.axhline(
                    dkf_ref[cid],
                    color=colorblind_colors[0],
                    linestyle="--",
                    linewidth=2,
                    label=rf"$\frac{{1}}{{T}}\sum_{{t=1}}^T \|r_{{{cid},c}}^t\|_2^2$",
                )
            ax.legend(fontsize=20)

        fig.tight_layout()
        out_path = diagnostics_dir / f"{_sanitize_name(metric_name)}.pdf"
        fig.savefig(out_path, format="pdf", dpi=800, bbox_inches="tight")
        plt.close(fig)

    _plot_server_loss_decomposition(all_metrics, diagnostics_dir)
    print(f"Diagnostics plots saved under {diagnostics_dir}")


if __name__ == "__main__":
    main()
