"""
plots.py -- figures for the report (PNG files, no display needed).

    plot_solve_time        grouped bars: solve time by formulation and size
    plot_lp_gap            grouped bars: LP gap by formulation and size
    plot_generation_stack  hourly generation by unit (stacked) vs demand
    plot_ramp_stress       LP gap and solve time when ramp limits are tightened
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # draw to files only
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

from .data_loader import UCInstance         # noqa: E402
from .formulations import FORMULATION_LABELS  # noqa: E402

COLORS = {"A": "#8d99ae", "B": "#2a9d8f", "C": "#e9c46a"}


def _label(key: str) -> str:
    return FORMULATION_LABELS.get(key, key)


def _grouped_bars(ax, summary: pd.DataFrame, value_col: str, fmt: str) -> None:
    """Draw one group of bars per size (n_gens), one bar per formulation."""
    sizes = sorted(summary["n_gens"].unique())
    forms = sorted(summary["formulation"].unique())
    width = 0.8 / max(1, len(forms))
    x = np.arange(len(sizes))
    for i, f in enumerate(forms):
        vals = []
        for s in sizes:
            sel = summary[(summary["formulation"] == f) & (summary["n_gens"] == s)][value_col]
            vals.append(float(sel.iloc[0]) if len(sel) and pd.notna(sel.iloc[0]) else np.nan)
        bars = ax.bar(x + (i - (len(forms) - 1) / 2) * width, vals, width,
                      label=_label(f), color=COLORS.get(f))
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} generators" for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot_solve_time(summary: pd.DataFrame, path: str | Path) -> None:
    """Average MIP solve time (seconds, log scale) by formulation and size."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = summary.copy()
    s["avg_solve_time_s"] = s["avg_solve_time_s"].clip(lower=1e-3)
    _grouped_bars(ax, s, "avg_solve_time_s", "{:.2f}")
    ax.set_yscale("log")
    ax.set_ylabel("average solve time (s, log scale)")
    ax.set_title("Solve time by formulation and instance size")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lp_gap(summary: pd.DataFrame, path: str | Path) -> None:
    """Average LP-relaxation gap (%) by formulation and size."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bars(ax, summary, "avg_lp_gap_pct", "{:.2f}")
    ax.set_ylabel("average LP gap (%)  =  (best MIP - LP) / best MIP")
    ax.set_title("Tightness of the LP relaxation (smaller = tighter)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_generation_stack(schedule: pd.DataFrame, inst: UCInstance, path: str | Path,
                          title: str = "", top_k: int = 8) -> None:
    """Stacked hourly generation (largest `top_k` units + 'other units') vs demand."""
    wide = schedule.pivot(index="hour", columns="generator", values="power_mw").fillna(0.0)
    order = wide.sum().sort_values(ascending=False).index.tolist()
    top, rest = order[:top_k], order[top_k:]
    series = [wide[g].to_numpy() for g in top]
    labels = list(top)
    if rest:
        series.append(wide[rest].sum(axis=1).to_numpy())
        labels.append(f"other units ({len(rest)})")
    hours = wide.index.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.stackplot(hours, *series, labels=labels, alpha=0.9)
    ax.plot(hours, [inst.demand[int(h)] for h in hours], "k--", lw=2, label="demand")
    ax.set_xlabel("hour")
    ax.set_ylabel("MW")
    ax.set_xlim(hours.min(), hours.max())
    ax.set_title(title or f"Hourly generation - {inst.name}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ramp_stress(df: pd.DataFrame, path: str | Path) -> None:
    """Two panels: LP gap and solve time versus ramp_scale (smaller = tighter ramps)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, ylabel in ((axes[0], "lp_gap_pct", "LP gap (%)"),
                            (axes[1], "solve_time_s", "solve time (s)")):
        for f, grp in df.groupby("formulation"):
            grp = grp.sort_values("ramp_scale")
            ax.plot(grp["ramp_scale"], grp[col], marker="o", label=_label(f), color=COLORS.get(f))
        ax.set_xlabel("ramp-limit scale (1 = original data, smaller = tighter ramps)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("LP gap under ramp stress")
    axes[1].set_title("Solve time under ramp stress")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
