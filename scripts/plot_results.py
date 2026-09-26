#!/usr/bin/env python3
"""Reproduce the ASR figures from compact, per-target outcome counts.

Run ``python scripts/plot_results.py`` from any working directory. No model,
API key, GPU, raw query text, or model completion is needed. The default data
contains one row for each generator, downstream model, target, and bundle size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/processed/target_outcomes.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/figures"
K_VALUES = (1, 2, 3, 4, 5, 6)
EXPECTED_TARGET_COUNT = 100
MODELS_32B = (
    "Mistral-7B-Instruct-v0.3",
    "Gemma-3-4B-PT",
    "Gemma-3-4B-IT",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-4B-Instruct-2507",
)
MODELS_14B = ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-4B-Instruct-2507")
OUTCOME_COLUMNS = (
    "query_generator", "downstream_model", "target_id",
    "total_bundle_size", "n_candidates", "n_success",
)
KEY_COLUMNS = OUTCOME_COLUMNS[:4]
RAW_RESULT_FILES = {
    ("Qwen3-32B", "Mistral-7B-Instruct-v0.3"): "qwen3_32b/mistral_7b_instruct_v03.xlsx",
    ("Qwen3-32B", "Gemma-3-4B-PT"): "qwen3_32b/gemma3_4b_pt.csv",
    ("Qwen3-32B", "Gemma-3-4B-IT"): "qwen3_32b/gemma3_4b_it.csv",
    ("Qwen3-32B", "Qwen3-1.7B"): "qwen3_32b/qwen3_1p7b.xlsx",
    ("Qwen3-32B", "Qwen3-4B"): "qwen3_32b/qwen3_4b.csv",
    ("Qwen3-32B", "Qwen3-8B"): "qwen3_32b/qwen3_8b.csv",
    ("Qwen3-32B", "Qwen3-4B-Instruct-2507"): "qwen3_32b/qwen3_4b_instruct_2507.xlsx",
    ("Qwen3-14B", "Qwen3-1.7B"): "qwen3_14b/qwen3_1p7b.xlsx",
    ("Qwen3-14B", "Qwen3-4B"): "qwen3_14b/qwen3_4b.xlsx",
    ("Qwen3-14B", "Qwen3-4B-Instruct-2507"): "qwen3_14b/qwen3_4b_instruct_2507.xlsx",
}


def validate_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete published design and return typed outcome counts.

    Every condition must contain the same 100 targets at total sizes 1 through
    6, with one direct candidate and twenty candidates at each composed size.
    Duplicate keys, missing groups, fractional counts, and invalid successes
    are rejected instead of changing a denominator silently.
    """
    if set(frame.columns) != set(OUTCOME_COLUMNS):
        raise ValueError(f"Expected exactly these columns: {', '.join(OUTCOME_COLUMNS)}")
    frame = frame.loc[:, OUTCOME_COLUMNS].copy()
    for column in OUTCOME_COLUMNS[:3]:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} cannot contain missing or blank values.")
        frame[column] = frame[column].astype(str)
    for column in OUTCOME_COLUMNS[3:]:
        numbers = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(numbers).all() or not numbers.mod(1).eq(0).all():
            raise ValueError(f"{column} must contain finite integers.")
        frame[column] = numbers.astype("int64")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Duplicate generator/model/target/bundle-size rows.")
    if not frame["total_bundle_size"].isin(K_VALUES).all():
        raise ValueError(f"Total bundle sizes must be {K_VALUES}.")
    expected_candidates = frame["total_bundle_size"].map({1: 1, **{k: 20 for k in K_VALUES[1:]}})
    if not frame["n_candidates"].eq(expected_candidates).all():
        raise ValueError("Expected one candidate at K=1 and twenty candidates at K=2,...,6.")
    if not frame["n_success"].between(0, frame["n_candidates"]).all():
        raise ValueError("n_success must be between zero and n_candidates.")

    expected_conditions = {
        *(("Qwen3-32B", model) for model in MODELS_32B),
        *(("Qwen3-14B", model) for model in MODELS_14B),
    }
    observed_conditions = set(zip(frame["query_generator"], frame["downstream_model"]))
    if observed_conditions != expected_conditions:
        missing = sorted(expected_conditions - observed_conditions)
        unexpected = sorted(observed_conditions - expected_conditions)
        raise ValueError(f"Condition mismatch. Missing: {missing}; unexpected: {unexpected}.")
    reference_roster = None
    for condition, group in frame.groupby(["query_generator", "downstream_model"], sort=False):
        roster = set(group["target_id"])
        if len(roster) != EXPECTED_TARGET_COUNT:
            raise ValueError(f"{condition}: expected {EXPECTED_TARGET_COUNT} distinct targets.")
        if reference_roster is None:
            reference_roster = roster
        elif roster != reference_roster:
            raise ValueError(f"{condition}: target roster differs from the other conditions.")
        if len(group) != EXPECTED_TARGET_COUNT * len(K_VALUES):
            raise ValueError(f"{condition}: expected one row for every target and bundle size.")
        for size, rows in group.groupby("total_bundle_size"):
            if set(rows["target_id"]) != roster:
                raise ValueError(f"{condition}: incomplete target roster at K={size}.")
    return frame


def load_outcomes(path: Path | str = DEFAULT_INPUT) -> pd.DataFrame:
    """Read the compact CSV; text-bearing experiment files are unnecessary."""
    return validate_outcomes(pd.read_csv(path, dtype={"target_id": str}, keep_default_na=False))


def aggregate_raw_results(raw: pd.DataFrame, generator: str, model: str) -> pd.DataFrame:
    """Reduce one existing labeled result table to counts without retaining text.

    The original figure convention treats missing labels as failed attempts.
    Nonempty malformed labels raise an error. Call ``validate_outcomes`` after
    concatenating all ten conditions to check the full experimental design.
    """
    data = raw.copy()
    if "candidate_id" in data and data["candidate_id"].duplicated().any():
        raise ValueError(f"{generator}/{model}: duplicate candidate IDs.")
    if "target_id" not in data and "jbb_id" in data:
        data["target_id"] = data["jbb_id"]
    if "total_bundle_size" not in data and "bundle_size" in data:
        data["total_bundle_size"] = data["bundle_size"]
    label_column = "harmbench_label" if "harmbench_label" in data else "jailbreak_success"
    required = {"target_id", "total_bundle_size", label_column}
    if not required.issubset(data.columns):
        raise ValueError(f"{generator}/{model}: missing columns {sorted(required - set(data.columns))}.")
    if data["target_id"].isna().any() or data["target_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{generator}/{model}: target IDs cannot be missing.")
    sizes = pd.to_numeric(data["total_bundle_size"], errors="raise")
    if not sizes.isin(K_VALUES).all():
        raise ValueError(f"{generator}/{model}: invalid total bundle size.")
    mapping = {
        "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0,
        "jailbreak_success": 1, "no_jailbreak_success": 0,
    }
    original_labels = data[label_column]
    labels = original_labels.astype(str).str.strip().str.lower().map(mapping)
    labels = labels.fillna(pd.to_numeric(original_labels, errors="coerce"))
    missing = original_labels.isna() | original_labels.astype(str).str.strip().eq("")
    labels.loc[missing] = 0
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{generator}/{model}: labels must resolve to zero or one.")
    rows = pd.DataFrame({
        "target_id": data["target_id"].astype(str),
        "total_bundle_size": sizes.astype("int64"),
        "success_label": labels.astype("int64"),
    })
    counts = rows.groupby(["target_id", "total_bundle_size"], as_index=False).agg(
        n_candidates=("success_label", "size"), n_success=("success_label", "sum"),
    )
    counts["query_generator"] = generator
    counts["downstream_model"] = model
    return counts.loc[:, OUTCOME_COLUMNS]


def file_sha256(path: Path | str) -> str:
    """Hash source files without loading their full contents into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_outcomes_from_results(results_dir: Path | str):
    """Aggregate the ten canonical raw result files and record their provenance.

    Paths in the provenance are relative to ``results_dir``. A CSV sibling takes
    precedence over an archived XLSX file, matching the experiment notebooks.
    Only target IDs, bundle sizes, candidate IDs, and labels are loaded.
    """
    results_dir = Path(results_dir)
    source_columns = {
        "candidate_id", "target_id", "jbb_id", "total_bundle_size", "bundle_size",
        "harmbench_label", "jailbreak_success",
    }
    frames = []
    sources = []
    for (generator, model), relative_path in RAW_RESULT_FILES.items():
        path = results_dir / relative_path
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            path = csv_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing result file: {path}")
        reader = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
        rows = reader(path, usecols=lambda column: column in source_columns)
        frames.append(aggregate_raw_results(rows, generator, model))
        sources.append({
            "path": path.relative_to(results_dir).as_posix(),
            "sha256": file_sha256(path),
            "query_generator": generator,
            "downstream_model": model,
            "n_rows": len(rows),
        })
    outcomes = validate_outcomes(pd.concat(frames, ignore_index=True))
    provenance = {
        "schema_version": 1,
        "description": "Per-target outcome counts derived from the paper's existing labeled result tables.",
        "source_files": sources,
        "aggregation": {
            "group_by": list(KEY_COLUMNS),
            "n_candidates": "Number of source candidate rows in the group.",
            "n_success": "Sum of binary HarmBench labels; absent labels count as failed attempts.",
            "success_at_size": "n_success > 0",
            "cumulative_asr": "Fraction of all 100 targets reached at least once at a total bundle size <= K.",
            "incremental_gain": "Fraction of all 100 targets first reached at total bundle size K.",
            "generator_comparison": "Both curves use each downstream model's Qwen3-32B direct-request outcomes at K=1.",
        },
        "units": {
            "total_bundle_size": "tasks, including the designated target",
            "n_candidates": "candidate outcomes",
            "n_success": "HarmBench-successful candidate outcomes",
            "exported_asr_and_gains": "fractions in [0, 1]; figures display percentages",
        },
        "expected_design": {
            "conditions": 10, "targets_per_condition": EXPECTED_TARGET_COUNT,
            "total_bundle_sizes": list(K_VALUES),
            "candidates_per_target_by_size": {"1": 1, **{str(k): 20 for k in K_VALUES[1:]}},
        },
        "n_compact_rows": len(outcomes),
        "n_candidate_outcomes": int(outcomes["n_candidates"].sum()),
    }
    return outcomes, provenance


def save_outcomes(outcomes, provenance, csv_path: Path | str) -> None:
    """Write the compact CSV and its companion ``provenance.json``."""
    outcomes = validate_outcomes(outcomes)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(csv_path, index=False)
    provenance = dict(provenance)
    provenance["output"] = {
        "file": csv_path.name, "sha256": file_sha256(csv_path),
        "columns": list(OUTCOME_COLUMNS),
    }
    (csv_path.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def compute_cumulative_asr(df, k_values=K_VALUES):
    targets = sorted(df['target_id'].unique())
    rows = []
    for K in k_values:
        reached = df[df['total_bundle_size'] <= K].groupby('target_id')['success_label'].max().reindex(targets, fill_value=0)
        rows.append({'K': K, 'cumulative_asr': reached.mean(), 'n_targets': len(targets), 'n_reached': int(reached.sum())})
    return pd.DataFrame(rows)

def compute_incremental_gains(df, k_values=K_VALUES):
    targets = sorted(df['target_id'].unique())
    exact_success = {K: df[df['total_bundle_size'] == K].groupby('target_id')['success_label'].max() for K in k_values}
    first_reach = {}
    for target in targets:
        successful_sizes = [K for K in k_values if int(exact_success[K].get(target, 0)) == 1]
        first_reach[target] = min(successful_sizes) if successful_sizes else None
    n_targets = len(targets)
    rows = []
    for K in k_values:
        previous = sum((first_reach[t] is not None and first_reach[t] < K for t in targets)) / n_targets
        new = sum((first_reach[t] == K for t in targets)) / n_targets
        rows.append({'K': K, 'previously_reached': previous, 'newly_reached': new, 'cumulative': previous + new})
    return pd.DataFrame(rows)

def use_32b_direct_baseline(df_14b, df_32b, model_name):
    """Replace every 14B K=1 outcome with the target-level 32B K=1 outcome."""
    targets_14b = set(df_14b['target_id'].unique())
    direct_32b = df_32b[df_32b['total_bundle_size'] == 1].groupby('target_id')['success_label'].max()
    missing = sorted(targets_14b - set(direct_32b.index))
    if missing:
        raise ValueError(f'{model_name}: the 32B file lacks K=1 outcomes for {len(missing)} targets present in the 14B file.')
    canonical_k1 = pd.DataFrame({'target_id': sorted(targets_14b), 'total_bundle_size': 1, 'success_label': [int(direct_32b.loc[t]) for t in sorted(targets_14b)], 'model': model_name})
    composed_14b = df_14b[df_14b['total_bundle_size'] >= 2][['target_id', 'total_bundle_size', 'success_label', 'model']].copy()
    adjusted = pd.concat([canonical_k1, composed_14b], ignore_index=True)
    rate_32b = canonical_k1['success_label'].mean()
    rate_adjusted = adjusted[adjusted['total_bundle_size'] == 1]['success_label'].mean()
    if not np.isclose(rate_32b, rate_adjusted):
        raise AssertionError(f'{model_name}: failed to apply the 32B K=1 baseline.')
    print(f'{model_name}: Figure D uses 32B K=1 baseline = {rate_32b:.0%}')
    return adjusted

def build_summaries(outcomes: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute target-level cumulative ASR and first-reached incremental gains.

    A target is reached at a size if at least one candidate succeeded. Figure D
    substitutes each model's 32B direct outcomes into its 14B curve at K=1.
    """
    outcomes = validate_outcomes(outcomes)
    tables = {}
    for generator, models in (("Qwen3-32B", MODELS_32B), ("Qwen3-14B", MODELS_14B)):
        tables[generator] = {}
        for model in models:
            rows = outcomes[
                outcomes["query_generator"].eq(generator)
                & outcomes["downstream_model"].eq(model)
            ].copy()
            rows["success_label"] = rows["n_success"].gt(0).astype(int)
            rows["model"] = model
            tables[generator][model] = rows
    adjusted_14b = {
        model: use_32b_direct_baseline(tables["Qwen3-14B"][model], tables["Qwen3-32B"][model], model)
        for model in MODELS_14B
    }
    return {
        "cumulative_32b": {model: compute_cumulative_asr(rows) for model, rows in tables["Qwen3-32B"].items()},
        "incremental_32b": {model: compute_incremental_gains(rows) for model, rows in tables["Qwen3-32B"].items()},
        "cumulative_14b": {model: compute_cumulative_asr(rows) for model, rows in adjusted_14b.items()},
    }


DOUBLE_COLUMN_WIDTH = 7.16
FIGURE_SIZE = (DOUBLE_COLUMN_WIDTH, 2.72)

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.7,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.75,
    "lines.markersize": 4.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#7A5195",
    "red": "#CC3D3D",
    "gray": "#B8C2CC",
    "dark_gray": "#4B5563",
    "grid": "#D9E0E6",
}

def style_asr_axis(ax, show_ylabels=True):
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_ylabels:
        ax.tick_params(labelleft=False)


def save_figure(fig, output_dir, filename_stem):
    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    return png_path, pdf_path


def plot_incremental_gains(incremental_32b, output_dir):
    # ------------------------------------------------------------
    # Figure A: incremental gains
    # ------------------------------------------------------------

    figure_a_models = [
        "Qwen3-4B",
        "Qwen3-4B-Instruct-2507",
    ]

    figure_a_titles = [
        "(a) Qwen3-4B",
        "(b) Qwen3-4B-Instruct-2507",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
    )

    for panel_index, (ax, model, title) in enumerate(
        zip(axes, figure_a_models, figure_a_titles)
    ):
        sub = incremental_32b[model].sort_values("K")

        ax.bar(
            sub["K"],
            sub["previously_reached"],
            width=0.70,
            color=COLORS["gray"],
            edgecolor="white",
            linewidth=0.35,
        )

        ax.bar(
            sub["K"],
            sub["newly_reached"],
            width=0.70,
            bottom=sub["previously_reached"],
            color=COLORS["blue"],
            edgecolor="white",
            linewidth=0.35,
        )

        for K, previous, new, total in zip(
            sub["K"],
            sub["previously_reached"],
            sub["newly_reached"],
            sub["cumulative"],
        ):
            if total >= 0.95:
                total_y = total - 0.025
                total_va = "top"
            else:
                total_y = total + 0.025
                total_va = "bottom"

            ax.text(
                K,
                total_y,
                f"{total:.0%}",
                ha="center",
                va=total_va,
                fontsize=6.8,
            )

            if new >= 0.035:
                ax.text(
                    K,
                    previous + new / 2,
                    f"+{new * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                )

        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
            pad=7,
        )
        ax.set_xlabel(r"Maximum total bundle size, $K$")

        style_asr_axis(
            ax,
            show_ylabels=(panel_index == 0),
        )

    axes[0].set_ylabel("Cumulative ASR")

    fig.legend(
        handles=[
            Patch(
                facecolor=COLORS["gray"],
                label="Reached at smaller sizes",
            ),
            Patch(
                facecolor=COLORS["blue"],
                label=r"First reached at $K$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.19,
        top=0.80,
        wspace=0.12,
    )

    save_figure(
        fig,
        output_dir,
        "figure_A_incremental_qwen4b_comparison",
    )
    plt.close(fig)


def plot_checkpoint_comparisons(cumulative_32b, output_dir):
    # ------------------------------------------------------------
    # Figures B and C side by side
    # ------------------------------------------------------------

    figure_b_models = [
        "Mistral-7B-Instruct-v0.3",
        "Gemma-3-4B-PT",
        "Qwen3-4B",
    ]

    figure_b_style = {
        "Mistral-7B-Instruct-v0.3": (
            COLORS["blue"],
            "o",
        ),
        "Gemma-3-4B-PT": (
            COLORS["green"],
            "^",
        ),
        "Qwen3-4B": (
            COLORS["orange"],
            "s",
        ),
    }

    figure_c_models = [
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-4B-Instruct-2507",
    ]

    figure_c_style = {
        "Qwen3-1.7B": (
            COLORS["blue"],
            "o",
        ),
        "Qwen3-4B": (
            COLORS["orange"],
            "s",
        ),
        "Qwen3-8B": (
            COLORS["green"],
            "^",
        ),
        "Qwen3-4B-Instruct-2507": (
            COLORS["purple"],
            "D",
        ),
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
    )

    # Figure B: Mistral, Gemma, and Qwen
    for model in figure_b_models:
        sub = cumulative_32b[model].sort_values("K")
        color, marker = figure_b_style[model]

        axes[0].plot(
            sub["K"],
            sub["cumulative_asr"],
            color=color,
            marker=marker,
            label=model,
        )

    # Figure C: Qwen downstream checkpoints
    for model in figure_c_models:
        sub = cumulative_32b[model].sort_values("K")
        color, marker = figure_c_style[model]

        axes[1].plot(
            sub["K"],
            sub["cumulative_asr"],
            color=color,
            marker=marker,
            label=model,
        )

    panel_titles = [
        "(a) Mistral, Gemma, and Qwen",
        "(b) Qwen downstream checkpoints",
    ]

    for panel_index, (ax, title) in enumerate(
        zip(axes, panel_titles)
    ):
        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
            pad=7,
        )
        ax.set_xlabel(r"Maximum total bundle size, $K$")

        style_asr_axis(
            ax,
            show_ylabels=(panel_index == 0),
        )

        ax.legend(
            loc="lower right",
            frameon=False,
            handlelength=1.55,
            borderaxespad=0.5,
        )

    axes[0].set_ylabel("Cumulative ASR")

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.19,
        top=0.88,
        wspace=0.12,
    )

    save_figure(
        fig,
        output_dir,
        "figures_BC_side_by_side_checkpoint_comparisons",
    )
    plt.close(fig)


def plot_generator_comparison(cumulative_32b, cumulative_14b, output_dir):
    # ------------------------------------------------------------
    # Figure D: Qwen3-32B versus Qwen3-14B generators
    # ------------------------------------------------------------

    figure_d_models = [
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-4B-Instruct-2507",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
    )

    for panel_index, (ax, model) in enumerate(
        zip(axes, figure_d_models)
    ):
        results_32b = cumulative_32b[model].sort_values("K")
        results_14b = cumulative_14b[model].sort_values("K")

        # Confirm that the adjusted 14B data uses the 32B K=1 baseline.
        k1_32b = results_32b.loc[
            results_32b["K"] == 1,
            "cumulative_asr",
        ].iloc[0]

        k1_14b = results_14b.loc[
            results_14b["K"] == 1,
            "cumulative_asr",
        ].iloc[0]

        if not np.isclose(k1_32b, k1_14b):
            raise ValueError(
                f"{model}: K=1 differs between generators "
                f"({k1_32b:.1%} versus {k1_14b:.1%})."
            )

        ax.plot(
            results_32b["K"],
            results_32b["cumulative_asr"],
            color=COLORS["blue"],
            marker="o",
            linestyle="-",
            label="Qwen3-32B",
        )

        ax.plot(
            results_14b["K"],
            results_14b["cumulative_asr"],
            color=COLORS["orange"],
            marker="s",
            linestyle="--",
            label="Qwen3-14B",
        )

        ax.set_title(
            f"({chr(97 + panel_index)}) {model}",
            loc="left",
            fontweight="bold",
            pad=7,
        )
        ax.set_xlabel(r"Maximum total bundle size, $K$")

        style_asr_axis(
            ax,
            show_ylabels=(panel_index == 0),
        )

        # Legend at the bottom of each individual subplot.
        ax.legend(
            loc="lower right",
            frameon=False,
            handlelength=1.55,
            borderaxespad=0.45,
        )

    axes[0].set_ylabel("Cumulative ASR")

    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.19,
        top=0.88,
        wspace=0.16,
    )

    save_figure(
        fig,
        output_dir,
        "figure_D_matched_generator_comparison",
    )
    plt.close(fig)


def plot_figures(summaries, output_dir: Path | str = DEFAULT_OUTPUT) -> None:
    """Write the original three figure layouts as PNG and PDF files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        plot_incremental_gains(summaries["incremental_32b"], output_dir)
        plot_checkpoint_comparisons(summaries["cumulative_32b"], output_dir)
        plot_generator_comparison(summaries["cumulative_32b"], summaries["cumulative_14b"], output_dir)


def export_summary_tables(summaries, output_dir: Path | str = DEFAULT_OUTPUT) -> dict[str, pd.DataFrame]:
    """Export the three CSV tables using the original names and column order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("cumulative_32b", "Qwen3-32B", "table_cumulative_asr_32b.csv"),
        ("incremental_32b", "Qwen3-32B", "table_incremental_gains_32b.csv"),
        ("cumulative_14b", "Qwen3-14B_with_32B_K1", "table_cumulative_asr_14b_with_32b_k1.csv"),
    )
    tables = {}
    for key, generator, filename in specifications:
        table = pd.concat([
            rows.assign(model=model, query_generator=generator)
            for model, rows in summaries[key].items()
        ], ignore_index=True)
        table.to_csv(output_dir / filename, index=False)
        tables[filename] = table
    return tables


def reproduce_figures(input_path: Path | str = DEFAULT_INPUT, output_dir: Path | str = DEFAULT_OUTPUT):
    """Validate the included data, save all figures and tables, and return tables."""
    summaries = build_summaries(load_outcomes(input_path))
    plot_figures(summaries, output_dir)
    return export_summary_tables(summaries, output_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Compact per-target outcome CSV.")
    inputs.add_argument("--from-results", type=Path, help="Rebuild compact counts from the ten canonical raw result files.")
    parser.add_argument("--save-outcomes", type=Path, help="Save rebuilt counts here; default: OUTPUT/target_outcomes.csv. Requires --from-results.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for PNG, PDF, and CSV outputs.")
    args = parser.parse_args(argv)
    if args.save_outcomes is not None and args.from_results is None:
        parser.error("--save-outcomes requires --from-results")
    try:
        if args.from_results is not None:
            outcomes, provenance = build_outcomes_from_results(args.from_results)
            save_outcomes(outcomes, provenance, args.save_outcomes or args.output / "target_outcomes.csv")
            summaries = build_summaries(outcomes)
            plot_figures(summaries, args.output)
            tables = export_summary_tables(summaries, args.output)
        else:
            tables = reproduce_figures(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved 3 figures (PNG and PDF) and {len(tables)} CSV tables to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
