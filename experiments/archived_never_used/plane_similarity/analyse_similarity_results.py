from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "plane_similarity"
    / "final"
)

INPUT_CSV = (
    FINAL_DIR
    / "similarity_final_results.csv"
)

SUMMARY_CSV = (
    FINAL_DIR
    / "similarity_final_summary.csv"
)

BY_SCENE_CSV = (
    FINAL_DIR
    / "similarity_final_by_scene.csv"
)

BY_PRIOR_CSV = (
    FINAL_DIR
    / "similarity_final_by_prior.csv"
)

BY_SCENE_PRIOR_CSV = (
    FINAL_DIR
    / "similarity_final_by_scene_prior.csv"
)

JOINT_POSE_CSV = (
    FINAL_DIR
    / "joint_pose_refinement_summary.csv"
)

FIGURE_DIR = (
    FINAL_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Method names
# ============================================================

WEIGHTED_NAME = "Weighted Geometric Cost"
HARD_NAME = "Hard Geometric Thresholds"
DESCRIPTOR_NAME = "Descriptor Similarity"
JOINT_NAME = "Joint Association + Pose Refinement V2"

PAIRWISE_METHODS = [
    WEIGHTED_NAME,
    HARD_NAME,
    DESCRIPTOR_NAME,
]


# ============================================================
# 3. CSV loading
# ============================================================

NUMERIC_FIELDS = {
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "num_gt_matches",
    "num_predicted_pairs",
    "num_gt_matched_a",
    "num_gt_unmatched_a",
    "mean_candidates_per_a",
    "candidate_density",
    "ambiguous_matched_a_rate",
    "missed_matched_a_rate",
    "unmatched_rejection_rate",
    "runtime_ms",
    "num_joint_iterations",
    "prior_rotation_error_deg",
    "final_rotation_error_deg",
    "rotation_improvement_deg",
    "prior_translation_error_m",
    "final_translation_error_m",
    "translation_improvement_m",
}


def parse_number(
    value,
):
    if value is None:
        return None

    value = str(
        value
    ).strip()

    if value == "":
        return None

    try:
        return float(
            value
        )

    except ValueError:
        return None


def load_rows():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find FINAL result file:\n{INPUT_CSV}\n\n"
            "Run run_similarity_benchmark.py first."
        )

    rows = []

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for raw in reader:
            row = dict(
                raw
            )

            for field in NUMERIC_FIELDS:
                if field in row:
                    row[
                        field
                    ] = parse_number(
                        row[
                            field
                        ]
                    )

            rows.append(
                row
            )

    return rows


# ============================================================
# 4. Generic statistical helpers
# ============================================================

def valid_values(
    rows,
    key,
):
    return [
        row[
            key
        ]
        for row in rows
        if row.get(
            key
        ) is not None
    ]


def mean_or_blank(
    rows,
    key,
):
    values = valid_values(
        rows,
        key,
    )

    if not values:
        return ""

    return float(
        np.mean(
            values
        )
    )


def median_or_blank(
    rows,
    key,
):
    values = valid_values(
        rows,
        key,
    )

    if not values:
        return ""

    return float(
        np.median(
            values
        )
    )


def std_or_blank(
    rows,
    key,
):
    values = valid_values(
        rows,
        key,
    )

    if not values:
        return ""

    if len(
        values
    ) < 2:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def micro_metrics(
    rows,
):
    tp = int(
        sum(
            row.get(
                "tp"
            )
            or 0
            for row in rows
        )
    )

    fp = int(
        sum(
            row.get(
                "fp"
            )
            or 0
            for row in rows
        )
    )

    fn = int(
        sum(
            row.get(
                "fn"
            )
            or 0
            for row in rows
        )
    )

    precision = (
        tp
        / (
            tp
            + fp
        )
        if (
            tp
            + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp
        / (
            tp
            + fn
        )
        if (
            tp
            + fn
        ) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    return {
        "total_tp":
            tp,

        "total_fp":
            fp,

        "total_fn":
            fn,

        "micro_precision":
            float(
                precision
            ),

        "micro_recall":
            float(
                recall
            ),

        "micro_f1":
            float(
                f1
            ),
    }


# ============================================================
# 5. Aggregate one group
# ============================================================

def aggregate_group(
    rows,
):
    if not rows:
        return {}

    micro = micro_metrics(
        rows
    )

    result = {
        "num_cases":
            len(
                rows
            ),

        "evaluation_scope":
            rows[
                0
            ][
                "evaluation_scope"
            ],

        "mean_precision":
            mean_or_blank(
                rows,
                "precision",
            ),

        "mean_recall":
            mean_or_blank(
                rows,
                "recall",
            ),

        "mean_f1":
            mean_or_blank(
                rows,
                "f1",
            ),

        **micro,

        "mean_runtime_ms":
            mean_or_blank(
                rows,
                "runtime_ms",
            ),

        "median_runtime_ms":
            median_or_blank(
                rows,
                "runtime_ms",
            ),

        "std_runtime_ms":
            std_or_blank(
                rows,
                "runtime_ms",
            ),

        # Candidate-generation diagnostics.
        "mean_candidates_per_a":
            mean_or_blank(
                rows,
                "mean_candidates_per_a",
            ),

        "mean_candidate_density":
            mean_or_blank(
                rows,
                "candidate_density",
            ),

        "mean_ambiguous_matched_a_rate":
            mean_or_blank(
                rows,
                "ambiguous_matched_a_rate",
            ),

        "mean_missed_matched_a_rate":
            mean_or_blank(
                rows,
                "missed_matched_a_rate",
            ),

        "mean_unmatched_rejection_rate":
            mean_or_blank(
                rows,
                "unmatched_rejection_rate",
            ),

        # Joint pose diagnostics.
        "mean_joint_iterations":
            mean_or_blank(
                rows,
                "num_joint_iterations",
            ),

        "mean_prior_rotation_error_deg":
            mean_or_blank(
                rows,
                "prior_rotation_error_deg",
            ),

        "mean_final_rotation_error_deg":
            mean_or_blank(
                rows,
                "final_rotation_error_deg",
            ),

        "mean_rotation_improvement_deg":
            mean_or_blank(
                rows,
                "rotation_improvement_deg",
            ),

        "mean_prior_translation_error_m":
            mean_or_blank(
                rows,
                "prior_translation_error_m",
            ),

        "mean_final_translation_error_m":
            mean_or_blank(
                rows,
                "final_translation_error_m",
            ),

        "mean_translation_improvement_m":
            mean_or_blank(
                rows,
                "translation_improvement_m",
            ),
    }

    return result


# ============================================================
# 6. Grouping helpers
# ============================================================

def group_rows(
    rows,
    keys,
):
    groups = defaultdict(
        list
    )

    for row in rows:

        key = tuple(
            row[
                field
            ]
            for field in keys
        )

        groups[
            key
        ].append(
            row
        )

    return groups


def write_aggregate_csv(
    rows,
    group_keys,
    output_path,
):
    groups = group_rows(
        rows,
        group_keys,
    )

    output_rows = []

    for key, group in sorted(
        groups.items(),
        key=lambda item: item[
            0
        ],
    ):
        aggregate = aggregate_group(
            group
        )

        output_row = {
            field:
                value
            for field, value in zip(
                group_keys,
                key,
            )
        }

        output_row.update(
            aggregate
        )

        output_rows.append(
            output_row
        )

    if not output_rows:
        return

    fieldnames = list(
        output_rows[
            0
        ].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )


# ============================================================
# 7. Dedicated Joint pose summary
# ============================================================

def write_joint_pose_summary(
    rows,
):
    joint_rows = [
        row
        for row in rows
        if row[
            "method"
        ] == JOINT_NAME
    ]

    groups = group_rows(
        joint_rows,
        [
            "prior_level",
        ],
    )

    output_rows = []

    for (
        prior_level,
    ), group in sorted(
        groups.items()
    ):

        output_rows.append(
            {
                "prior_level":
                    prior_level,

                "num_cases":
                    len(
                        group
                    ),

                "mean_f1":
                    mean_or_blank(
                        group,
                        "f1",
                    ),

                "mean_prior_rotation_error_deg":
                    mean_or_blank(
                        group,
                        "prior_rotation_error_deg",
                    ),

                "mean_final_rotation_error_deg":
                    mean_or_blank(
                        group,
                        "final_rotation_error_deg",
                    ),

                "mean_rotation_improvement_deg":
                    mean_or_blank(
                        group,
                        "rotation_improvement_deg",
                    ),

                "mean_prior_translation_error_m":
                    mean_or_blank(
                        group,
                        "prior_translation_error_m",
                    ),

                "mean_final_translation_error_m":
                    mean_or_blank(
                        group,
                        "final_translation_error_m",
                    ),

                "mean_translation_improvement_m":
                    mean_or_blank(
                        group,
                        "translation_improvement_m",
                    ),

                "mean_iterations":
                    mean_or_blank(
                        group,
                        "num_joint_iterations",
                    ),
            }
        )

    if not output_rows:
        return

    with open(
        JOINT_POSE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )


# ============================================================
# 8. Figure helpers
# ============================================================

PRIOR_ORDER = [
    "exact",
    "mild",
    "moderate",
]


def mean_metric_for(
    rows,
    method,
    prior_level,
    metric,
):
    subset = [
        row
        for row in rows
        if (
            row[
                "method"
            ] == method
            and row[
                "prior_level"
            ] == prior_level
        )
    ]

    values = valid_values(
        subset,
        metric,
    )

    if not values:
        return np.nan

    return float(
        np.mean(
            values
        )
    )


def plot_pairwise_f1_by_prior(
    rows,
):
    x = np.arange(
        len(
            PRIOR_ORDER
        )
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    for index, method in enumerate(
        PAIRWISE_METHODS
    ):
        values = [
            mean_metric_for(
                rows,
                method,
                prior,
                "f1",
            )
            for prior in PRIOR_ORDER
        ]

        ax.bar(
            x
            + (
                index
                - 1
            )
            * width,
            values,
            width=width,
            label=method,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        PRIOR_ORDER
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_xlabel(
        "Pose-prior level"
    )

    ax.set_ylabel(
        "Mean candidate-pair F1"
    )

    ax.set_title(
        "Pairwise Similarity Performance vs Pose Prior"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "pairwise_similarity_f1_by_prior.png",
        dpi=200,
    )

    plt.close(
        fig
    )


def plot_pairwise_ambiguity_by_prior(
    rows,
):
    x = np.arange(
        len(
            PRIOR_ORDER
        )
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    for index, method in enumerate(
        PAIRWISE_METHODS
    ):
        values = [
            mean_metric_for(
                rows,
                method,
                prior,
                "ambiguous_matched_a_rate",
            )
            for prior in PRIOR_ORDER
        ]

        ax.bar(
            x
            + (
                index
                - 1
            )
            * width,
            values,
            width=width,
            label=method,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        PRIOR_ORDER
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_xlabel(
        "Pose-prior level"
    )

    ax.set_ylabel(
        "Ambiguous matched-A rate"
    )

    ax.set_title(
        "Candidate Ambiguity vs Pose Prior"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "pairwise_similarity_ambiguity_by_prior.png",
        dpi=200,
    )

    plt.close(
        fig
    )


def plot_pairwise_f1_by_scene(
    rows,
):
    scenes = sorted(
        {
            row[
                "scene_type"
            ]
            for row in rows
        }
    )

    x = np.arange(
        len(
            scenes
        )
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for index, method in enumerate(
        PAIRWISE_METHODS
    ):
        values = []

        for scene in scenes:
            subset = [
                row
                for row in rows
                if (
                    row[
                        "method"
                    ] == method
                    and row[
                        "scene_type"
                    ] == scene
                )
            ]

            values.append(
                mean_or_blank(
                    subset,
                    "f1",
                )
                or 0.0
            )

        ax.bar(
            x
            + (
                index
                - 1
            )
            * width,
            values,
            width=width,
            label=method,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        scenes,
        rotation=15,
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_xlabel(
        "Scene type"
    )

    ax.set_ylabel(
        "Mean candidate-pair F1"
    )

    ax.set_title(
        "Pairwise Similarity Performance by Scene"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "pairwise_similarity_f1_by_scene.png",
        dpi=200,
    )

    plt.close(
        fig
    )


def plot_joint_f1_by_prior(
    rows,
):
    values = [
        mean_metric_for(
            rows,
            JOINT_NAME,
            prior,
            "f1",
        )
        for prior in PRIOR_ORDER
    ]

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    ax.bar(
        PRIOR_ORDER,
        values,
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_xlabel(
        "Pose-prior level"
    )

    ax.set_ylabel(
        "Mean final-correspondence F1"
    )

    ax.set_title(
        "Joint Association Correspondence Performance"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "joint_correspondence_f1_by_prior.png",
        dpi=200,
    )

    plt.close(
        fig
    )


def plot_joint_rotation_error(
    rows,
):
    prior_values = [
        mean_metric_for(
            rows,
            JOINT_NAME,
            prior,
            "prior_rotation_error_deg",
        )
        for prior in PRIOR_ORDER
    ]

    final_values = [
        mean_metric_for(
            rows,
            JOINT_NAME,
            prior,
            "final_rotation_error_deg",
        )
        for prior in PRIOR_ORDER
    ]

    x = np.arange(
        len(
            PRIOR_ORDER
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    ax.bar(
        x
        - width / 2,
        prior_values,
        width=width,
        label="Prior",
    )

    ax.bar(
        x
        + width / 2,
        final_values,
        width=width,
        label="Refined",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        PRIOR_ORDER
    )

    ax.set_xlabel(
        "Pose-prior level"
    )

    ax.set_ylabel(
        "Mean rotation error (deg)"
    )

    ax.set_title(
        "Joint Pose Refinement: Rotation Error"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "joint_rotation_error_by_prior.png",
        dpi=200,
    )

    plt.close(
        fig
    )


def plot_joint_translation_error(
    rows,
):
    prior_values = [
        mean_metric_for(
            rows,
            JOINT_NAME,
            prior,
            "prior_translation_error_m",
        )
        for prior in PRIOR_ORDER
    ]

    final_values = [
        mean_metric_for(
            rows,
            JOINT_NAME,
            prior,
            "final_translation_error_m",
        )
        for prior in PRIOR_ORDER
    ]

    x = np.arange(
        len(
            PRIOR_ORDER
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    ax.bar(
        x
        - width / 2,
        prior_values,
        width=width,
        label="Prior",
    )

    ax.bar(
        x
        + width / 2,
        final_values,
        width=width,
        label="Refined",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        PRIOR_ORDER
    )

    ax.set_xlabel(
        "Pose-prior level"
    )

    ax.set_ylabel(
        "Mean translation error (m)"
    )

    ax.set_title(
        "Joint Pose Refinement: Translation Error"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "joint_translation_error_by_prior.png",
        dpi=200,
    )

    plt.close(
        fig
    )


# ============================================================
# 9. Console summary
# ============================================================

def print_console_summary(
    rows,
):
    print()
    print(
        "=" * 78
    )

    print(
        "FINAL SIMILARITY SUMMARY"
    )

    print(
        "=" * 78
    )

    for method in [
        *PAIRWISE_METHODS,
        JOINT_NAME,
    ]:

        subset = [
            row
            for row in rows
            if row[
                "method"
            ] == method
        ]

        aggregate = aggregate_group(
            subset
        )

        print()
        print(
            method
        )

        print(
            f"  Evaluation scope : "
            f"{aggregate['evaluation_scope']}"
        )

        print(
            f"  Mean F1          : "
            f"{aggregate['mean_f1']:.4f}"
        )

        print(
            f"  Micro F1         : "
            f"{aggregate['micro_f1']:.4f}"
        )

        print(
            f"  Mean runtime     : "
            f"{aggregate['mean_runtime_ms']:.3f} ms"
        )

        if method in PAIRWISE_METHODS:

            print(
                f"  Candidates / A   : "
                f"{aggregate['mean_candidates_per_a']:.4f}"
            )

            print(
                f"  Ambiguity rate   : "
                f"{aggregate['mean_ambiguous_matched_a_rate']:.4f}"
            )

            print(
                f"  Missed rate      : "
                f"{aggregate['mean_missed_matched_a_rate']:.4f}"
            )

            unmatched = aggregate[
                "mean_unmatched_rejection_rate"
            ]

            if unmatched != "":
                print(
                    f"  Unmatched reject : "
                    f"{unmatched:.4f}"
                )

        else:

            print(
                f"  Final rot. error : "
                f"{aggregate['mean_final_rotation_error_deg']:.4f} deg"
            )

            print(
                f"  Rot. improvement : "
                f"{aggregate['mean_rotation_improvement_deg']:+.4f} deg"
            )

            print(
                f"  Final trans err. : "
                f"{aggregate['mean_final_translation_error_m']:.5f} m"
            )

            print(
                f"  Trans improvement: "
                f"{aggregate['mean_translation_improvement_m']:+.5f} m"
            )


# ============================================================
# 10. Main
# ============================================================

def main():
    rows = load_rows()

    expected_rows = (
        90
        * 4
    )

    if len(
        rows
    ) != expected_rows:
        print(
            "WARNING:"
        )

        print(
            f"Expected {expected_rows} rows "
            f"but found {len(rows)}."
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    write_aggregate_csv(
        rows,
        [
            "method",
        ],
        SUMMARY_CSV,
    )

    # --------------------------------------------------------
    # Robustness breakdowns
    # --------------------------------------------------------

    write_aggregate_csv(
        rows,
        [
            "method",
            "scene_type",
        ],
        BY_SCENE_CSV,
    )

    write_aggregate_csv(
        rows,
        [
            "method",
            "prior_level",
        ],
        BY_PRIOR_CSV,
    )

    write_aggregate_csv(
        rows,
        [
            "method",
            "scene_type",
            "prior_level",
        ],
        BY_SCENE_PRIOR_CSV,
    )

    # --------------------------------------------------------
    # Joint-specific pose analysis
    # --------------------------------------------------------

    write_joint_pose_summary(
        rows
    )

    # --------------------------------------------------------
    # Figures
    #
    # Pairwise methods and Joint are deliberately plotted
    # separately because they have different evaluation scope.
    # --------------------------------------------------------

    plot_pairwise_f1_by_prior(
        rows
    )

    plot_pairwise_ambiguity_by_prior(
        rows
    )

    plot_pairwise_f1_by_scene(
        rows
    )

    plot_joint_f1_by_prior(
        rows
    )

    plot_joint_rotation_error(
        rows
    )

    plot_joint_translation_error(
        rows
    )

    print_console_summary(
        rows
    )

    print()
    print(
        "=" * 78
    )

    print(
        "ANALYSIS FILES"
    )

    print(
        "=" * 78
    )

    print(
        SUMMARY_CSV
    )

    print(
        BY_SCENE_CSV
    )

    print(
        BY_PRIOR_CSV
    )

    print(
        BY_SCENE_PRIOR_CSV
    )

    print(
        JOINT_POSE_CSV
    )

    print(
        FIGURE_DIR
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The first three methods measure candidate-generation "
        "quality before assignment."
    )

    print(
        "Joint V2 measures final one-to-one correspondences "
        "together with pose refinement."
    )

    print(
        "Do not interpret their ambiguity metrics as directly "
        "comparable; Joint has no candidate-ambiguity metric."
    )


if __name__ == "__main__":
    main()
