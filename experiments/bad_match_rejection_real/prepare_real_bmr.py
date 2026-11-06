
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

REQUIRED_FILES = (
    "bmr_assigned_pairs.csv",
    "bmr_gt_labels_EVALUATION_ONLY.csv",
    "bmr_plane_features_original_frames.csv",
)

BASELINE_GATING = "contour_point_gating"
BASELINE_SIMILARITY = "planematch_similarity"
BASELINE_ASSIGNMENT = "jonker_volgenant"

EXPECTED_MATCHES = 26
EXPECTED_CASES = 4

DEFAULT_SEED = 20261105


# ============================================================
# 2. Paths and CSV loading
# ============================================================

def experiments_dir():
    here = Path(__file__).resolve().parent

    for parent in (here, *here.parents):
        if parent.name.lower() == "experiments":
            return parent

    return here


def locate_csv(folder, canonical_name):
    """
    Accept both canonical names and Windows copies:
        bmr_assigned_pairs.csv
        bmr_assigned_pairs(1).csv
    """
    exact = folder / canonical_name

    if exact.is_file():
        return exact

    stem = Path(canonical_name).stem
    suffix = Path(canonical_name).suffix

    matches = sorted(
        folder.glob(f"{stem}(*){suffix}")
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple copies of {canonical_name} found:\n"
            + "\n".join(str(p) for p in matches)
        )

    return None


def find_input_dir(user_path=None):
    if user_path is not None:
        folder = Path(user_path).expanduser().resolve()

        missing = [
            name for name in REQUIRED_FILES
            if locate_csv(folder, name) is None
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing input files in {folder}: {missing}"
            )

        return folder

    here = Path(__file__).resolve().parent
    root = experiments_dir()

    candidates = [
        here,
        Path.cwd(),
        here / "input",
        root / "data" / "Lab_Book_11_Confirmed_Selected_Baseline_26",
        root / "data" / "Lab_Book_11_Selected_Baseline_to_BMR",
        root / "data" / "Attempt_F_to_Bad_Match_Rejection",
        root / "data",
    ]

    for folder in candidates:
        if folder.is_dir() and all(
            locate_csv(folder, name) is not None
            for name in REQUIRED_FILES
        ):
            return folder.resolve()

    raise FileNotFoundError(
        "Cannot locate the three required CSV files.\n"
        "Place them beside prepare_real_bmr.py, "
        "or specify --input-dir."
    )


def read_tables(folder):
    paths = [
        locate_csv(folder, name)
        for name in REQUIRED_FILES
    ]

    print("\nINPUT FILES:")
    for path in paths:
        print(f"  {path}")

    pairs = pd.read_csv(paths[0])
    gt = pd.read_csv(paths[1])
    features = pd.read_csv(paths[2])

    pair_columns = {
        "match_id",
        "case_id",
        "source_frame",
        "target_frame",
        "source_plane_id",
        "target_plane_id",
        "gating_algorithm",
        "similarity_algorithm",
        "assignment_algorithm",
    }

    gt_columns = {
        "match_id",
        "case_id",
        "source_plane_id",
        "target_plane_id",
        "gt_match_label",
        "source_surface_id",
        "target_surface_id",
    }

    feature_columns = {
        "frame_id",
        "plane_id",
        "normal_x_original",
        "normal_y_original",
        "normal_z_original",
        "centroid_x_original_m",
        "centroid_y_original_m",
        "centroid_z_original_m",
    }

    for name, df, required in [
        ("Assignment", pairs, pair_columns),
        ("Ground Truth", gt, gt_columns),
        ("Plane Features", features, feature_columns),
    ]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

    if pairs["match_id"].duplicated().any():
        raise ValueError("Duplicate Assignment match_id")

    if gt["match_id"].duplicated().any():
        raise ValueError("Duplicate GT match_id")

    if features.duplicated(
        ["frame_id", "plane_id"]
    ).any():
        raise ValueError("Duplicate original plane features")

    merged = pairs.merge(
        gt,
        on="match_id",
        how="left",
        suffixes=("", "_gt"),
        validate="one_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError("Some Assignment matches lack GT labels")

    for column in (
        "case_id",
        "source_plane_id",
        "target_plane_id",
    ):
        a = merged[column].astype(str)
        b = merged[f"{column}_gt"].astype(str)

        if not a.eq(b).all():
            raise ValueError(
                f"GT and Assignment disagree on {column}"
            )

    merged["gt_match_label"] = pd.to_numeric(
        merged["gt_match_label"],
        errors="raise",
    ).astype(int)

    if not merged["gt_match_label"].isin([0, 1]).all():
        raise ValueError("Invalid GT labels")

    if merged[
        ["source_surface_id", "target_surface_id"]
    ].isna().any().any():
        raise ValueError("Missing Surface IDs")

    expected_labels = (
        merged["source_surface_id"].astype(str)
        == merged["target_surface_id"].astype(str)
    ).astype(int)

    if not merged["gt_match_label"].eq(
        expected_labels
    ).all():
        raise ValueError(
            "GT labels disagree with Surface IDs"
        )

    return merged.drop(columns="_merge"), features


# ============================================================
# 3. Select the Lab Book 11 baseline
# ============================================================

def select_baseline(merged):
    gating = (
        merged["gating_algorithm"]
        .astype(str).str.strip().str.lower()
    )

    similarity = (
        merged["similarity_algorithm"]
        .astype(str).str.strip().str.lower()
    )

    assignment = (
        merged["assignment_algorithm"]
        .astype(str).str.strip().str.lower()
    )

    mask = (
        gating.eq(BASELINE_GATING)
        & similarity.eq(BASELINE_SIMILARITY)
        & assignment.eq(BASELINE_ASSIGNMENT)
    )

    selected = merged.loc[mask].copy()

    print("\n" + "=" * 65)
    print("LAB BOOK 11 BASELINE VERIFICATION")
    print("=" * 65)

    print(f"All Assignment records : {len(merged)}")
    print(f"Selected matches       : {len(selected)}")
    print(f"Selected cases         : {selected.case_id.nunique()}")

    if (
        len(selected) != EXPECTED_MATCHES
        or selected.case_id.nunique() != EXPECTED_CASES
    ):
        raise ValueError(
            "The selected baseline does not contain "
            f"{EXPECTED_MATCHES} matches in "
            f"{EXPECTED_CASES} cases.\n"
            "Check the source CSV files."
        )

    correct = int(
        selected["gt_match_label"].eq(1).sum()
    )

    incorrect = int(
        selected["gt_match_label"].eq(0).sum()
    )

    print(f"GT consistent          : {correct}")
    print(f"GT inconsistent        : {incorrect}")

    if incorrect:
        bad = selected.loc[
            selected["gt_match_label"].eq(0),
            [
                "match_id",
                "case_id",
                "source_surface_id",
                "target_surface_id",
            ],
        ]

        raise ValueError(
            "A selected Baseline match disagrees with GT:\n"
            + bad.to_string(index=False)
        )

    print("Baseline verification PASSED")

    return selected


# ============================================================
# 4. Build original-frame plane arrays
# ============================================================

def build_plane_inventory(features):
    inventory = {}

    for frame_id, group in features.groupby("frame_id"):
        group = group.sort_values("plane_id")

        plane_ids = group["plane_id"].to_numpy(
            dtype=np.int64
        )

        normals = group[
            [
                "normal_x_original",
                "normal_y_original",
                "normal_z_original",
            ]
        ].to_numpy(dtype=float)

        centroids = group[
            [
                "centroid_x_original_m",
                "centroid_y_original_m",
                "centroid_z_original_m",
            ]
        ].to_numpy(dtype=float)

        if not (
            np.isfinite(normals).all()
            and np.isfinite(centroids).all()
        ):
            raise ValueError(
                f"Invalid geometry in frame {frame_id}"
            )

        lengths = np.linalg.norm(
            normals,
            axis=1,
        )

        if np.any(lengths < 1e-9):
            raise ValueError(
                f"Zero-length normal in frame {frame_id}"
            )

        normals = normals / lengths[:, None]

        # Plane equation: n^T x + d = 0
        # Reconstruct d from a point on each detected plane.
        offsets = -np.einsum(
            "ij,ij->i",
            normals,
            centroids,
        )

        planes = np.column_stack(
            [normals, offsets]
        )

        inventory[int(frame_id)] = {
            "planes": planes,
            "centroids": centroids,
            "plane_ids": plane_ids,
            "lookup": {
                int(pid): i
                for i, pid in enumerate(plane_ids)
            },
        }

    return inventory


def verify_geometry(selected, inventory):
    normal_columns = [
        "target_normal_x_target_frame",
        "target_normal_y_target_frame",
        "target_normal_z_target_frame",
    ]

    offset_column = "target_offset_target_frame_m"

    if not set(
        normal_columns + [offset_column]
    ).issubset(selected.columns):
        print("Geometry reference columns unavailable")
        return

    worst_normal = 0.0
    worst_offset = 0.0

    for row in selected.itertuples(index=False):
        frame = int(row.target_frame)
        pid = int(row.target_plane_id)

        inv = inventory[frame]

        original = inv["planes"][
            inv["lookup"][pid]
        ]

        observed_normal = np.array(
            [
                getattr(row, column)
                for column in normal_columns
            ],
            dtype=float,
        )

        observed_offset = float(
            getattr(row, offset_column)
        )

        if np.dot(
            observed_normal,
            original[:3],
        ) < 0:
            observed_normal *= -1
            observed_offset *= -1

        worst_normal = max(
            worst_normal,
            float(
                np.linalg.norm(
                    observed_normal - original[:3]
                )
            ),
        )

        worst_offset = max(
            worst_offset,
            abs(observed_offset - original[3]),
        )

    if worst_normal > 1e-4 or worst_offset > 1e-4:
        raise ValueError(
            "Reconstructed plane geometry is inconsistent "
            "with the saved target-frame parameters.\n"
            f"Normal difference: {worst_normal}\n"
            f"Offset difference: {worst_offset}"
        )

    print("\nOriginal-frame geometry verification PASSED")


# ============================================================
# 5. Convert plane IDs to array indices
# ============================================================

def prepare_case(group, inventory):
    group = group.sort_values(
        "match_id",
        kind="stable",
    ).reset_index(drop=True)

    frame_a = int(group.source_frame.iloc[0])
    frame_b = int(group.target_frame.iloc[0])

    if (
        group.source_frame.nunique() != 1
        or group.target_frame.nunique() != 1
    ):
        raise ValueError("Mixed frames inside a case")

    if (
        group.source_plane_id.duplicated().any()
        or group.target_plane_id.duplicated().any()
    ):
        raise ValueError("One-to-one Assignment violation")

    inv_a = inventory[frame_a]
    inv_b = inventory[frame_b]

    assigned = np.array(
        [
            [
                inv_a["lookup"][int(source)],
                inv_b["lookup"][int(target)],
            ]
            for source, target in zip(
                group.source_plane_id,
                group.target_plane_id,
            )
        ],
        dtype=np.int64,
    )

    return (
        group,
        frame_a,
        frame_b,
        assigned,
    )


# ============================================================
# 6. Controlled mismatch generation
# ============================================================

def create_controlled_mismatches(
    group,
    assigned_pairs,
    ratio,
    rng,
):
    """
    Swap assigned targets without modifying the plane geometry.

    Preserve one-to-one assignment and require each corrupted
    correspondence to have a different physical Surface ID.
    """
    n = len(assigned_pairs)

    if n < 2:
        raise ValueError(
            "At least two matches required for corruption"
        )

    bad_count = min(
        n,
        max(2, int(round(ratio * n))),
    )

    source_surfaces = (
        group.source_surface_id.astype(str).tolist()
    )

    target_surfaces = (
        group.target_surface_id.astype(str).tolist()
    )

    selections = list(
        combinations(range(n), bad_count)
    )

    for selection_index in rng.permutation(
        len(selections)
    ):
        selected = np.array(
            selections[int(selection_index)],
            dtype=int,
        )

        for _ in range(120):
            permuted = rng.permutation(selected)

            valid = all(
                source_surfaces[int(s)]
                != target_surfaces[int(t)]
                for s, t in zip(selected, permuted)
            )

            if not valid:
                continue

            corrupted = assigned_pairs.copy()

            corrupted[selected, 1] = (
                assigned_pairs[permuted, 1]
            )

            updated_targets = target_surfaces.copy()

            for src_index, target_index in zip(
                selected,
                permuted,
            ):
                updated_targets[int(src_index)] = (
                    target_surfaces[int(target_index)]
                )

            labels = np.array(
                [
                    a == b
                    for a, b in zip(
                        source_surfaces,
                        updated_targets,
                    )
                ],
                dtype=bool,
            )

            if int(np.sum(~labels)) != bad_count:
                raise AssertionError(
                    "Unexpected controlled corruption count"
                )

            return (
                corrupted,
                labels,
                source_surfaces,
                updated_targets,
            )

    raise RuntimeError(
        "Unable to generate valid controlled mismatches "
        f"for case {group.case_id.iloc[0]}"
    )


# ============================================================
# 7. Save one prepared case
# ============================================================

def save_case(
    output_dir,
    case_name,
    original_case,
    scenario,
    group,
    inventory,
    assigned_pairs,
    gt_labels,
    source_surfaces,
    target_surfaces,
    seed,
    requested_ratio,
    overwrite,
):
    frame_a = int(group.source_frame.iloc[0])
    frame_b = int(group.target_frame.iloc[0])

    inv_a = inventory[frame_a]
    inv_b = inventory[frame_b]

    if len(assigned_pairs) != len(group):
        raise ValueError("Assignment count mismatch")

    if (
        len(np.unique(assigned_pairs[:, 0]))
        != len(assigned_pairs)
        or len(np.unique(assigned_pairs[:, 1]))
        != len(assigned_pairs)
    ):
        raise ValueError(
            "One-to-one matching constraint violated"
        )

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        case_name,
    )

    output_file = output_dir / f"{safe_name}.npz"

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"{output_file} exists. Use --overwrite."
        )

    # Algorithm input: NO Ground Truth.
    np.savez_compressed(
        output_file,
        planes_a=inv_a["planes"],
        planes_b=inv_b["planes"],
        centroids_a=inv_a["centroids"],
        centroids_b=inv_b["centroids"],
        source_plane_ids=inv_a["plane_ids"],
        target_plane_ids=inv_b["plane_ids"],
        assigned_pairs=assigned_pairs,
        case_id=np.array(case_name),
        original_case_id=np.array(original_case),
        scenario=np.array(scenario),
        source_frame=np.array(frame_a),
        target_frame=np.array(frame_b),
        seed=np.array(seed, dtype=np.int64),
    )

    labels = []

    for pair_index, ((ia, ib), label) in enumerate(
        zip(assigned_pairs, gt_labels)
    ):
        labels.append(
            {
                "prepared_case_id": case_name,
                "pair_index": pair_index,
                "original_case_id": original_case,
                "scenario": scenario,
                "source_frame": frame_a,
                "target_frame": frame_b,
                "source_plane_id": int(
                    inv_a["plane_ids"][ia]
                ),
                "target_plane_id": int(
                    inv_b["plane_ids"][ib]
                ),
                "gt_match_label": int(label),
                "source_surface_id": source_surfaces[pair_index],
                "target_surface_id": target_surfaces[pair_index],
            }
        )

    manifest = {
        "prepared_case_id": case_name,
        "original_case_id": original_case,
        "scenario": scenario,
        "source_frame": frame_a,
        "target_frame": frame_b,
        "data_file": output_file.name,
        "n_assigned": len(assigned_pairs),
        "n_known_bad_EVAL_ONLY": int(
            np.sum(~gt_labels)
        ),
        "requested_corruption_ratio": requested_ratio,
        "actual_corruption_ratio_EVAL_ONLY": float(
            np.mean(~gt_labels)
        ),
        "seed": seed,
    }

    return manifest, labels


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare RealSense BMR inputs from Lab Book 11"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            experiments_dir()
            / "data"
            / "bad_match_rejection"
            / "real"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--corruption-ratios",
        default="0.25,0.50",
    )

    parser.add_argument(
        "--skip-corruption",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    ratios = [
        float(value)
        for value in args.corruption_ratios.split(",")
        if value.strip()
    ]

    if not all(0 < ratio <= 1 for ratio in ratios):
        parser.error("Corruption ratios must be in (0,1]")

    input_dir = find_input_dir(args.input_dir)
    output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("LAB BOOK 12 - REAL BAD-MATCH REJECTION PREPARATION")
    print("=" * 65)

    merged, features = read_tables(input_dir)

    baseline = select_baseline(merged)

    inventory = build_plane_inventory(features)

    verify_geometry(baseline, inventory)

    manifest_file = (
        output_dir
        / "case_manifest_EVALUATION_METADATA.csv"
    )

    labels_file = (
        output_dir
        / "labels_EVALUATION_ONLY.csv"
    )

    if args.overwrite and manifest_file.exists():
        old_manifest = pd.read_csv(manifest_file)

        if "data_file" in old_manifest.columns:
            for name in old_manifest[
                "data_file"
            ].dropna().astype(str).unique():
                # Delete only files listed in the old manifest.
                if Path(name).name != name or not name.endswith(".npz"):
                    raise ValueError(
                        "Unsafe filename in existing manifest"
                    )

                old_file = output_dir / name

                if old_file.is_file():
                    old_file.unlink()

    rng_master = np.random.default_rng(args.seed)

    manifest_rows = []
    evaluation_rows = []

    for case_id, raw_group in baseline.groupby(
        "case_id",
        sort=True,
    ):
        (
            group,
            frame_a,
            frame_b,
            original_pairs,
        ) = prepare_case(raw_group, inventory)

        baseline_labels = (
            group.gt_match_label.to_numpy(dtype=bool)
        )

        source_surfaces = (
            group.source_surface_id.astype(str).tolist()
        )

        target_surfaces = (
            group.target_surface_id.astype(str).tolist()
        )

        # ----------------------------------------------
        # A. Original selected baseline
        # ----------------------------------------------

        case_name = f"{case_id}__baseline"

        row, labels = save_case(
            output_dir=output_dir,
            case_name=case_name,
            original_case=str(case_id),
            scenario="baseline",
            group=group,
            inventory=inventory,
            assigned_pairs=original_pairs,
            gt_labels=baseline_labels,
            source_surfaces=source_surfaces,
            target_surfaces=target_surfaces,
            seed=-1,
            requested_ratio=0.0,
            overwrite=args.overwrite,
        )

        manifest_rows.append(row)
        evaluation_rows.extend(labels)

        # ----------------------------------------------
        # B. Controlled mismatch cases
        # ----------------------------------------------

        if args.skip_corruption:
            continue

        for ratio in ratios:
            for repeat in range(args.repeats):
                case_seed = int(
                    rng_master.integers(
                        0,
                        2**32 - 1,
                    )
                )

                rng = np.random.default_rng(
                    case_seed
                )

                (
                    corrupted_pairs,
                    corrupted_labels,
                    corrupted_source_surfaces,
                    corrupted_target_surfaces,
                ) = create_controlled_mismatches(
                    group=group,
                    assigned_pairs=original_pairs,
                    ratio=ratio,
                    rng=rng,
                )

                suffix = (
                    f"swap{int(round(ratio * 100)):03d}"
                    f"_r{repeat:02d}"
                )

                corrupted_name = (
                    f"{case_id}__{suffix}"
                )

                row, labels = save_case(
                    output_dir=output_dir,
                    case_name=corrupted_name,
                    original_case=str(case_id),
                    scenario="controlled_corruption",
                    group=group,
                    inventory=inventory,
                    assigned_pairs=corrupted_pairs,
                    gt_labels=corrupted_labels,
                    source_surfaces=corrupted_source_surfaces,
                    target_surfaces=corrupted_target_surfaces,
                    seed=case_seed,
                    requested_ratio=ratio,
                    overwrite=args.overwrite,
                )

                manifest_rows.append(row)
                evaluation_rows.extend(labels)

    for path in (manifest_file, labels_file):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite."
            )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    evaluation_df = pd.DataFrame(
        evaluation_rows
    )

    manifest_df.to_csv(
        manifest_file,
        index=False,
    )

    evaluation_df.to_csv(
        labels_file,
        index=False,
    )

    print("\n" + "=" * 65)
    print("PREPARATION COMPLETE")
    print("=" * 65)

    print(f"Output directory: {output_dir}")
    print(f"Prepared cases  : {len(manifest_df)}")

    print(
        manifest_df.groupby("scenario")[
            "n_assigned"
        ].agg(["count", "sum"])
    )

    print(f"\nManifest : {manifest_file}")
    print(f"GT labels: {labels_file}")

    print(
        "\nNo Bad-Match Rejection algorithms were executed."
    )

    print(
        "No Ground Truth labels were included in the NPZ inputs."
    )


if __name__ == "__main__":
    main()
