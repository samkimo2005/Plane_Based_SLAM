"""
generate_candidate_pairs.py

Generate Candidate Gating pairs from the exported per-frame plane files.

Input:
    results/candidate_gating/tum_plane_features_with_gt_ids/frame_XXXX.npz

For every consecutive frame pair (t, t+1), generate the full Cartesian
product of detected planes:

    plane_i(t) x plane_j(t+1)

Ground-truth label:
    1  = positive  -> both GT IDs are known and equal
    0  = negative  -> both GT IDs are known and different
   -1  = ignore    -> at least one plane has unknown GT identity

IMPORTANT:
GT IDs are used ONLY to create evaluation labels.
They are not used to decide which candidate pairs are generated.
"""

from pathlib import Path
import csv
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\PlaneBasedSLAM")

INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tum_plane_features_with_gt_ids"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
)

OUTPUT_CSV = OUTPUT_DIR / "candidate_pairs.csv"


# ============================================================
# HELPERS
# ============================================================

def frame_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_frame(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "normals": np.asarray(data["normals"], dtype=np.float64),
            "offsets": np.asarray(data["offsets"], dtype=np.float64),
            "centroids": np.asarray(data["centroids"], dtype=np.float64),
            "contours": np.asarray(data["contours"], dtype=np.float64),
            "support_counts": np.asarray(data["support_counts"], dtype=np.int32),
            "global_plane_ids": np.asarray(
                data["global_plane_ids"], dtype=np.int32
            ),
            "gt_known": np.asarray(data["gt_known"], dtype=bool),
        }


def gt_label(id_a, known_a, id_b, known_b):
    if not known_a or not known_b:
        return -1

    return 1 if int(id_a) == int(id_b) else 0


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input folder not found:\n{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(
        INPUT_DIR.glob("frame_*.npz"),
        key=frame_id_from_path,
    )

    if len(frame_files) < 2:
        raise RuntimeError(
            "Need at least two frame files to generate candidate pairs."
        )

    rows = []
    pair_id = 0

    positive = 0
    negative = 0
    ignored = 0

    for file_a, file_b in zip(frame_files[:-1], frame_files[1:]):
        frame_a = frame_id_from_path(file_a)
        frame_b = frame_id_from_path(file_b)

        # Only use true consecutive frames.
        if frame_b != frame_a + 1:
            print(
                f"[skip] non-consecutive frames: "
                f"{frame_a} -> {frame_b}"
            )
            continue

        data_a = load_frame(file_a)
        data_b = load_frame(file_b)

        n_a = len(data_a["normals"])
        n_b = len(data_b["normals"])

        frame_pair_count = 0

        for plane_a in range(n_a):
            for plane_b in range(n_b):
                label = gt_label(
                    data_a["global_plane_ids"][plane_a],
                    data_a["gt_known"][plane_a],
                    data_b["global_plane_ids"][plane_b],
                    data_b["gt_known"][plane_b],
                )

                if label == 1:
                    positive += 1
                elif label == 0:
                    negative += 1
                else:
                    ignored += 1

                rows.append(
                    {
                        "pair_id": pair_id,
                        "frame_a": frame_a,
                        "plane_a": plane_a,
                        "frame_b": frame_b,
                        "plane_b": plane_b,
                        "gt_label": label,
                        "gt_plane_id_a": int(
                            data_a["global_plane_ids"][plane_a]
                        ),
                        "gt_plane_id_b": int(
                            data_b["global_plane_ids"][plane_b]
                        ),
                    }
                )

                pair_id += 1
                frame_pair_count += 1

        print(
            f"frame_{frame_a:04d} -> frame_{frame_b:04d}: "
            f"{n_a} x {n_b} = {frame_pair_count} candidates"
        )

    fieldnames = [
        "pair_id",
        "frame_a",
        "plane_a",
        "frame_b",
        "plane_b",
        "gt_label",
        "gt_plane_id_a",
        "gt_plane_id_b",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("CANDIDATE PAIR GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total pairs : {len(rows)}")
    print(f"Positive    : {positive}")
    print(f"Negative    : {negative}")
    print(f"Ignore      : {ignored}")
    print(f"Saved to    : {OUTPUT_CSV}")
    print()
    print(
        "gt_label meanings:\n"
        "  1  = same known ETH3D plane\n"
        "  0  = different known ETH3D planes\n"
        " -1  = at least one plane has unknown GT identity"
    )


if __name__ == "__main__":
    main()
