"""
generate_similarity_real.py

Generate real TUM/ETH3D-derived samples for the
Pairwise Plane Similarity experiment.

Input to Similarity:
    selected Contour-point Candidate Gating output.

GT labels:
     1 = true correspondence
     0 = known non-correspondence
    -1 = unknown / ignored

Unknown pairs are retained so that all real candidate pairs
can be processed for runtime, but they must later be excluded
from classification metrics.
"""

from pathlib import Path
import csv

import numpy as np

from similarity_common import (
    PlaneObservation,
    SimilaritySample,
    save_samples_jsonl,
    validate_samples,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"D:\PlaneBasedSLAM")

GATING_RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tum"
    / "candidate_gating_tum_results.csv"
)

PLANE_FEATURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "candidate_gating"
    / "tum_plane_features_with_gt_ids"
)

# Prepared point clouds used by the detector.
PREPARED_FRAME_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / "rgbd_dataset_freiburg1_xyz"
    / "frames"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "similarity"
    / "real_samples.jsonl"
)

PATCH_POINTS = 150


# ============================================================
# Load Candidate Gating output
# ============================================================

def load_retained_pairs():

    if not GATING_RESULT_FILE.exists():
        raise FileNotFoundError(
            GATING_RESULT_FILE
        )

    retained = []

    with GATING_RESULT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        for row in csv.DictReader(f):

            if row["method"] != "contour_point":
                continue

            if int(row["keep"]) != 1:
                continue

            retained.append(
                {
                    "pair_id":
                        int(row["pair_id"]),

                    "frame_a":
                        int(row["frame_a"]),

                    "plane_a":
                        int(row["plane_a"]),

                    "frame_b":
                        int(row["frame_b"]),

                    "plane_b":
                        int(row["plane_b"]),

                    "gt_label":
                        int(row["gt_label"]),

                    "gt_plane_id_a":
                        int(row["gt_plane_id_a"]),

                    "gt_plane_id_b":
                        int(row["gt_plane_id_b"]),
                }
            )

    if not retained:
        raise RuntimeError(
            "No Contour-point retained pairs found."
        )

    return retained


# ============================================================
# Load plane feature file
# ============================================================

def load_plane_features(frame_id):

    path = (
        PLANE_FEATURE_DIR
        / f"frame_{frame_id:04d}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        return {
            "normals":
                np.asarray(
                    data["normals"],
                    dtype=float,
                ),

            "offsets":
                np.asarray(
                    data["offsets"],
                    dtype=float,
                ),

            "centroids":
                np.asarray(
                    data["centroids"],
                    dtype=float,
                ),

            "contours":
                np.asarray(
                    data["contours"],
                    dtype=float,
                ),

            "support_counts":
                np.asarray(
                    data["support_counts"],
                    dtype=int,
                ),

            "support_masks":
                np.asarray(
                    data["support_masks"],
                    dtype=bool,
                ),

            "global_plane_ids":
                np.asarray(
                    data["global_plane_ids"],
                    dtype=int,
                ),

            "gt_known":
                np.asarray(
                    data["gt_known"],
                    dtype=bool,
                ),
        }


# ============================================================
# Load detector point cloud
# ============================================================

def load_detector_points(frame_id):

    path = (
        PREPARED_FRAME_DIR
        / f"frame_{frame_id:04d}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"\nPrepared TUM frame not found:\n"
            f"{path}\n"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if "detector_points" not in data.files:
            raise KeyError(
                f"\n'detector_points' not found in:\n"
                f"{path}\n\n"
                f"Available keys:\n{data.files}"
            )

        return np.asarray(
            data["detector_points"],
            dtype=float,
        )


# ============================================================
# Patch utilities
# ============================================================

def deterministic_subsample(
    points,
    maximum_points=PATCH_POINTS,
):

    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) <= maximum_points:
        return points

    indices = np.linspace(
        0,
        len(points) - 1,
        maximum_points,
        dtype=int,
    )

    return points[indices]


def plane_area_from_points(
    points,
    normal,
):
    """
    Estimate bounded plane area from the 2-D extent of
    the supporting patch after projection into the plane.

    This does not require contour vertex ordering.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    normal = np.asarray(
        normal,
        dtype=float,
    )

    normal = normal / np.linalg.norm(normal)

    if len(points) < 3:
        return 0.0

    helper = np.array(
        [1.0, 0.0, 0.0]
    )

    if abs(
        float(helper @ normal)
    ) > 0.9:

        helper = np.array(
            [0.0, 1.0, 0.0]
        )

    u = np.cross(
        normal,
        helper,
    )

    u = u / np.linalg.norm(u)

    v = np.cross(
        normal,
        u,
    )

    projected_u = points @ u
    projected_v = points @ v

    width = (
        np.max(projected_u)
        - np.min(projected_u)
    )

    height = (
        np.max(projected_v)
        - np.min(projected_v)
    )

    return float(
        max(width * height, 1e-8)
    )


# ============================================================
# Convert one detected plane
# ============================================================

def make_plane_observation(
    frame_id,
    plane_id,
    features,
    detector_points,
):

    normal = np.asarray(
        features["normals"][plane_id],
        dtype=float,
    )

    normal = (
        normal
        / np.linalg.norm(normal)
    )

    offset = float(
        features["offsets"][plane_id]
    )

    centroid = np.asarray(
        features["centroids"][plane_id],
        dtype=float,
    )

    contour = np.asarray(
        features["contours"][plane_id],
        dtype=float,
    )

    support_mask = np.asarray(
        features["support_masks"][plane_id],
        dtype=bool,
    )

    if len(support_mask) != len(
        detector_points
    ):
        raise RuntimeError(
            f"frame {frame_id}, plane {plane_id}: "
            f"support mask has {len(support_mask)} values "
            f"but detector cloud has "
            f"{len(detector_points)} points."
        )

    patch = detector_points[
        support_mask
    ]

    if len(patch) == 0:
        raise RuntimeError(
            f"frame {frame_id}, plane {plane_id}: "
            f"empty supporting patch."
        )

    area = plane_area_from_points(
        patch,
        normal,
    )

    patch = deterministic_subsample(
        patch
    )

    global_plane_id = int(
        features[
            "global_plane_ids"
        ][plane_id]
    )

    gt_known = bool(
        features["gt_known"][plane_id]
    )

    return PlaneObservation(
        plane_id=(
            f"f{frame_id:04d}_"
            f"p{plane_id}"
        ),

        normal=normal,
        offset=offset,
        centroid=centroid,

        area=area,

        support_size=int(
            features[
                "support_counts"
            ][plane_id]
        ),

        metadata={
            "frame_id":
                int(frame_id),

            "local_plane_id":
                int(plane_id),

            "global_plane_id":
                global_plane_id,

            "gt_known":
                gt_known,

            "contour":
                contour,

            "patch":
                patch,
        },
    )


# ============================================================
# Main
# ============================================================

def main():

    retained_pairs = (
        load_retained_pairs()
    )

    # Cache frame data so that each NPZ file is loaded once.
    feature_cache = {}
    point_cache = {}

    def get_frame(frame_id):

        if frame_id not in feature_cache:

            feature_cache[frame_id] = (
                load_plane_features(
                    frame_id
                )
            )

            point_cache[frame_id] = (
                load_detector_points(
                    frame_id
                )
            )

        return (
            feature_cache[frame_id],
            point_cache[frame_id],
        )

    samples = []

    positive = 0
    negative = 0
    ignored = 0

    for pair in retained_pairs:

        frame_a = pair["frame_a"]
        frame_b = pair["frame_b"]

        features_a, points_a = (
            get_frame(frame_a)
        )

        features_b, points_b = (
            get_frame(frame_b)
        )

        plane_a = (
            make_plane_observation(
                frame_id=frame_a,
                plane_id=pair["plane_a"],
                features=features_a,
                detector_points=points_a,
            )
        )

        plane_b = (
            make_plane_observation(
                frame_id=frame_b,
                plane_id=pair["plane_b"],
                features=features_b,
                detector_points=points_b,
            )
        )

        gt_label = pair["gt_label"]

        if gt_label == 1:
            positive += 1

        elif gt_label == 0:
            negative += 1

        else:
            ignored += 1

        # Candidate Gating real-data experiment used an
        # identity / zero-motion prior. Keep the same
        # stage-level convention here to avoid GT-pose leakage.
        samples.append(
            SimilaritySample(
                sample_id=(
                    f"tum_pair_"
                    f"{pair['pair_id']:05d}"
                ),

                current_plane=plane_a,
                reference_plane=plane_b,

                # SimilaritySample currently requires a bool.
                # For unknown GT pairs this value is only a
                # placeholder. Evaluation must instead use
                # metadata['gt_label'].
                gt_match=(
                    gt_label == 1
                ),

                data_source="real",
                split="final",
                condition="tum_fr1_xyz",

                sequence_id=(
                    "rgbd_dataset_freiburg1_xyz"
                ),

                current_frame_id=frame_a,
                reference_frame_id=frame_b,

                R_ref_cur=np.eye(
                    3,
                    dtype=float,
                ),

                t_ref_cur=np.zeros(
                    3,
                    dtype=float,
                ),

                metadata={
                    "pair_id":
                        pair["pair_id"],

                    "gt_label":
                        gt_label,

                    "gt_plane_id_a":
                        pair[
                            "gt_plane_id_a"
                        ],

                    "gt_plane_id_b":
                        pair[
                            "gt_plane_id_b"
                        ],

                    "candidate_gating_method":
                        "contour_point",

                    "candidate_gating_keep":
                        True,

                    "pose_prior":
                        "identity_zero_motion",
                },
            )
        )

    # Basic structural validation.
    validate_samples(
        samples
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_samples_jsonl(
        OUTPUT_FILE,
        samples,
    )

    print()
    print(
        "=== Real Similarity Dataset ==="
    )

    print(
        f"Contour-point retained : "
        f"{len(samples)}"
    )

    print(
        f"Known positive         : "
        f"{positive}"
    )

    print(
        f"Known negative         : "
        f"{negative}"
    )

    print(
        f"Unknown / ignored      : "
        f"{ignored}"
    )

    print(
        f"Evaluable              : "
        f"{positive + negative}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()