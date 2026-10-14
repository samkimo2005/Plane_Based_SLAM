from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_multiplane_benchmark"
)

FINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_multiplane_benchmark_final"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_FILE = (
    FINAL_DIR
    / "manifest.csv"
)


# ============================================================
# 2. FINAL benchmark design
#
# Same condition grid as the original multi-plane benchmark:
#
#   plane counts   = 2, 3, 4
#   noise sigma    = 0, 0.020 m
#   outlier ratio  = 0, 0.20
#   repetitions    = 5
#
# Total:
#
#   3 x 2 x 2 x 5 = 60 unseen FINAL cases
#
# The original benchmark is used ONLY to recover the fixed
# plane geometry and approximate finite patch extents.
# New points are sampled continuously using completely new
# random seeds.
# ============================================================

PLANE_COUNTS = [
    2,
    3,
    4,
]

NOISE_SIGMAS_M = [
    0.0,
    0.020,
]

OUTLIER_RATIOS = [
    0.0,
    0.20,
]

NUM_REPEATS = 5

TOTAL_INLIER_POINTS = 6000

FINAL_SEED_BASE = 9_180_000

EXPECTED_FINAL_CASES = (
    len(PLANE_COUNTS)
    * len(NOISE_SIGMAS_M)
    * len(OUTLIER_RATIOS)
    * NUM_REPEATS
)


# ============================================================
# 3. Geometry helpers
# ============================================================

def normalize(
    vector,
):
    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm < 1e-12:
        raise ValueError(
            "Cannot normalize a near-zero vector."
        )

    return vector / norm


def canonicalise_plane(
    normal,
    d,
):
    normal = normalize(
        normal
    )

    index = int(
        np.argmax(
            np.abs(
                normal
            )
        )
    )

    if normal[
        index
    ] < 0:
        normal = -normal
        d = -d

    return normal, float(
        d
    )


def load_reference_sample(
    npz_file,
):
    data = np.load(
        npz_file
    )

    points = np.asarray(
        data[
            "points"
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        data[
            "labels"
        ],
        dtype=np.int32,
    )

    gt_file = npz_file.with_name(
        npz_file.stem
        + "_gt.json"
    )

    with open(
        gt_file,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(
            f
        )

    return (
        points,
        labels,
        metadata,
    )


def find_clean_reference(
    plane_count,
):
    """
    Find one old sample with:
        noise = 0
        outlier ratio = 0

    for the requested plane count.

    This is used only to recover the original fixed scene
    geometry and finite plane-patch extents.
    """

    sample_files = sorted(
        REFERENCE_DIR.glob(
            "P*_N*_O*_R*.npz"
        )
    )

    for npz_file in sample_files:
        (
            points,
            labels,
            metadata,
        ) = load_reference_sample(
            npz_file
        )

        if (
            int(
                metadata[
                    "plane_count"
                ]
            )
            != plane_count
        ):
            continue

        if not np.isclose(
            float(
                metadata[
                    "noise_sigma_m"
                ]
            ),
            0.0,
        ):
            continue

        if not np.isclose(
            float(
                metadata[
                    "outlier_ratio"
                ]
            ),
            0.0,
        ):
            continue

        return (
            npz_file,
            points,
            labels,
            metadata,
        )

    raise FileNotFoundError(
        "Could not find a clean reference sample for "
        f"{plane_count} planes in:\n{REFERENCE_DIR}"
    )


def build_plane_patch_template(
    points,
    labels,
    plane_metadata,
):
    """
    Recover a continuous finite patch model from the old
    clean sample.

    The GT plane equation is preserved exactly. PCA is used
    only to estimate the two in-plane patch directions and
    their approximate finite ranges.
    """

    plane_id = int(
        plane_metadata[
            "id"
        ]
    )

    plane_points = points[
        labels
        == plane_id
    ]

    if len(
        plane_points
    ) < 3:
        raise RuntimeError(
            f"Reference plane {plane_id} has too few points."
        )

    normal, d = canonicalise_plane(
        plane_metadata[
            "normal"
        ],
        plane_metadata[
            "d"
        ],
    )

    centroid = np.mean(
        plane_points,
        axis=0,
    )

    # Project centroid exactly onto the GT infinite plane.
    signed_distance = (
        np.dot(
            normal,
            centroid,
        )
        + d
    )

    plane_origin = (
        centroid
        - signed_distance
        * normal
    )

    centred = (
        plane_points
        - centroid
    )

    covariance = (
        centred.T
        @ centred
        / max(
            len(
                centred
            ),
            1,
        )
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[
        ::-1
    ]

    tangent_u = normalize(
        eigenvectors[
            :,
            order[
                0
            ],
        ]
    )

    # Make the second tangent direction exactly orthogonal
    # to both the GT normal and tangent_u.
    tangent_v = normalize(
        np.cross(
            normal,
            tangent_u,
        )
    )

    tangent_u = normalize(
        np.cross(
            tangent_v,
            normal,
        )
    )

    local_u = (
        plane_points
        - plane_origin
    ) @ tangent_u

    local_v = (
        plane_points
        - plane_origin
    ) @ tangent_v

    return {
        "id":
            plane_id,

        "name":
            plane_metadata[
                "name"
            ],

        "normal":
            normal,

        "d":
            float(
                d
            ),

        "origin":
            plane_origin,

        "u":
            tangent_u,

        "v":
            tangent_v,

        "u_min":
            float(
                np.min(
                    local_u
                )
            ),

        "u_max":
            float(
                np.max(
                    local_u
                )
            ),

        "v_min":
            float(
                np.min(
                    local_v
                )
            ),

        "v_max":
            float(
                np.max(
                    local_v
                )
            ),
    }


def build_scene_template(
    plane_count,
):
    (
        reference_file,
        points,
        labels,
        metadata,
    ) = find_clean_reference(
        plane_count
    )

    patches = [
        build_plane_patch_template(
            points,
            labels,
            plane,
        )
        for plane in metadata[
            "gt_planes"
        ]
    ]

    # Bounding box for random scene outliers.
    minimum = np.min(
        points,
        axis=0,
    )

    maximum = np.max(
        points,
        axis=0,
    )

    extent = np.maximum(
        maximum
        - minimum,
        0.5,
    )

    margin = 0.10 * extent

    return {
        "reference_file":
            reference_file.name,

        "patches":
            patches,

        "bbox_min":
            minimum
            - margin,

        "bbox_max":
            maximum
            + margin,
    }


# ============================================================
# 4. Point generation
# ============================================================

def allocate_inlier_counts(
    total_points,
    num_planes,
):
    base = (
        total_points
        // num_planes
    )

    remainder = (
        total_points
        % num_planes
    )

    counts = [
        base
        + (
            1
            if index < remainder
            else 0
        )
        for index in range(
            num_planes
        )
    ]

    return counts


def sample_plane_points(
    rng,
    patch,
    num_points,
    noise_sigma_m,
):
    u_values = rng.uniform(
        patch[
            "u_min"
        ],
        patch[
            "u_max"
        ],
        size=num_points,
    )

    v_values = rng.uniform(
        patch[
            "v_min"
        ],
        patch[
            "v_max"
        ],
        size=num_points,
    )

    points = (
        patch[
            "origin"
        ][
            None,
            :
        ]
        + u_values[
            :,
            None
        ]
        * patch[
            "u"
        ][
            None,
            :
        ]
        + v_values[
            :,
            None
        ]
        * patch[
            "v"
        ][
            None,
            :
        ]
    )

    if noise_sigma_m > 0.0:
        normal_noise = rng.normal(
            loc=0.0,
            scale=noise_sigma_m,
            size=num_points,
        )

        points = (
            points
            + normal_noise[
                :,
                None
            ]
            * patch[
                "normal"
            ][
                None,
                :
            ]
        )

    return points


def calculate_outlier_count(
    num_inliers,
    outlier_ratio,
):
    """
    Interpret outlier_ratio as the fraction of the final
    point cloud occupied by outliers.
    """

    if outlier_ratio <= 0.0:
        return 0

    if outlier_ratio >= 1.0:
        raise ValueError(
            "outlier_ratio must be below 1."
        )

    return int(
        round(
            num_inliers
            * outlier_ratio
            / (
                1.0
                - outlier_ratio
            )
        )
    )


def generate_case(
    scene_template,
    plane_count,
    noise_sigma_m,
    outlier_ratio,
    repeat_id,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    inlier_counts = allocate_inlier_counts(
        TOTAL_INLIER_POINTS,
        plane_count,
    )

    point_blocks = []
    label_blocks = []

    gt_planes = []

    for patch, num_points in zip(
        scene_template[
            "patches"
        ],
        inlier_counts,
    ):
        plane_points = sample_plane_points(
            rng,
            patch,
            num_points,
            noise_sigma_m,
        )

        point_blocks.append(
            plane_points
        )

        label_blocks.append(
            np.full(
                num_points,
                int(
                    patch[
                        "id"
                    ]
                ),
                dtype=np.int32,
            )
        )

        gt_planes.append(
            {
                "id":
                    int(
                        patch[
                            "id"
                        ]
                    ),

                "name":
                    patch[
                        "name"
                    ],

                "normal":
                    [
                        float(
                            value
                        )
                        for value in patch[
                            "normal"
                        ]
                    ],

                "d":
                    float(
                        patch[
                            "d"
                        ]
                    ),
            }
        )

    num_outliers = calculate_outlier_count(
        TOTAL_INLIER_POINTS,
        outlier_ratio,
    )

    if num_outliers > 0:
        outliers = rng.uniform(
            scene_template[
                "bbox_min"
            ],
            scene_template[
                "bbox_max"
            ],
            size=(
                num_outliers,
                3,
            ),
        )

        point_blocks.append(
            outliers
        )

        # 0 is reserved for clutter / outliers.
        label_blocks.append(
            np.zeros(
                num_outliers,
                dtype=np.int32,
            )
        )

    points = np.vstack(
        point_blocks
    ).astype(
        np.float64,
        copy=False,
    )

    labels = np.concatenate(
        label_blocks
    )

    permutation = rng.permutation(
        len(
            points
        )
    )

    points = points[
        permutation
    ]

    labels = labels[
        permutation
    ]

    sample_id = (
        f"P{plane_count}_"
        f"N{int(round(noise_sigma_m * 1000)):03d}_"
        f"O{int(round(outlier_ratio * 100)):02d}_"
        f"R{repeat_id:02d}"
    )

    metadata = {
        "sample_id":
            sample_id,

        "dataset_split":
            "FINAL",

        "plane_count":
            int(
                plane_count
            ),

        "noise_sigma_m":
            float(
                noise_sigma_m
            ),

        "outlier_ratio":
            float(
                outlier_ratio
            ),

        "repeat_id":
            int(
                repeat_id
            ),

        "seed":
            int(
                seed
            ),

        "num_inlier_points":
            int(
                TOTAL_INLIER_POINTS
            ),

        "num_outlier_points":
            int(
                num_outliers
            ),

        "num_total_points":
            int(
                len(
                    points
                )
            ),

        "reference_geometry_source":
            scene_template[
                "reference_file"
            ],

        "gt_planes":
            gt_planes,
    }

    return (
        sample_id,
        points,
        labels,
        metadata,
    )


# ============================================================
# 5. Save one case
# ============================================================

def save_case(
    sample_id,
    points,
    labels,
    metadata,
):
    npz_file = (
        FINAL_DIR
        / f"{sample_id}.npz"
    )

    gt_file = (
        FINAL_DIR
        / f"{sample_id}_gt.json"
    )

    np.savez_compressed(
        npz_file,
        points=points,
        labels=labels,
    )

    with open(
        gt_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 6. Main
# ============================================================

def main():
    print(
        "=" * 78
    )

    print(
        "Generate Plane-Detection FINAL Multi-Plane Benchmark"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Reference geometry:"
    )

    print(
        REFERENCE_DIR
    )

    print()
    print(
        "FINAL output:"
    )

    print(
        FINAL_DIR
    )

    # --------------------------------------------------------
    # Build one fixed geometry template for each plane count.
    # --------------------------------------------------------

    scene_templates = {
        plane_count:
            build_scene_template(
                plane_count
            )
        for plane_count in PLANE_COUNTS
    }

    # --------------------------------------------------------
    # Remove only old FINAL benchmark files generated by this
    # script. The original DEV/preliminary benchmark is never
    # touched.
    # --------------------------------------------------------

    for pattern in [
        "P*_N*_O*_R*.npz",
        "P*_N*_O*_R*_gt.json",
    ]:
        for path in FINAL_DIR.glob(
            pattern
        ):
            path.unlink()

    rows = []

    case_index = 0

    for plane_count in PLANE_COUNTS:
        for noise_sigma_m in NOISE_SIGMAS_M:
            for outlier_ratio in OUTLIER_RATIOS:
                for repeat_id in range(
                    1,
                    NUM_REPEATS
                    + 1,
                ):
                    case_index += 1

                    seed = (
                        FINAL_SEED_BASE
                        + case_index
                        * 1009
                    )

                    (
                        sample_id,
                        points,
                        labels,
                        metadata,
                    ) = generate_case(
                        scene_templates[
                            plane_count
                        ],
                        plane_count,
                        noise_sigma_m,
                        outlier_ratio,
                        repeat_id,
                        seed,
                    )

                    save_case(
                        sample_id,
                        points,
                        labels,
                        metadata,
                    )

                    rows.append(
                        {
                            "sample_id":
                                sample_id,

                            "plane_count":
                                plane_count,

                            "noise_sigma_m":
                                noise_sigma_m,

                            "outlier_ratio":
                                outlier_ratio,

                            "repeat_id":
                                repeat_id,

                            "seed":
                                seed,

                            "num_inlier_points":
                                metadata[
                                    "num_inlier_points"
                                ],

                            "num_outlier_points":
                                metadata[
                                    "num_outlier_points"
                                ],

                            "num_total_points":
                                metadata[
                                    "num_total_points"
                                ],

                            "reference_geometry_source":
                                metadata[
                                    "reference_geometry_source"
                                ],
                        }
                    )

                    print(
                        f"[{case_index:02d}/{EXPECTED_FINAL_CASES}] "
                        f"{sample_id} "
                        f"| points={len(points)} "
                        f"| seed={seed}"
                    )

    if len(
        rows
    ) != EXPECTED_FINAL_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_CASES} generated cases, "
            f"got {len(rows)}."
        )

    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with open(
        MANIFEST_FILE,
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
            rows
        )

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL DATASET GENERATION COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Cases: {len(rows)}"
    )

    print(
        f"Manifest: {MANIFEST_FILE}"
    )

    print()
    print(
        "These cases use new seeds and must not be used "
        "for further parameter tuning."
    )


if __name__ == "__main__":
    main()
