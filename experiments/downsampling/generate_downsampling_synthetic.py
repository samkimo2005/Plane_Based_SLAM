# generate_downsampling_synthetic.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent

OUTPUT_DIR = (
    EXPERIMENTS_DIR
    / "data"
    / "downsampling"
    / "synthetic"
)

SAMPLE_DIR = (
    OUTPUT_DIR
    / "samples"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "manifest.csv"
)


# ============================================================
# Experiment configuration
# ============================================================

MASTER_SEED = 20260929

PLANE_COUNTS = [
    1,
    3,
    5,
]

NOISE_SIGMAS_M = [
    0.000,
    0.005,
    0.010,
]

# Number of points along each side
# of each rectangular plane.
#
# 30 x 30 = 900 points / plane
# 50 x 50 = 2500 points / plane
GRID_SIZES = [
    30,
    50,
]

REPEATS = 3


# Plane patch dimensions.
MIN_WIDTH_M = 0.8
MAX_WIDTH_M = 1.5

MIN_HEIGHT_M = 0.8
MAX_HEIGHT_M = 1.5


# Scene placement.
XY_LIMIT_M = 1.5

MIN_DEPTH_M = 1.5
MAX_DEPTH_M = 3.5

# Keep plane centres sufficiently separated.
MIN_CENTRE_DISTANCE_M = 0.8


# ============================================================
# Basic geometry
# ============================================================

def normalise(
    vector: np.ndarray,
) -> np.ndarray:

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm <= 1e-12:
        raise ValueError(
            "Cannot normalise zero vector."
        )

    return (
        vector
        / norm
    )


def canonicalise_plane(
    normal: np.ndarray,
    offset: float,
) -> tuple[
    np.ndarray,
    float,
]:

    """
    Give every plane a deterministic sign.

    The first sufficiently non-zero normal
    component is forced positive.
    """

    normal = normalise(
        normal
    )

    offset = float(
        offset
    )

    for value in normal:

        if abs(value) <= 1e-12:
            continue

        if value < 0.0:

            normal = (
                -normal
            )

            offset = (
                -offset
            )

        break

    return (
        normal,
        offset,
    )


def random_unit_normal(
    rng: np.random.Generator,
) -> np.ndarray:

    """
    Generate a random 3-D unit normal.
    """

    while True:

        vector = rng.normal(
            size=3
        )

        norm = np.linalg.norm(
            vector
        )

        if norm > 1e-8:

            return (
                vector
                / norm
            )


def plane_basis(
    normal: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    """
    Construct two orthonormal vectors
    lying in the plane.
    """

    normal = normalise(
        normal
    )

    # Select an axis that is not
    # nearly parallel to the normal.
    axis_candidates = np.eye(
        3,
        dtype=np.float64,
    )

    dots = np.abs(
        axis_candidates
        @ normal
    )

    helper = axis_candidates[
        np.argmin(
            dots
        )
    ]

    u = np.cross(
        normal,
        helper,
    )

    u = normalise(
        u
    )

    v = np.cross(
        normal,
        u,
    )

    v = normalise(
        v
    )

    return (
        u,
        v,
    )


# ============================================================
# Scene geometry
# ============================================================

def generate_plane_centres(
    n_planes: int,
    rng: np.random.Generator,
) -> np.ndarray:

    """
    Generate spatially separated plane-patch
    centres in front of the camera.
    """

    centres = []

    max_attempts = 10000
    attempts = 0

    while len(centres) < n_planes:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Failed to generate sufficiently "
                "separated plane centres."
            )

        centre = np.array(
            [
                rng.uniform(
                    -XY_LIMIT_M,
                    XY_LIMIT_M,
                ),
                rng.uniform(
                    -XY_LIMIT_M,
                    XY_LIMIT_M,
                ),
                rng.uniform(
                    MIN_DEPTH_M,
                    MAX_DEPTH_M,
                ),
            ],
            dtype=np.float64,
        )

        if not centres:

            centres.append(
                centre
            )

            continue

        distances = [
            np.linalg.norm(
                centre
                - existing
            )
            for existing
            in centres
        ]

        if min(
            distances
        ) >= MIN_CENTRE_DISTANCE_M:

            centres.append(
                centre
            )

    return np.asarray(
        centres,
        dtype=np.float64,
    )


def generate_plane_patch(
    centre: np.ndarray,
    normal: np.ndarray,
    width: float,
    height: float,
    grid_size: int,
) -> np.ndarray:

    """
    Generate an ordered rectangular grid
    of points lying exactly on a plane.

    The row-major ordering is deliberately
    retained because deterministic every-k
    sampling depends on input ordering.
    """

    u_axis, v_axis = (
        plane_basis(
            normal
        )
    )

    u_values = np.linspace(
        -0.5 * width,
        0.5 * width,
        grid_size,
    )

    v_values = np.linspace(
        -0.5 * height,
        0.5 * height,
        grid_size,
    )

    points = []

    # Row-major ordering.
    for v_coord in v_values:

        for u_coord in u_values:

            point = (
                centre
                + u_coord * u_axis
                + v_coord * v_axis
            )

            points.append(
                point
            )

    return np.asarray(
        points,
        dtype=np.float64,
    )


# ============================================================
# Generate one synthetic cloud
# ============================================================

def generate_sample(
    n_planes: int,
    grid_size: int,
    noise_sigma_m: float,
    seed: int,
) -> dict:

    rng = np.random.default_rng(
        seed
    )

    centres = (
        generate_plane_centres(
            n_planes=n_planes,
            rng=rng,
        )
    )

    clean_clouds = []
    noisy_clouds = []

    plane_ids = []

    gt_normals = []
    gt_offsets = []

    widths = []
    heights = []

    for plane_id in range(
        n_planes
    ):

        centre = centres[
            plane_id
        ]

        normal = (
            random_unit_normal(
                rng
            )
        )

        width = rng.uniform(
            MIN_WIDTH_M,
            MAX_WIDTH_M,
        )

        height = rng.uniform(
            MIN_HEIGHT_M,
            MAX_HEIGHT_M,
        )

        # Plane equation:
        #
        # n^T p + d = 0
        #
        # and centre lies on the plane.
        offset = -float(
            normal
            @ centre
        )

        normal, offset = (
            canonicalise_plane(
                normal,
                offset,
            )
        )

        clean_points = (
            generate_plane_patch(
                centre=centre,
                normal=normal,
                width=width,
                height=height,
                grid_size=grid_size,
            )
        )

        # Independent isotropic Gaussian
        # measurement noise.
        if noise_sigma_m > 0.0:

            noise = rng.normal(
                loc=0.0,
                scale=noise_sigma_m,
                size=clean_points.shape,
            )

            noisy_points = (
                clean_points
                + noise
            )

        else:

            noisy_points = (
                clean_points.copy()
            )

        clean_clouds.append(
            clean_points
        )

        noisy_clouds.append(
            noisy_points
        )

        plane_ids.append(
            np.full(
                len(clean_points),
                plane_id,
                dtype=np.int32,
            )
        )

        gt_normals.append(
            normal
        )

        gt_offsets.append(
            offset
        )

        widths.append(
            width
        )

        heights.append(
            height
        )

    clean_points = np.concatenate(
        clean_clouds,
        axis=0,
    )

    points = np.concatenate(
        noisy_clouds,
        axis=0,
    )

    plane_ids = np.concatenate(
        plane_ids,
        axis=0,
    )

    gt_normals = np.asarray(
        gt_normals,
        dtype=np.float64,
    )

    gt_offsets = np.asarray(
        gt_offsets,
        dtype=np.float64,
    )

    widths = np.asarray(
        widths,
        dtype=np.float64,
    )

    heights = np.asarray(
        heights,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if len(points) != (
        n_planes
        * grid_size
        * grid_size
    ):
        raise RuntimeError(
            "Unexpected point count."
        )

    if len(
        np.unique(
            plane_ids
        )
    ) != n_planes:
        raise RuntimeError(
            "Plane-ID generation failed."
        )

    # Clean points must lie essentially
    # exactly on their analytical GT planes.
    for plane_id in range(
        n_planes
    ):

        mask = (
            plane_ids
            == plane_id
        )

        residuals = np.abs(
            clean_points[
                mask
            ]
            @ gt_normals[
                plane_id
            ]
            + gt_offsets[
                plane_id
            ]
        )

        if np.max(
            residuals
        ) > 1e-9:
            raise RuntimeError(
                "Synthetic clean plane does not "
                "satisfy analytical GT."
            )

    return {
        "points":
            points,

        "clean_points":
            clean_points,

        "plane_ids":
            plane_ids,

        "gt_normals":
            gt_normals,

        "gt_offsets":
            gt_offsets,

        "plane_centres":
            centres,

        "plane_widths":
            widths,

        "plane_heights":
            heights,

        "n_planes":
            int(
                n_planes
            ),

        "grid_size":
            int(
                grid_size
            ),

        "noise_sigma_m":
            float(
                noise_sigma_m
            ),

        "seed":
            int(
                seed
            ),
    }


# ============================================================
# Save one sample
# ============================================================

def save_sample(
    path: Path,
    sample: dict,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,

        points=sample[
            "points"
        ],

        clean_points=sample[
            "clean_points"
        ],

        plane_ids=sample[
            "plane_ids"
        ],

        gt_normals=sample[
            "gt_normals"
        ],

        gt_offsets=sample[
            "gt_offsets"
        ],

        plane_centres=sample[
            "plane_centres"
        ],

        plane_widths=sample[
            "plane_widths"
        ],

        plane_heights=sample[
            "plane_heights"
        ],

        n_planes=np.asarray(
            sample[
                "n_planes"
            ],
            dtype=np.int32,
        ),

        grid_size=np.asarray(
            sample[
                "grid_size"
            ],
            dtype=np.int32,
        ),

        noise_sigma_m=np.asarray(
            sample[
                "noise_sigma_m"
            ],
            dtype=np.float64,
        ),

        seed=np.asarray(
            sample[
                "seed"
            ],
            dtype=np.int64,
        ),
    )


# ============================================================
# Main dataset generation
# ============================================================

def main():

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    sample_index = 0

    seed_sequence = (
        np.random.SeedSequence(
            MASTER_SEED
        )
    )

    n_samples = (
        len(
            PLANE_COUNTS
        )
        * len(
            GRID_SIZES
        )
        * len(
            NOISE_SIGMAS_M
        )
        * REPEATS
    )

    child_sequences = (
        seed_sequence.spawn(
            n_samples
        )
    )

    print(
        "=" * 72
    )

    print(
        "GENERATE DOWNSAMPLING SYNTHETIC DATA"
    )

    print(
        "=" * 72
    )

    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Total samples    : "
        f"{n_samples}"
    )

    print()

    for n_planes in (
        PLANE_COUNTS
    ):

        for grid_size in (
            GRID_SIZES
        ):

            for noise_sigma_m in (
                NOISE_SIGMAS_M
            ):

                for repeat in range(
                    REPEATS
                ):

                    child = (
                        child_sequences[
                            sample_index
                        ]
                    )

                    seed = int(
                        child.generate_state(
                            1,
                            dtype=np.uint32,
                        )[0]
                    )

                    sample = (
                        generate_sample(
                            n_planes=n_planes,
                            grid_size=grid_size,
                            noise_sigma_m=noise_sigma_m,
                            seed=seed,
                        )
                    )

                    filename = (
                        f"sample_"
                        f"{sample_index:04d}.npz"
                    )

                    path = (
                        SAMPLE_DIR
                        / filename
                    )

                    save_sample(
                        path,
                        sample,
                    )

                    input_count = (
                        len(
                            sample[
                                "points"
                            ]
                        )
                    )

                    rows.append(
                        {
                            "sample":
                                filename,

                            "sample_index":
                                sample_index,

                            "n_planes":
                                n_planes,

                            "grid_size":
                                grid_size,

                            "points_per_plane":
                                (
                                    grid_size
                                    * grid_size
                                ),

                            "input_count":
                                input_count,

                            "noise_sigma_m":
                                noise_sigma_m,

                            "repeat":
                                repeat,

                            "seed":
                                seed,
                        }
                    )

                    print(
                        f"[{sample_index + 1:02d}/"
                        f"{n_samples:02d}] "
                        f"{filename} | "
                        f"planes={n_planes} | "
                        f"grid={grid_size}x{grid_size} | "
                        f"sigma={noise_sigma_m:.3f} m | "
                        f"N={input_count}"
                    )

                    sample_index += 1

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    fieldnames = [
        "sample",
        "sample_index",
        "n_planes",
        "grid_size",
        "points_per_plane",
        "input_count",
        "noise_sigma_m",
        "repeat",
        "seed",
    ]

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()

    print(
        "=" * 72
    )

    print(
        "GENERATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Samples generated : "
        f"{sample_index}"
    )

    print(
        f"Manifest          : "
        f"{MANIFEST_FILE}"
    )

    print(
        f"Sample directory  : "
        f"{SAMPLE_DIR}"
    )


if __name__ == "__main__":
    main()