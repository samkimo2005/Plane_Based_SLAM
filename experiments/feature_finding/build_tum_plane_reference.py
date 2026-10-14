"""
build_tum_plane_reference.py

Interactive semi-manual reference-plane annotation for the
TUM RGB-D Plane Detection benchmark.

Workflow for each selected TUM frame:

    RGB image
        ↓
    User draws polygon around each real planar surface
        ↓
    Corresponding depth pixels -> 3-D points
        ↓
    Robust SVD plane fitting
        ↓
    Reference plane equation n^T x + d = 0
        ↓
    Reference labels mapped onto detector_points
        ↓
    Saved reference .npz

The reference planes are therefore independent of the five
plane-detection algorithms being evaluated.

DIRECT RUN:
    Run directly from the IDE.

Controls:
    - Click polygon vertices around one plane.
    - Press ENTER to finish that polygon.
    - Close the figure if you want to abort the frame.

The terminal will ask how many planes you want to annotate.
"""

from __future__ import annotations

import json
import math

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


# ============================================================
# USER SETTINGS
# ============================================================

SEQUENCE_NAME = "rgbd_dataset_freiburg1_xyz"


# ------------------------------------------------------------
# Which prepared TUM frames should be annotated?
# ------------------------------------------------------------

# None:
#     process all available prepared frames
#
# Or explicitly specify:
#
# REFERENCE_FRAME_IDS = [
#     0, 5, 10, 15, 20
# ]
#
# For the first test, I strongly recommend only 2-3 frames.
REFERENCE_FRAME_IDS = [
    3,
    4,
    5,
]


# ------------------------------------------------------------
# Plane fitting
# ------------------------------------------------------------

# Minimum number of valid raw depth points inside one manually
# drawn polygon.
MIN_RAW_POINTS_FOR_PLANE = 200

# Robust fitting:
#
# fit -> reject points farther than threshold -> refit
ROBUST_FIT_DISTANCE_M = 0.020

ROBUST_FIT_ITERATIONS = 3


# ------------------------------------------------------------
# Mapping reference planes to detector cloud
# ------------------------------------------------------------

# Detector point must satisfy BOTH:
#
# 1. its image projection lies inside the manually drawn polygon
# 2. its point-to-plane distance is below this threshold
REFERENCE_SUPPORT_DISTANCE_M = 0.025


# ------------------------------------------------------------
# Small support cleanup
# ------------------------------------------------------------

MIN_REFERENCE_DETECTOR_POINTS = 80


# ------------------------------------------------------------
# Output behaviour
# ------------------------------------------------------------

OVERWRITE_EXISTING_REFERENCE = False


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TUM_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "tum"
    / "plane_detection"
    / SEQUENCE_NAME
)

FRAMES_DIR = (
    TUM_DATA_DIR
    / "frames"
)

REFERENCES_DIR = (
    TUM_DATA_DIR
    / "references"
)

REFERENCE_MANIFEST_FILE = (
    REFERENCES_DIR
    / "reference_manifest.json"
)


# ============================================================
# Basic utilities
# ============================================================

EPS = 1e-12


def normalize(
    vector: np.ndarray,
) -> np.ndarray:

    vector = np.asarray(
        vector,
        dtype=float,
    )

    length = float(
        np.linalg.norm(
            vector
        )
    )

    if length < EPS:

        raise ValueError(
            "Cannot normalize zero-length vector."
        )

    return vector / length


def canonicalise_plane(
    normal: np.ndarray,
    offset: float,
) -> Tuple[np.ndarray, float]:
    """
    Resolve plane sign ambiguity deterministically.
    """

    normal = normalize(
        normal
    )

    offset = float(
        offset
    )

    # Prefer normal facing roughly toward the camera.
    #
    # TUM camera convention:
    #     +Z points forward.
    #
    # Indoor visible surfaces typically face the camera,
    # therefore their normal often has negative Z.
    if normal[2] > 0:

        normal = -normal
        offset = -offset

    return (
        normal,
        offset,
    )


def frame_id_from_path(
    path: Path,
) -> int:

    return int(
        path.stem.split(
            "_"
        )[-1]
    )


# ============================================================
# Plane fitting
# ============================================================

def fit_plane_svd(
    points: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    np.ndarray,
]:
    """
    Least-squares plane fit.

    Plane equation:
        n^T x + d = 0
    """

    if len(
        points
    ) < 3:

        raise ValueError(
            "At least three points are required."
        )

    centroid = np.mean(
        points,
        axis=0,
    )

    centred = (
        points
        - centroid
    )

    _, _, vh = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    normal = vh[
        -1
    ]

    normal = normalize(
        normal
    )

    offset = -float(
        normal
        @ centroid
    )

    normal, offset = (
        canonicalise_plane(
            normal,
            offset,
        )
    )

    return (
        normal,
        offset,
        centroid,
    )


def robust_fit_plane(
    points: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    Iteratively fit plane and remove points whose distance
    exceeds ROBUST_FIT_DISTANCE_M.
    """

    working_points = np.asarray(
        points,
        dtype=float,
    )

    original_indices = np.arange(
        len(
            working_points
        )
    )

    for _ in range(
        ROBUST_FIT_ITERATIONS
    ):

        if len(
            working_points
        ) < 3:

            break

        (
            normal,
            offset,
            centroid,
        ) = fit_plane_svd(
            working_points
        )

        distances = np.abs(
            working_points
            @ normal
            + offset
        )

        keep = (
            distances
            <= ROBUST_FIT_DISTANCE_M
        )

        if np.count_nonzero(
            keep
        ) < 3:

            break

        if np.all(
            keep
        ):

            break

        working_points = (
            working_points[
                keep
            ]
        )

        original_indices = (
            original_indices[
                keep
            ]
        )

    (
        normal,
        offset,
        centroid,
    ) = fit_plane_svd(
        working_points
    )

    return (
        normal,
        offset,
        centroid,
        original_indices,
    )


# ============================================================
# Polygon annotation
# ============================================================

def collect_polygon(
    rgb: np.ndarray,
    plane_number: int,
    total_planes: int,
) -> np.ndarray:
    """
    Display RGB frame and collect polygon vertices.
    """

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    axis.imshow(
        rgb
    )

    axis.set_title(
        f"Plane {plane_number}/{total_planes}\n"
        "Click polygon vertices around ONE planar surface.\n"
        "Press ENTER when finished."
    )

    axis.axis(
        "off"
    )

    print()
    print(
        f"Annotating plane "
        f"{plane_number}/{total_planes}"
    )

    print(
        "Click at least 3 polygon vertices, "
        "then press ENTER."
    )

    clicked_points = plt.ginput(
        n=-1,
        timeout=0,
        show_clicks=True,
    )

    plt.close(
        figure
    )

    polygon = np.asarray(
        clicked_points,
        dtype=float,
    )

    if len(
        polygon
    ) < 3:

        raise RuntimeError(
            "A plane polygon requires at least 3 points."
        )

    return polygon


# ============================================================
# Pixel / polygon operations
# ============================================================

def points_inside_polygon(
    pixels_uv: np.ndarray,
    polygon_uv: np.ndarray,
) -> np.ndarray:
    """
    pixels_uv:
        shape (N,2), [u,v]

    Returns boolean mask.
    """

    polygon_path = MplPath(
        polygon_uv
    )

    return polygon_path.contains_points(
        pixels_uv.astype(
            float
        )
    )


# ============================================================
# Detector-point projection
# ============================================================

def project_points_to_image(
    points: np.ndarray,
    K: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Project detector cloud points into image.

    Returns:
        pixels_uv
        valid_projection
    """

    fx = float(
        K[0, 0]
    )

    fy = float(
        K[1, 1]
    )

    cx = float(
        K[0, 2]
    )

    cy = float(
        K[1, 2]
    )

    x = points[
        :,
        0
    ]

    y = points[
        :,
        1
    ]

    z = points[
        :,
        2
    ]

    valid = (
        z > EPS
    )

    u = np.full(
        len(
            points
        ),
        np.nan,
        dtype=float,
    )

    v = np.full(
        len(
            points
        ),
        np.nan,
        dtype=float,
    )

    u[
        valid
    ] = (

        fx
        * x[
            valid
        ]
        / z[
            valid
        ]

        + cx
    )

    v[
        valid
    ] = (

        fy
        * y[
            valid
        ]
        / z[
            valid
        ]

        + cy
    )

    pixels = np.column_stack(
        (
            u,
            v,
        )
    )

    return (
        pixels,
        valid,
    )


# ============================================================
# Reference construction for one frame
# ============================================================

def annotate_frame(
    frame_file: Path,
) -> dict:

    with np.load(
        frame_file,
        allow_pickle=False,
    ) as data:

        rgb = np.asarray(
            data[
                "rgb"
            ]
        )

        raw_points = np.asarray(
            data[
                "raw_points"
            ],
            dtype=float,
        )

        raw_pixels = np.asarray(
            data[
                "raw_pixels"
            ],
            dtype=float,
        )

        detector_points = np.asarray(
            data[
                "detector_points"
            ],
            dtype=float,
        )

        K = np.asarray(
            data[
                "camera_intrinsics"
            ],
            dtype=float,
        )

    frame_id = frame_id_from_path(
        frame_file
    )

    print()
    print("=" * 80)

    print(
        f"FRAME {frame_id:04d}"
    )

    print("=" * 80)

    print(
        f"Raw valid points : "
        f"{len(raw_points)}"
    )

    print(
        f"Detector points  : "
        f"{len(detector_points)}"
    )

    print()

    # --------------------------------------------------------
    # Show frame first
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 8)
    )

    plt.imshow(
        rgb
    )

    plt.title(
        f"TUM frame {frame_id:04d}"
    )

    plt.axis(
        "off"
    )

    plt.show()

    # --------------------------------------------------------
    # Ask number of major reference planes
    # --------------------------------------------------------

    while True:

        answer = input(
            "How many major planes do you want to "
            "annotate in this frame? "
        ).strip()

        try:

            number_planes = int(
                answer
            )

            if number_planes >= 1:

                break

        except ValueError:

            pass

        print(
            "Please enter a positive integer."
        )

    reference_planes = []

    # --------------------------------------------------------
    # Annotate each plane
    # --------------------------------------------------------

    for plane_index in range(
        number_planes
    ):

        while True:

            polygon = collect_polygon(

                rgb,

                plane_index + 1,

                number_planes,
            )

            inside_raw = (
                points_inside_polygon(
                    raw_pixels,
                    polygon,
                )
            )

            selected_raw_points = (
                raw_points[
                    inside_raw
                ]
            )

            print(
                f"Polygon contains "
                f"{len(selected_raw_points)} "
                f"valid depth points."
            )

            if (
                len(
                    selected_raw_points
                )
                < MIN_RAW_POINTS_FOR_PLANE
            ):

                print(
                    "Too few valid depth points. "
                    "Please redraw this plane."
                )

                continue

            try:

                (
                    normal,
                    offset,
                    centroid,
                    fit_indices,
                ) = robust_fit_plane(
                    selected_raw_points
                )

            except Exception as error:

                print(
                    f"Plane fitting failed: "
                    f"{error}"
                )

                print(
                    "Please redraw."
                )

                continue

            fitted_points = (
                selected_raw_points[
                    fit_indices
                ]
            )

            distances = np.abs(
                fitted_points
                @ normal
                + offset
            )

            rms = float(
                np.sqrt(
                    np.mean(
                        distances ** 2
                    )
                )
            )

            print()
            print(
                f"Plane {plane_index}:"
            )

            print(
                f"  normal = "
                f"{normal}"
            )

            print(
                f"  offset = "
                f"{offset:.5f} m"
            )

            print(
                f"  fitted points = "
                f"{len(fitted_points)}"
            )

            print(
                f"  RMS residual = "
                f"{rms * 1000:.2f} mm"
            )

            confirmation = input(
                "Accept this plane? [Y/n]: "
            ).strip().lower()

            if confirmation in {
                "",
                "y",
                "yes",
            }:

                reference_planes.append(
                    {
                        "normal":
                            normal,

                        "offset":
                            offset,

                        "centroid":
                            centroid,

                        "polygon":
                            polygon,

                        "raw_support":
                            len(
                                fitted_points
                            ),

                        "rms_m":
                            rms,
                    }
                )

                break

            print(
                "Redrawing plane..."
            )

    # ========================================================
    # Map manually defined planes to detector_points
    # ========================================================

    (
        detector_pixels,
        projection_valid,
    ) = project_points_to_image(
        detector_points,
        K,
    )

    number_detector_points = len(
        detector_points
    )

    number_reference_planes = len(
        reference_planes
    )

    # One support mask per GT plane.
    support_masks = np.zeros(
        (
            number_reference_planes,
            number_detector_points,
        ),
        dtype=bool,
    )

    # Distance matrix used to resolve overlap conflicts.
    distance_matrix = np.full(
        (
            number_reference_planes,
            number_detector_points,
        ),
        np.inf,
        dtype=float,
    )

    for plane_index, plane in enumerate(
        reference_planes
    ):

        normal = plane[
            "normal"
        ]

        offset = plane[
            "offset"
        ]

        polygon = plane[
            "polygon"
        ]

        inside_polygon = np.zeros(
            number_detector_points,
            dtype=bool,
        )

        valid_indices = np.flatnonzero(
            projection_valid
        )

        valid_pixels = (
            detector_pixels[
                valid_indices
            ]
        )

        valid_inside = (
            points_inside_polygon(
                valid_pixels,
                polygon,
            )
        )

        inside_polygon[
            valid_indices[
                valid_inside
            ]
        ] = True

        distances = np.abs(
            detector_points
            @ normal
            + offset
        )

        geometric_support = (
            distances
            <= REFERENCE_SUPPORT_DISTANCE_M
        )

        support = (
            inside_polygon
            & geometric_support
        )

        support_masks[
            plane_index
        ] = support

        distance_matrix[
            plane_index,
            support
        ] = distances[
            support
        ]

    # --------------------------------------------------------
    # Remove extremely small reference supports
    # --------------------------------------------------------

    valid_plane_mask = np.array(
        [

            np.count_nonzero(
                support_masks[
                    plane_index
                ]
            )
            >= MIN_REFERENCE_DETECTOR_POINTS

            for plane_index
            in range(
                number_reference_planes
            )
        ],
        dtype=bool,
    )

    if not np.all(
        valid_plane_mask
    ):

        removed = np.flatnonzero(
            ~valid_plane_mask
        )

        print()
        print(
            "WARNING:"
        )

        print(
            "These manually annotated planes had too "
            "few detector-cloud support points and "
            "will be removed:"
        )

        print(
            removed.tolist()
        )

        reference_planes = [

            plane

            for index, plane
            in enumerate(
                reference_planes
            )

            if valid_plane_mask[
                index
            ]
        ]

        support_masks = (
            support_masks[
                valid_plane_mask
            ]
        )

        distance_matrix = (
            distance_matrix[
                valid_plane_mask
            ]
        )

    number_reference_planes = len(
        reference_planes
    )

    if number_reference_planes == 0:

        raise RuntimeError(
            "No valid reference planes remain."
        )

    # --------------------------------------------------------
    # Resolve possible support overlap
    #
    # If a detector point satisfies multiple reference planes,
    # assign it to the nearest plane.
    # --------------------------------------------------------

    gt_labels = np.full(
        number_detector_points,
        -1,
        dtype=np.int32,
    )

    minimum_distances = np.min(
        distance_matrix,
        axis=0,
    )

    nearest_planes = np.argmin(
        distance_matrix,
        axis=0,
    )

    assigned = np.isfinite(
        minimum_distances
    )

    gt_labels[
        assigned
    ] = nearest_planes[
        assigned
    ].astype(
        np.int32
    )

    # Rebuild exclusive support masks from final labels.
    support_masks = np.stack(
        [

            gt_labels
            == plane_index

            for plane_index
            in range(
                number_reference_planes
            )
        ],
        axis=0,
    )

    # ========================================================
    # Pack output arrays
    # ========================================================

    gt_normals = np.stack(
        [
            plane[
                "normal"
            ]

            for plane
            in reference_planes
        ],
        axis=0,
    )

    gt_offsets = np.asarray(
        [
            plane[
                "offset"
            ]

            for plane
            in reference_planes
        ],
        dtype=float,
    )

    gt_centroids = np.stack(
        [
            plane[
                "centroid"
            ]

            for plane
            in reference_planes
        ],
        axis=0,
    )

    # Polygons are variable length in general.
    # Here ginput can produce different vertex counts,
    # so save as padded array.
    max_vertices = max(
        len(
            plane[
                "polygon"
            ]
        )

        for plane
        in reference_planes
    )

    polygons = np.full(
        (
            number_reference_planes,
            max_vertices,
            2,
        ),
        np.nan,
        dtype=float,
    )

    polygon_lengths = np.zeros(
        number_reference_planes,
        dtype=np.int32,
    )

    for plane_index, plane in enumerate(
        reference_planes
    ):

        polygon = plane[
            "polygon"
        ]

        polygons[
            plane_index,
            :len(
                polygon
            ),
            :,
        ] = polygon

        polygon_lengths[
            plane_index
        ] = len(
            polygon
        )

    gt_raw_support_counts = np.asarray(
        [
            plane[
                "raw_support"
            ]

            for plane
            in reference_planes
        ],
        dtype=np.int32,
    )

    gt_detector_support_counts = np.asarray(
        [
            np.count_nonzero(
                gt_labels
                == plane_index
            )

            for plane_index
            in range(
                number_reference_planes
            )
        ],
        dtype=np.int32,
    )

    gt_fit_rms_m = np.asarray(
        [
            plane[
                "rms_m"
            ]

            for plane
            in reference_planes
        ],
        dtype=float,
    )

    return {

        "frame_id":
            frame_id,

        "gt_normals":
            gt_normals,

        "gt_offsets":
            gt_offsets,

        "gt_centroids":
            gt_centroids,

        "gt_labels":
            gt_labels,

        "gt_support_masks":
            support_masks,

        "gt_polygons_uv":
            polygons,

        "gt_polygon_lengths":
            polygon_lengths,

        "gt_raw_support_counts":
            gt_raw_support_counts,

        "gt_detector_support_counts":
            gt_detector_support_counts,

        "gt_fit_rms_m":
            gt_fit_rms_m,
    }


# ============================================================
# Select frames
# ============================================================

def get_frame_files() -> List[Path]:

    frame_files = sorted(
        FRAMES_DIR.glob(
            "frame_*.npz"
        )
    )

    if not frame_files:

        raise FileNotFoundError(
            "\nNo prepared TUM frames found:\n"
            f"{FRAMES_DIR}\n\n"
            "Run prepare_tum_plane_detection.py first."
        )

    if REFERENCE_FRAME_IDS is None:

        return frame_files

    requested_ids = set(
        REFERENCE_FRAME_IDS
    )

    selected = [

        path

        for path
        in frame_files

        if frame_id_from_path(
            path
        ) in requested_ids
    ]

    missing = (

        requested_ids

        - {
            frame_id_from_path(
                path
            )
            for path
            in selected
        }
    )

    if missing:

        print(
            "WARNING: requested frame IDs were "
            "not found:"
        )

        print(
            sorted(
                missing
            )
        )

    return selected


# ============================================================
# Save manifest
# ============================================================

def save_reference_manifest(
    entries: list,
):

    data = {

        "sequence_name":
            SEQUENCE_NAME,

        "method":
            (
                "Manual RGB polygon annotation followed by "
                "depth-based robust 3-D plane fitting."
            ),

        "plane_equation":
            "n^T x + d = 0",

        "robust_fit_distance_m":
            ROBUST_FIT_DISTANCE_M,

        "robust_fit_iterations":
            ROBUST_FIT_ITERATIONS,

        "reference_support_distance_m":
            REFERENCE_SUPPORT_DISTANCE_M,

        "minimum_raw_points":
            MIN_RAW_POINTS_FOR_PLANE,

        "minimum_detector_support":
            MIN_REFERENCE_DETECTOR_POINTS,

        "frames":
            entries,
    }

    with open(
        REFERENCE_MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# Main
# ============================================================

def main():

    REFERENCES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_files = (
        get_frame_files()
    )

    print()
    print("=" * 80)

    print(
        "TUM PLANE REFERENCE BUILDER"
    )

    print("=" * 80)

    print(
        f"Sequence:"
    )

    print(
        SEQUENCE_NAME
    )

    print()

    print(
        f"Frames selected:"
    )

    print(
        [
            frame_id_from_path(
                path
            )
            for path
            in frame_files
        ]
    )

    print()
    print(
        "Reference generation method:"
    )

    print(
        "Manual polygon annotation + "
        "depth-based robust SVD plane fitting."
    )

    print("=" * 80)

    manifest_entries = []

    for frame_file in frame_files:

        frame_id = frame_id_from_path(
            frame_file
        )

        output_file = (
            REFERENCES_DIR
            / frame_file.name
        )

        # ----------------------------------------------------
        # Existing reference
        # ----------------------------------------------------

        if (
            output_file.exists()
            and not OVERWRITE_EXISTING_REFERENCE
        ):

            print()
            print(
                f"Skipping frame {frame_id:04d}: "
                f"reference already exists."
            )

            with np.load(
                output_file,
                allow_pickle=False,
            ) as existing:

                number_planes = len(
                    existing[
                        "gt_normals"
                    ]
                )

            manifest_entries.append(
                {
                    "frame_id":
                        frame_id,

                    "reference_file":
                        output_file.name,

                    "number_reference_planes":
                        int(
                            number_planes
                        ),

                    "status":
                        "existing",
                }
            )

            continue

        # ----------------------------------------------------
        # New annotation
        # ----------------------------------------------------

        reference = annotate_frame(
            frame_file
        )

        np.savez_compressed(

            output_file,

            gt_normals=(
                reference[
                    "gt_normals"
                ]
            ),

            gt_offsets=(
                reference[
                    "gt_offsets"
                ]
            ),

            gt_centroids=(
                reference[
                    "gt_centroids"
                ]
            ),

            gt_labels=(
                reference[
                    "gt_labels"
                ]
            ),

            gt_support_masks=(
                reference[
                    "gt_support_masks"
                ]
            ),

            gt_polygons_uv=(
                reference[
                    "gt_polygons_uv"
                ]
            ),

            gt_polygon_lengths=(
                reference[
                    "gt_polygon_lengths"
                ]
            ),

            gt_raw_support_counts=(
                reference[
                    "gt_raw_support_counts"
                ]
            ),

            gt_detector_support_counts=(
                reference[
                    "gt_detector_support_counts"
                ]
            ),

            gt_fit_rms_m=(
                reference[
                    "gt_fit_rms_m"
                ]
            ),
        )

        number_planes = len(
            reference[
                "gt_normals"
            ]
        )

        manifest_entries.append(
            {
                "frame_id":
                    frame_id,

                "reference_file":
                    output_file.name,

                "number_reference_planes":
                    int(
                        number_planes
                    ),

                "status":
                    "annotated",
            }
        )

        print()
        print(
            f"Saved reference:"
        )

        print(
            output_file
        )

        print(
            f"Reference planes: "
            f"{number_planes}"
        )

    save_reference_manifest(
        manifest_entries
    )

    print()
    print("=" * 80)

    print(
        "REFERENCE GENERATION COMPLETE"
    )

    print("=" * 80)

    print(
        f"Reference folder:"
    )

    print(
        REFERENCES_DIR
    )

    print()

    print(
        f"Manifest:"
    )

    print(
        REFERENCE_MANIFEST_FILE
    )

    print("=" * 80)
    print()


if __name__ == "__main__":

    main()