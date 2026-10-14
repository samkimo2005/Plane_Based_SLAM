import sys
from pathlib import Path


# Find project root directory.
project_root = Path(__file__).resolve().parents[1]

# Allow Python to import tum_loader.py from src/.
sys.path.insert(
    0,
    str(project_root / "src")
)


from tum_loader import (
    read_groundtruth,
    find_nearest_groundtruth,
    TUMDataset
)


dataset_dir = (
    project_root
    / "data"
    / "tum"
    / "rgbd_dataset_freiburg1_xyz"
)


print(
    "=== 2026-09-17 Ground-Truth Loader Verification ==="
)


# --------------------------------------------------
# Test 1:
# Read groundtruth.txt
# --------------------------------------------------

print(
    "\nTest 1: Read groundtruth.txt"
)

groundtruth_list = read_groundtruth(
    dataset_dir / "groundtruth.txt"
)

print(
    "Ground-truth entries:",
    len(groundtruth_list)
)

print(
    "\nFirst ground-truth entry:"
)

print(
    groundtruth_list[0]
)


# --------------------------------------------------
# Test 2:
# Find nearest GT pose to Frame 0 timestamp
# --------------------------------------------------

print(
    "\nTest 2: Find nearest ground-truth pose"
)

dataset = TUMDataset(
    dataset_dir
)

frame = dataset.load_frame(0)

groundtruth = find_nearest_groundtruth(
    groundtruth_list,
    frame["rgb_timestamp"]
)

print(
    "RGB timestamp:",
    frame["rgb_timestamp"]
)

if groundtruth is None:

    print(
        "No suitable ground truth found."
    )

else:

    print(
        "GT timestamp:",
        groundtruth["timestamp"]
    )

    print(
        "Translation:",
        groundtruth["translation"]
    )

    print(
        "Quaternion:",
        groundtruth["quaternion"]
    )

    print(
        "Time difference:",
        groundtruth["time_difference"],
        "seconds"
    )


# --------------------------------------------------
# Test 3:
# Verify Ground Truth is included directly
# in the loaded frame.
# --------------------------------------------------

print(
    "\nTest 3: Ground truth inside Frame structure"
)

frame = dataset.load_frame(0)

if frame["groundtruth"] is None:

    raise RuntimeError(
        "No ground truth associated with Frame 0."
    )

print(
    "RGB timestamp:",
    frame["rgb_timestamp"]
)

print(
    "Ground-truth timestamp:",
    frame["groundtruth"]["timestamp"]
)

print(
    "GT time difference:",
    frame["groundtruth"]["time_difference"],
    "seconds"
)

print(
    "Ground-truth translation:",
    frame["groundtruth"]["translation"]
)

print(
    "Ground-truth quaternion:",
    frame["groundtruth"]["quaternion"]
)


print(
    "\n=== 2026-09-17 Verification PASSED ==="
)