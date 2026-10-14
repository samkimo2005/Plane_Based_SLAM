import sys
from pathlib import Path


# Find project root directory.
project_root = Path(__file__).resolve().parents[1]

# Allow Python to import tum_loader.py from src/.
sys.path.insert(
    0,
    str(project_root / "src")
)

# Import the loader
from tum_loader import (
    read_image_list,
    associate_rgb_depth,
    load_rgb_depth_pair,
    TUMDataset
)


dataset_dir = (
    project_root
    / "data"
    / "tum"
    / "rgbd_dataset_freiburg1_xyz"
)


print(
    "=== 2026-09-16 TUM RGB-D Loader Verification ==="
)


# --------------------------------------------------
# Test 1:
# Read rgb.txt and depth.txt
# --------------------------------------------------

print("\nTest 1: Read RGB and depth index files")

rgb_list = read_image_list(
    dataset_dir / "rgb.txt"
)

depth_list = read_image_list(
    dataset_dir / "depth.txt"
)

print(
    "RGB entries:",
    len(rgb_list)
)

print(
    "Depth entries:",
    len(depth_list)
)


# --------------------------------------------------
# Test 2:
# RGB-depth timestamp association
# --------------------------------------------------

print(
    "\nTest 2: RGB-depth timestamp association"
)

matches = associate_rgb_depth(
    rgb_list,
    depth_list
)

print(
    "Associated pairs:",
    len(matches)
)

(
    rgb_timestamp,
    rgb_path,
    depth_timestamp,
    depth_path,
    time_difference
) = matches[0]

print(
    "RGB timestamp:",
    rgb_timestamp
)

print(
    "RGB path:",
    rgb_path
)

print(
    "Depth timestamp:",
    depth_timestamp
)

print(
    "Depth path:",
    depth_path
)

print(
    "Time difference:",
    time_difference,
    "seconds"
)


# --------------------------------------------------
# Test 3:
# Load one RGB-depth pair
# --------------------------------------------------

print(
    "\nTest 3: Load RGB-depth image pair"
)

frame = load_rgb_depth_pair(
    dataset_dir,
    matches[0]
)

print(
    "RGB shape:",
    frame["rgb"].shape
)

print(
    "RGB dtype:",
    frame["rgb"].dtype
)

print(
    "Depth shape:",
    frame["depth"].shape
)

print(
    "Depth dtype:",
    frame["depth"].dtype
)


# --------------------------------------------------
# Test 4:
# Test TUMDataset interface
# Ground truth deliberately disabled.
# --------------------------------------------------

print(
    "\nTest 4: TUMDataset frame interface"
)

dataset = TUMDataset(
    dataset_dir,
    load_groundtruth=False
)

print(
    "Number of frames:",
    len(dataset)
)

frame = dataset.load_frame(0)

print(
    "Frame 0 RGB timestamp:",
    frame["rgb_timestamp"]
)

print(
    "Frame 0 depth timestamp:",
    frame["depth_timestamp"]
)

print(
    "Frame 0 RGB shape:",
    frame["rgb"].shape
)

print(
    "Frame 0 depth shape:",
    frame["depth"].shape
)


print(
    "\n=== 2026-09-16 Verification PASSED ==="
)