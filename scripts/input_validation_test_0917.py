import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Find the project root directory
project_root = Path(__file__).resolve().parents[1]

# Allow Python to import modules from src/
sys.path.insert(
    0,
    str(project_root / "src")
)

# Import the loader
from tum_loader import TUMDataset


dataset_dir = (
    project_root
    / "data"
    / "tum"
    / "rgbd_dataset_freiburg1_xyz"
)

dataset = TUMDataset(dataset_dir)

print(
    "=== RGB-D Input Validation ==="
)

print(
    "Number of associated frames:",
    len(dataset)
)

# --------------------------------------------------
# Test 1:
# Check the Basic Properties of A Frame
# --------------------------------------------------

print("\nTest 1: Check the Basic Properties of A Frame")

frame = dataset.load_frame(0)

rgb = frame["rgb"]
depth = frame["depth"]

print(
    "RGB shape:",
    rgb.shape
)

print(
    "RGB dtype:",
    rgb.dtype
)

print(
    "Depth shape:",
    depth.shape
)

print(
    "Depth dtype:",
    depth.dtype
)

print(
    "RGB-depth timestamp difference:",
    frame["time_difference"],
    "seconds"
)

# --------------------------------------------------
# Test 2:
# Depth Validity
# --------------------------------------------------

print("\nTest 2: Depth Validity")
valid_mask = depth > 0
valid_depth = depth[valid_mask]
invalid_count = np.count_nonzero(
    depth == 0
)
valid_count = np.count_nonzero(
    valid_mask
)

total_count = depth.size

print(
    "Total depth pixels:",
    total_count
)

print(
    "Total valid depth pixels:",
    valid_count
)

print(
    "Total invalid depth pixels:",
    invalid_count
)

# --------------------------------------------------
# Test 3:
# Check the range of the depth
# --------------------------------------------------
if valid_depth.size > 0:

    min_depth_raw = valid_depth.min()
    min_depth = min_depth_raw / 5000.0 # Z = D/5000
    max_depth_raw = valid_depth.max()
    max_depth = max_depth_raw / 5000.0
    # D: Raw depth value from the pixel of the depth image
    # Z: Real depth after analyzing the raw depth

    print(
        "The raw value of minimum depth: ",
        min_depth_raw
    )

    print(
        "The real minimum depth: ",
        min_depth,
        "m"
    )

    print(
        "The raw value of the maximum depth: ",
        max_depth_raw
    )

    print(
        "The real maximum depth: ",
        max_depth,
        "m"
    )

# --------------------------------------------------
# Test 4:
# Display the RGB-D image for a single frame
# --------------------------------------------------

print("\nTest 4: Display the RGB-D image")
# Convert the BGR read by cv2 into RGB format
rgb_display = cv2.cvtColor(
    rgb,
    cv2.COLOR_BGR2RGB
)

# Generate the RGB image
plt.figure()
plt.imshow(rgb_display)
plt.title(
    "Frame 0 - RGB Image"
)
plt.axis("off")
plt.show()

# Generate the depth image
plt.figure()
plt.imshow(depth)
plt.title(
    "Frame 0 - Depth Image"
)
plt.colorbar(
    label="Raw depth value"
)
plt.axis("off")
plt.show()

# --------------------------------------------------
# Test 5:
# Display the RGB-D images for multiple frames
# --------------------------------------------------
print("\nTest 5: Multiframe validation")

# Choose ten frames
test_indices = [0, 80, 160, 240, 320, 400, 480, 560, 640, 720]

for index in test_indices:
    frame = dataset.load_frame(index)

    rgb = frame["rgb"]
    depth = frame["depth"]

    valid_depth = depth[
        depth > 0
    ]

    print(
        f"\nFrame{index}"
    )

    print(
        "RGB shape:",
        rgb.shape
    )

    print(
        "Depth shape:",
        depth.shape
    )

    print(
        "RGB dtype: ",
        rgb.dtype
    )

    print(
        "Depth dtype: ",
        depth.dtype
    )

    print(
        "Timestamp difference: ",
        frame["time_difference"]
    )

    if valid_depth.size > 0:
        print(
            "Depth_range: ",
            valid_depth.min()/5000.0,
            "to",
            valid_depth.max()/5000.0
        )

    # Automatic verification thresholds
    assert rgb.shape == (480, 640, 3), (
        f"Unexpected RGB shape at Frame {index}"
    )

    assert depth.shape == (480, 640), (
        f"Unexpected depth shape at Frame {index}"
    )

    assert rgb.dtype == np.uint8, (
        f"Unexpected RGB dtype at Frame {index}"
    )

    assert depth.dtype == np.uint16, (
        f"Unexpected depth dtype at Frame {index}"
    )

    assert frame["time_difference"] <= 0.02, (
        f"RGB-depth timestamp mismatch at Frame {index}"
    )

print(
    "\n=== All RGB-D Input Validation Tests PASSED ==="
)