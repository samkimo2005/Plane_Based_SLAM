import sys 
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(project_root / "src")
)

from tum_loader import TUMDataset

dataset_dir = (
    project_root
    / "data"
    / "tum"
    / "rgbd_dataset_freiburg1_xyz"
)

# Build up the structure of the intrinsics
intrinsics = {
    "fx": None,
    "fy": None,
    "cx": None,
    "cy": None,
    "Width": 640,
    "Height": 480
}

# Build up the localised dataset
dataset = TUMDataset(
    dataset_dir,
    intrinsics = intrinsics
)

# --------------------------------------------------
# Test 1:
# Read a frame & print all the keys
# --------------------------------------------------
frame = dataset.load_frame(0)
print(
    "\nTest 1: Unified frame structure test"
)

print(
    "Frame keys: "
)

for key in frame.keys():
    print(
        "-",
        key
    )

# print the keys with one following each other
print(
    "\nFrame index:",
    frame["index"]
)

print(
    "RGB timestamp:",
    frame["rgb_timestamp"]
)

print(
    "Depth timestamp:",
    frame["depth_timestamp"]
)

print(
    "Timestamp difference:",
    frame["time_difference"]
)

print(
    "RGB shape:",
    frame["rgb"].shape
)

print(
    "Depth shape:",
    frame["depth"].shape
)

print(
    "Intrinsics:",
    frame["intrinsics"]
)

print(
    "Ground truth:",
    frame["groundtruth"]
)

# --------------------------------------------------
# Test 2:
# Missing Key Test
# --------------------------------------------------
print(
    "\nTest 2: Missing key test"
)
required_keys = [
    "index",
    "rgb",
    "depth",
    "rgb_timestamp",
    "depth_timestamp",
    "time_difference",
    "intrinsics",
    "groundtruth"
]

for key in required_keys:
    assert key in frame, (
        f"Missing frame field: {key}"
    )

assert isinstance(
    frame["index"],
    int
)

assert frame["rgb"].shape == (
    480,
    640,
    3
)

assert frame["depth"].shape == (
    480,
    640
)

assert frame["intrinsics"] is not None
assert frame["groundtruth"] is not None

print(
    "\n=== Unified Frame Structure Test PASSED ==="
)