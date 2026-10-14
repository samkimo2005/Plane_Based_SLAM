

from pathlib import Path
from bisect import bisect_left

import cv2

## Created on 16 September 2026
def read_image_list(txt_file):
    """
    Read a TUM RGB-D index file such as rgb.txt or depth.txt.

    Returns:
        list of tuples:
        [(timestamp, relative_path), ...]
    """

    entries = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Ignore empty lines and comments.
            if not line or line.startswith("#"):
                continue

            # Split the line using spaces or tabs.
            parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"Invalid line in {txt_file}: {line}"
                )

            timestamp = float(parts[0])
            image_path = parts[1]

            entries.append(
                (timestamp, image_path)
            )

    return entries

def associate_rgb_depth(
    rgb_list,
    depth_list,
    max_difference=0.02
):
    """
    Associate each RGB frame with the nearest depth frame.

    max_difference:
        Maximum allowed timestamp difference in seconds.
    """

    depth_timestamps = [
        entry[0]
        for entry in depth_list
    ]

    matches = []

    for rgb_timestamp, rgb_path in rgb_list:

        pos = bisect_left(
            depth_timestamps,
            rgb_timestamp
        )

        candidates = []

        if pos < len(depth_list):
            candidates.append(
                depth_list[pos]
            )

        if pos > 0:
            candidates.append(
                depth_list[pos - 1]
            )

        if not candidates:
            continue

        depth_timestamp, depth_path = min(
            candidates,
            key=lambda x: abs(
                x[0] - rgb_timestamp
            )
        )

        time_difference = abs(
            rgb_timestamp - depth_timestamp
        )

        if time_difference <= max_difference:
            matches.append(
                (
                    rgb_timestamp,
                    rgb_path,
                    depth_timestamp,
                    depth_path,
                    time_difference
                )
            )

    return matches

def load_rgb_depth_pair(dataset_dir, match):
    """
    Load one associated RGB-depth pair.
    """

    dataset_dir = Path(dataset_dir)

    (
        rgb_timestamp,
        rgb_path,
        depth_timestamp,
        depth_path,
        time_difference
    ) = match

    rgb_file = dataset_dir / rgb_path
    depth_file = dataset_dir / depth_path

    rgb = cv2.imread(
        str(rgb_file),
        cv2.IMREAD_COLOR
    )

    depth = cv2.imread(
        str(depth_file),
        cv2.IMREAD_UNCHANGED
    )

    if rgb is None:
        raise FileNotFoundError(
            f"Could not load RGB image: {rgb_file}"
        )

    if depth is None:
        raise FileNotFoundError(
            f"Could not load depth image: {depth_file}"
        )

    return {
        "rgb_timestamp": rgb_timestamp,
        "depth_timestamp": depth_timestamp,
        "time_difference": time_difference,
        "rgb": rgb,
        "depth": depth
    }

def read_groundtruth(txt_file):
    """
    Read TUM groundtruth.txt.

    Returns:
        list of dictionaries containing:
        timestamp, translation, quaternion
    """

    entries = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Ignore empty lines and comments.
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 8:
                raise ValueError(
                    f"Invalid ground-truth line: {line}"
                )

            timestamp = float(parts[0])

            tx = float(parts[1])
            ty = float(parts[2])
            tz = float(parts[3])

            qx = float(parts[4])
            qy = float(parts[5])
            qz = float(parts[6])
            qw = float(parts[7])

            entries.append(
                {
                    "timestamp": timestamp,
                    "translation": (tx, ty, tz),
                    "quaternion": (qx, qy, qz, qw)
                }
            )

    return entries

def find_nearest_groundtruth(
    groundtruth_list,
    timestamp,
    max_difference=0.02
):
    """
    Find the nearest ground-truth pose to a given timestamp.
    """

    groundtruth_timestamps = [
        entry["timestamp"]
        for entry in groundtruth_list
    ]

    pos = bisect_left(
        groundtruth_timestamps,
        timestamp
    )

    candidates = []

    if pos < len(groundtruth_list):
        candidates.append(
            groundtruth_list[pos]
        )

    if pos > 0:
        candidates.append(
            groundtruth_list[pos - 1]
        )

    if not candidates:
        return None

    nearest = min(
        candidates,
        key=lambda entry: abs(
            entry["timestamp"] - timestamp
        )
    )

    time_difference = abs(
        nearest["timestamp"] - timestamp
    )

    if time_difference > max_difference:
        return None

    return {
        "timestamp": nearest["timestamp"],
        "translation": nearest["translation"],
        "quaternion": nearest["quaternion"],
        "time_difference": time_difference
    }

class TUMDataset:

    def __init__(
        self,
        dataset_dir,
        load_groundtruth = True,
        intrinsics = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.intrinsics = intrinsics

        self.rgb_list = read_image_list(
            self.dataset_dir / "rgb.txt"
        )

        self.depth_list = read_image_list(
            self.dataset_dir / "depth.txt"
        )

        self.matches = associate_rgb_depth(
            self.rgb_list,
            self.depth_list
        )

        self.groundtruth_list = None

        if load_groundtruth:
            self.groundtruth_list = read_groundtruth(
                self.dataset_dir / "groundtruth.txt"
            )

    def __len__(self):
        return len(self.matches)

    def load_frame(self, index):
        frame = load_rgb_depth_pair(
            self.dataset_dir,
            self.matches[index]
        )

        frame["index"] = index
        frame["intrinsics"] = self.intrinsics
        
        if self.groundtruth_list is not None:
            frame["groundtruth"] = (
                find_nearest_groundtruth(
                    self.groundtruth_list,
                    frame["rgb_timestamp"]
                )
            )
        else:
            frame["groundtruth"] = None
        return frame


# Created on 17 September
