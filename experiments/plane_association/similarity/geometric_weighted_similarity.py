import numpy as np

from similarity_common import (
    SimilaritySample,
    common_geometric_features,
)


def compute_score(sample: SimilaritySample, params=None) -> float:
    params = params or {}

    w_n = params.get("w_normal", 0.4)
    w_d = params.get("w_offset", 0.4)
    w_c = params.get("w_centroid", 0.2)

    f = common_geometric_features(sample)

    s_n = np.exp(-f["normal_angle_deg"] / 10.0)
    s_d = np.exp(-f["offset_difference_m"] / 0.10)
    s_c = np.exp(-f["centroid_distance_m"] / 0.30)

    return float(
        w_n * s_n
        + w_d * s_d
        + w_c * s_c
    )