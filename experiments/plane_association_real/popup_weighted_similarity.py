import numpy as np

from similarity_common import (
    SimilaritySample,
    common_geometric_features,
)


def compute_score(sample: SimilaritySample, params=None) -> float:
    params = params or {}

    w_n = params.get("w_normal", 0.50)
    w_c = params.get("w_centroid", 0.30)
    w_a = params.get("w_area", 0.20)

    f = common_geometric_features(sample)

    s_n = np.exp(-f["normal_angle_deg"] / 10.0)
    s_c = np.exp(-f["centroid_distance_m"] / 0.30)

    area_diff = f.get("relative_area_difference", 0.0)
    s_a = np.exp(-area_diff)

    score = (
        w_n * s_n
        + w_c * s_c
        + w_a * s_a
    )

    return float(score)