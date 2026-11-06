# Attempt F -> Bad-Match Rejection : real-data handoff

This folder contains actual saved **Assignment outputs** packaged for the next stage. No algorithm was rerun.

## Which file to feed into Bad-Match Rejection?

**`bmr_assigned_pairs.csv`** — PRIMARY INPUT, 282 selected correspondence records from the saved Jonker–Volgenant results. Includes `case_id`, `match_id`, frame and plane IDs, gating/similarity method, similarity score, transformed plane normals and offsets, and measured geometric residuals. It deliberately has **no GT labels**.

Use `case_id` to split evaluation cases. Do NOT run correspondence rejection across unrelated cases or combine the same physical pair across different configurations without keeping their case provenance.

`bmr_case_manifest.csv` — all 64 experimental cases, including the 16 HIA-TCD cases with zero selected matches. Each nonempty case has 4–8 matches. Only 48 cases have any assigned pairs.

`bmr_unmatched_planes.csv` — 440 unmatched-plane rows from JV (some zero-candidate planes are omitted upstream).

`bmr_plane_features_original_frames.csv` — 48 original frame-local plane features (normals, centroids, support counts), obtained from GT_Manual(5).xlsx **without** the GT physical-surface identities. These are useful when an algorithm needs the original planes; centroids from different frames are not in a common coordinate system.

## For evaluation ONLY — do not feed into the rejection algorithm

`bmr_gt_labels_EVALUATION_ONLY.csv` — one-to-one join via `match_id`: 1 = same manually labelled physical surface, 0 = different; note no explicit per-patch verification-status field in the source workbook.

`bmr_gt_surface_ids_EVALUATION_ONLY.csv` — all 48 physical surface labels, for reference/diagnostics only.

`bmr_unique_pair_audit_EVALUATION_ONLY.csv` — 38 unique frame-plane pairs over 282 case-specific records; repeated case evaluations are not independent physical correspondences.

## Critical caveats

* 280 labelled positive vs 2 labelled negative selected pairs (both Pop-up Weighted with normal-based gating). **This is not enough negative data to establish reliable rejection Precision/Recall/F1** on varied failures. Do not fabricate bad matches and call them naturally occurring.
* Scores from different Similarity methods are not calibrated. HIA-TCD selected no pairs under the shared unmatched cost.
* The pair-wise transformed plane normals/offsets and scalar geometric residuals can support elementary normal-/offset-/distance-based rejection. RANSAC geometric verification or SelMaP-style multi-match consistency may need original-frame plane geometry, verified pose transforms and/or per-plane 3D support points; those additional arrays are **not** in this handoff. The previous `prepare_bad_match_rejection_tum.py` was developed for a different TUM input layout, so the current files are **not** drop-in compatible with its NPZ schema.
* Physical surface IDs allow over-segmentation, but a one-to-one Assignment can select at most one target per source and vice versa. Evaluate GT carefully and keep the train/tune/test separation.
* All 282 matching records belong to the same four frame pairs, evaluated across multiple gating/similarity configurations.

## Example Python loader

```python
import csv

with open('bmr_assigned_pairs.csv', newline='', encoding='utf-8') as f:
    pairs = list(csv.DictReader(f))

# Group by case_id to evaluate rejection case by case.
from collections import defaultdict
cases = defaultdict(list)
for p in pairs:
    cases[p['case_id']].append(p)
```
