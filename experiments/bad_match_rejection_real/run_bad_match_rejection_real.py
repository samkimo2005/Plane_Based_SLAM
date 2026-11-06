"""Run the FIVE EXISTING Bad-Match Rejection implementations on prepared RealSense cases.

Algorithm code is imported, not reimplemented. Prepared .npz files contain
original-frame geometry and assignments; ground-truth labels are not read.

Structural failures trigger an explicit keep-all operational fallback (as in
run_bad_match_rejection_tum.py), not a false 'reject all' measurement.

Output: case_results.csv, match_decisions.csv, parameters_used.json.
This is NOT a pose-accuracy benchmark: real-data ground-truth R,t are absent.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

METHODS = ("Normal", "Distance", "RANSAC", "SelMaP-inspired", "Trilateration-inspired")


def project_root() -> Path:
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (start, *start.parents):
            if parent.name.lower() == "experiments":
                return parent
            if (parent / "experiments").is_dir():
                return parent / "experiments"
    return Path(__file__).resolve().parent


def setup_algorithm_imports(algorithm_dir: Path | None):
    here, root = Path(__file__).resolve().parent, project_root()
    dirs = [here, here / "algorithms", root / "scripts", root / "algorithms",
            root / "bad_match_rejection_real" / "algorithms",
            root / "bad_match_rejection_real", root]
    if algorithm_dir is not None:
        dirs.insert(0, algorithm_dir.expanduser().resolve())
    for path in reversed(dirs):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    names = {
        "common": "bad_match_rejection_common",
        "Normal": "normal_compatibility",
        "Distance": "distance_spatial_rejection",
        "RANSAC": "ransac_geometric_verification",
        "SelMaP-inspired": "selmap_compatibility",
        "Trilateration-inspired": "trilateration_rejection",
    }
    modules = {}
    try:
        for key, name in names.items():
            modules[key] = importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Missing algorithm module: {error}. Put your original algorithm files "
            "(WITHOUT the '(1)' filename suffix) and bad_match_rejection_common.py "
            "together in --algorithm-dir, or alongside this runner."
        ) from error

    for key in METHODS:
        if not callable(getattr(modules[key], "reject_bad_matches", None)):
            raise TypeError(f"{key} has no reject_bad_matches()")
    distance_args = set(inspect.signature(modules["Distance"].reject_bad_matches).parameters)
    if "offset_threshold" not in distance_args or "centroids_a" in distance_args:
        raise RuntimeError("Expected the restored offset-only Distance implementation")
    selmap_args = set(inspect.signature(modules["SelMaP-inspired"].reject_bad_matches).parameters)
    if not {"angle_threshold_deg", "separation_threshold", "min_compatible_fraction"} <= selmap_args:
        raise RuntimeError("Expected the corrected SelMaP-inspired implementation")
    return modules


def find_parameter_file(requested: Path | None) -> Path | None:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    here, root = Path(__file__).resolve().parent, project_root()
    for path in (
        root / "results" / "bad_match_rejection" / "tuning_v2" / "best_parameters.json",
        here / "best_parameters.json",
        here / "tuning_v2" / "best_parameters.json",
    ):
        if path.is_file():
            return path
    return None


def load_parameters(modules: dict, requested: Path | None, use_defaults: bool):
    found = find_parameter_file(requested)
    if found is None:
        if not use_defaults:
            raise FileNotFoundError(
                "Frozen DEV parameter JSON not found. Pass --parameter-file "
                ".../tuning_v2/best_parameters.json. To do ONLY a preliminary smoke "
                "test, explicitly pass --use-module-defaults (not a frozen comparison)."
            )
        defaults = {
            "Normal": {"normal_threshold_deg": modules["Normal"].NORMAL_THRESHOLD_DEG},
            "Distance": {"offset_threshold_m": modules["Distance"].OFFSET_THRESHOLD},
            "RANSAC": {
                "normal_threshold_deg": modules["RANSAC"].NORMAL_THRESHOLD_DEG,
                "offset_threshold_m": modules["RANSAC"].OFFSET_THRESHOLD,
                "iterations": modules["RANSAC"].RANSAC_ITERATIONS,
            },
            "SelMaP-inspired": {
                "angle_threshold_deg": modules["SelMaP-inspired"].ANGLE_THRESHOLD_DEG,
                "separation_threshold_m": modules["SelMaP-inspired"].SEPARATION_THRESHOLD,
                "min_compatible_fraction": modules["SelMaP-inspired"].MIN_COMPATIBLE_FRACTION,
                "parallel_threshold_deg": modules["SelMaP-inspired"].PARALLEL_THRESHOLD_DEG,
            },
            "Trilateration-inspired": {
                "cluster_radius_m": modules["Trilateration-inspired"].CLUSTER_RADIUS,
                "min_support": modules["Trilateration-inspired"].MIN_SUPPORT,
                "max_subsets": modules["Trilateration-inspired"].MAX_SUBSETS,
            },
        }
        return defaults, "MODULE DEFAULTS (UNFROZEN; SMOKE TEST ONLY)"
    with found.open(encoding="utf-8") as file:
        values = json.load(file)
    required = {
        "Normal": ("normal_threshold_deg",),
        "Distance": ("offset_threshold_m",),
        "RANSAC": ("normal_threshold_deg", "offset_threshold_m"),
        "SelMaP-inspired": ("angle_threshold_deg", "separation_threshold_m", "min_compatible_fraction"),
        "Trilateration-inspired": ("cluster_radius_m", "min_support"),
    }
    for method, fields in required.items():
        if method not in values or not all(k in values[method] for k in fields):
            raise ValueError(f"Frozen JSON missing {method} fields: {fields}")
    return values, str(found)


def load_case(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = {"planes_a", "planes_b", "assigned_pairs", "source_plane_ids",
                    "target_plane_ids", "case_id", "original_case_id", "scenario",
                    "source_frame", "target_frame"}
        if missing := required - set(data.files):
            raise ValueError(f"{path.name}: missing {sorted(missing)}")
        case = {key: data[key].copy() for key in data.files}
    pa, pb = np.asarray(case["planes_a"], float), np.asarray(case["planes_b"], float)
    pairs = np.asarray(case["assigned_pairs"], int).reshape(-1, 2)
    if pa.ndim != 2 or pa.shape[1] != 4 or pb.ndim != 2 or pb.shape[1] != 4:
        raise ValueError(f"{path.name}: expected Nx4 plane arrays")
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        raise ValueError(f"{path.name}: non-finite plane geometry")
    if len(pairs):
        if np.any(pairs < 0) or np.any(pairs[:, 0] >= len(pa)) or np.any(pairs[:, 1] >= len(pb)):
            raise ValueError(f"{path.name}: assignment index out of bounds")
        if len(np.unique(pairs[:, 0])) != len(pairs) or len(np.unique(pairs[:, 1])) != len(pairs):
            raise ValueError(f"{path.name}: assignment is not one-to-one")
    if len(case["source_plane_ids"]) != len(pa) or len(case["target_plane_ids"]) != len(pb):
        raise ValueError(f"{path.name}: Plane ID inventory length mismatch")
    case["planes_a"], case["planes_b"], case["assigned_pairs"] = pa, pb, pairs
    return case


def perform_method(method: str, modules: dict, params: dict, case: dict, seed: int):
    pa, pb, pairs = case["planes_a"], case["planes_b"], case["assigned_pairs"]
    if method == "Normal":
        return modules[method].reject_bad_matches(
            pa, pb, pairs,
            normal_threshold_deg=float(params[method]["normal_threshold_deg"])
        )
    if method == "Distance":
        return modules[method].reject_bad_matches(
            pa, pb, pairs,
            offset_threshold=float(params[method]["offset_threshold_m"])
        )
    if method == "RANSAC":
        return modules[method].reject_bad_matches(
            pa, pb, pairs, rng=np.random.default_rng(seed)
        )
    if method == "SelMaP-inspired":
        kw = {
            "angle_threshold_deg": float(params[method]["angle_threshold_deg"]),
            "separation_threshold": float(params[method]["separation_threshold_m"]),
            "min_compatible_fraction": float(params[method]["min_compatible_fraction"]),
        }
        if "parallel_threshold_deg" in params[method]:
            kw["parallel_threshold_deg"] = float(params[method]["parallel_threshold_deg"])
        return modules[method].reject_bad_matches(pa, pb, pairs, **kw)
    if method == "Trilateration-inspired":
        return modules[method].reject_bad_matches(pa, pb, pairs)
    raise KeyError(method)


def pose_diagnostics(common, case: dict, effective_keep: np.ndarray, max_condition: float):
    """Descriptive pose observability only; NO reference pose accuracy is claimed."""
    kept = case["assigned_pairs"][effective_keep]
    empty = {"pose_estimated": False, "pose_usable": False,
             "pose_rank": np.nan, "pose_condition_number": np.nan,
             "estimated_translation_norm_m": np.nan, "estimated_rotation_angle_deg": np.nan,
             "pose_failure_reason": ""}
    if len(kept) < 3:
        return {**empty, "pose_failure_reason": "fewer_than_3_kept"}
    try:
        pose = common.estimate_pose_from_plane_matches(
            case["planes_a"][kept[:, 0]], case["planes_b"][kept[:, 1]]
        )
        R = np.asarray(pose["R"], dtype=float)
        t = np.asarray(pose["t"], dtype=float)
        d = pose["translation_diagnostics"]
        cond, rank = float(d["condition_number"]), int(d["rank"])
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        numeric_ok = bool(np.isfinite(R).all() and np.isfinite(t).all())
        usable = numeric_ok and rank == 3 and np.isfinite(cond) and cond <= max_condition
        return {
            "pose_estimated": numeric_ok, "pose_usable": usable,
            "pose_rank": rank, "pose_condition_number": cond,
            "estimated_translation_norm_m": float(np.linalg.norm(t)) if numeric_ok else np.nan,
            "estimated_rotation_angle_deg": angle if numeric_ok else np.nan,
            "pose_failure_reason": "" if usable else "degenerate_or_ill_conditioned",
        }
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, KeyError) as error:
        return {**empty, "pose_failure_reason": f"{type(error).__name__}: {error}"}


def run_case_method(case: dict, method: str, modules: dict, params: dict,
                    seed: int, repeats: int, max_condition: float):
    pairs = case["assigned_pairs"]
    reason, error, result = "", "", None
    runtimes = []
    for repeat in range(repeats):
        start = time.perf_counter()
        try:
            candidate = perform_method(method, modules, params, case, seed)
            elapsed_ms = 1e3 * (time.perf_counter() - start)
            mask = np.asarray(candidate["keep_mask"], dtype=bool).reshape(-1)
            if len(mask) != len(pairs):
                raise ValueError("keep_mask length mismatch")
            if result is None:
                result = candidate
            elif not np.array_equal(mask, np.asarray(result["keep_mask"], bool)):
                raise ValueError("The same seeded case produced inconsistent decisions")
            runtimes.append(elapsed_ms)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    if result is not None:
        diagnosis = result.get("diagnostics", {})
        if isinstance(diagnosis, dict):
            reason = str(diagnosis.get("reason") or "")
    if error:
        reason = "exception"
    structural_failure = bool(reason) or result is None
    raw_mask = np.asarray(result["keep_mask"], bool).reshape(-1) if result is not None else None
    # Some original algorithms return all-False as a failure SENTINEL.
    # Only a valid decision may actually reject matches.
    effective = np.ones(len(pairs), dtype=bool) if structural_failure else raw_mask
    diag = pose_diagnostics(modules["common"], case, effective, max_condition)
    row = {
        "prepared_case_id": str(case["case_id"].item()),
        "original_case_id": str(case["original_case_id"].item()),
        "scenario": str(case["scenario"].item()),
        "source_frame": int(case["source_frame"]),
        "target_frame": int(case["target_frame"]),
        "method": method, "n_assigned": len(pairs),
        "decision_available": not structural_failure,
        "fallback_used": structural_failure,
        "algorithm_success_flag": bool(result.get("success", False)) if result is not None else False,
        "failure_reason": reason,
        "exception": error,
        "raw_kept": int(raw_mask.sum()) if raw_mask is not None else np.nan,
        "raw_rejected": int((~raw_mask).sum()) if raw_mask is not None else np.nan,
        "effective_kept": int(effective.sum()),
        "effective_rejected": int((~effective).sum()),
        "runtime_median_ms": float(np.median(runtimes)) if runtimes else np.nan,
        "runtime_p95_ms": float(np.percentile(runtimes, 95)) if runtimes else np.nan,
        "timed_runs": len(runtimes),
        **diag,
    }
    details = []
    for k, (source_idx, target_idx) in enumerate(pairs):
        details.append({
            "prepared_case_id": row["prepared_case_id"], "original_case_id": row["original_case_id"],
            "scenario": row["scenario"], "source_frame": row["source_frame"],
            "target_frame": row["target_frame"], "method": method, "pair_index": k,
            "source_plane_id": int(case["source_plane_ids"][source_idx]),
            "target_plane_id": int(case["target_plane_ids"][target_idx]),
            "decision_available": not structural_failure, "fallback_used": structural_failure,
            "raw_keep": bool(raw_mask[k]) if raw_mask is not None else None,
            "effective_keep": bool(effective[k]),
        })
    return row, details


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=root / "data" / "bad_match_rejection" / "real")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "results" / "bad_match_rejection" / "real")
    parser.add_argument("--algorithm-dir", type=Path,
                        help="Directory containing the FIVE original algorithm modules and common.py")
    parser.add_argument("--parameter-file", type=Path,
                        help="Frozen Synthetic DEV best_parameters.json")
    parser.add_argument("--use-module-defaults", action="store_true",
                        help="Explicit smoke-test fallback ONLY, not a frozen evaluation")
    parser.add_argument("--scenarios", default="baseline,controlled_corruption,exploratory")
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--rng-seed", type=int, default=20261105)
    parser.add_argument("--max-pose-condition", type=float, default=1e6)
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be >=1")
    modules = setup_algorithm_imports(args.algorithm_dir)
    params, param_source = load_parameters(modules, args.parameter_file, args.use_module_defaults)
    modules["Normal"].NORMAL_THRESHOLD_DEG = float(params["Normal"]["normal_threshold_deg"])
    modules["Distance"].OFFSET_THRESHOLD = float(params["Distance"]["offset_threshold_m"])
    modules["RANSAC"].NORMAL_THRESHOLD_DEG = float(params["RANSAC"]["normal_threshold_deg"])
    modules["RANSAC"].OFFSET_THRESHOLD = float(params["RANSAC"]["offset_threshold_m"])
    if "iterations" in params["RANSAC"]:
        modules["RANSAC"].RANSAC_ITERATIONS = int(params["RANSAC"]["iterations"])
    tri = modules["Trilateration-inspired"]
    tri.CLUSTER_RADIUS = float(params["Trilateration-inspired"]["cluster_radius_m"])
    tri.MIN_SUPPORT = float(params["Trilateration-inspired"]["min_support"])
    if "max_subsets" in params["Trilateration-inspired"]:
        tri.MAX_SUBSETS = int(params["Trilateration-inspired"]["max_subsets"])

    data_dir = args.input_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Prepared input directory not found: {data_dir}")
    allowed = {x.strip() for x in args.scenarios.split(",") if x.strip()}
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No prepared .npz cases in {data_dir}; run prepare_real_bmr.py")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_rows, detail_rows = [], []
    selected = 0
    for index, file in enumerate(files):
        case = load_case(file)
        scenario = str(case["scenario"].item())
        if scenario not in allowed:
            continue
        selected += 1
        for method in METHODS:
            row, detail = run_case_method(
                case, method, modules, params, seed=args.rng_seed + index,
                repeats=args.timing_repeats, max_condition=args.max_pose_condition,
            )
            case_rows.append(row)
            detail_rows.extend(detail)
        print(f"[{selected}] {case['case_id'].item()}: "
              f"{len(case['assigned_pairs'])} matches, {len(METHODS)} methods")
    if not selected:
        raise ValueError(f"No cases matching --scenarios {sorted(allowed)}")
    pd.DataFrame(case_rows).to_csv(output / "case_results.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output / "match_decisions.csv", index=False)
    with (output / "parameters_used.json").open("w", encoding="utf-8") as fp:
        json.dump({
            "parameter_source": param_source,
            "parameters": params,
            "algorithm_files": {k: str(Path(m.__file__).resolve()) for k, m in modules.items()},
            "prepared_data_dir": str(data_dir),
            "max_pose_condition": args.max_pose_condition,
            "timing_repeats": args.timing_repeats,
            "rng_seed": args.rng_seed,
            "note": "No GT labels or GT poses loaded; fallback keeps all on structural failure.",
        }, fp, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(case_rows)} method-case results and "
          f"{len(detail_rows)} decisions to {output}")
    print("GT not accessed. Execute analyse_bad_match_rejection_real.py to score results.")


if __name__ == "__main__":
    main()
