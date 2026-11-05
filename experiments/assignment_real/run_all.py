"""One-command real-data Assignment comparison using the USER'S four original solvers.

From this directory:
    python run_all.py

No experiment is run by package creation; running this file performs the experiment.
The supplied CSVs contain no GT labels; no real-data accuracy/F1 is invented.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_ALGORITHMS = (
    'hungarian', 'auction', 'jonker_volgenant', 'cost_scaling'
)
IMPORTS = {
    'hungarian': ('hungarian_assignment', 'run_hungarian_assignment'),
    'auction': ('auction_assignment', 'run_auction_assignment'),
    'jonker_volgenant': ('jonker_volgenant_assignment', 'run_jonker_volgenant_assignment'),
    'cost_scaling': ('cost_scaling_assignment', 'run_cost_scaling_assignment'),
}
REQUIRED_COLUMNS = {
    'source_frame', 'target_frame', 'gating_algorithm', 'similarity_algorithm',
    'source_plane_id', 'target_plane_id', 'score', 'higher_is_better', 'gt_match',
}
PROVISIONAL = {
    'shared_unmatched_cost': 0.25,
    'auction_epsilon': 0.001,
    'invalid_cost': 1000000.0,
    'cost_scaling_cost_scale': 1000000,
    'cost_scaling_epsilon_factor': 4.0,
}


def find_algorithm_dir(user_path: str | None) -> Path:
    if user_path:
        requested = Path(user_path).expanduser().resolve()
        paths = [requested, requested / 'algorithms']
    else:
        paths = [
            ROOT / 'algorithms',
            ROOT / 'uploaded_programs' / 'algorithms',
            ROOT.parent / 'assignment' / 'algorithms',
            ROOT.parent / 'experiments' / 'assignment' / 'algorithms',
        ]
        for ancestor in (ROOT, *ROOT.parents):
            paths.extend([
                ancestor / 'experiments' / 'assignment' / 'algorithms',
                ancestor / 'assignment' / 'algorithms',
            ])
    for path in paths:
        if all((path / (module + '.py')).is_file() for module, _ in IMPORTS.values()):
            return path.resolve()
    expected = '\n'.join(f'  - {module}.py' for module, _ in IMPORTS.values())
    raise FileNotFoundError(
        'Could not locate the ORIGINAL four Assignment algorithm modules.\n'
        'They were not included in the files you uploaded, so this package deliberately\n'
        'does not substitute other algorithms.\n'
        f'Expected inside an algorithms/ directory:\n{expected}\n'
        'Put that directory next to run_all.py, or specify:\n'
        '  python run_all.py --algorithms-dir "D:\\PlaneBasedSLAM\\experiments\\assignment\\algorithms"'
    )


def load_solvers(directory: Path):
    # Make original algorithms import their expected assignment_common module.
    # Our root includes an unchanged copy of the user's assignment_common.py.
    sys.path.insert(0, str(directory.parent))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    runners = {}
    for name, (module_name, function_name) in IMPORTS.items():
        module = importlib.import_module('algorithms.' + module_name)
        runners[name] = getattr(module, function_name)
    from assignment_common import AssignmentCase, save_assignment_case
    return runners, AssignmentCase, save_assignment_case


def find_parameters(user_path: str | None):
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Frozen parameter file does not exist: {path}')
        return path
    choices = [
        ROOT / 'frozen_parameters.json',
        ROOT / 'results' / 'assignment' / 'tuning' / 'frozen_parameters.json',
        ROOT.parent / 'results' / 'assignment' / 'tuning' / 'frozen_parameters.json',
    ]
    for parent in ROOT.parents:
        choices.append(parent / 'results' / 'assignment' / 'tuning' / 'frozen_parameters.json')
    return next((p for p in choices if p.is_file()), None)


def load_parameters(user_path: str | None, require_frozen: bool):
    path = find_parameters(user_path)
    if path:
        values = json.loads(path.read_text(encoding='utf-8-sig'))
        for key in ('shared_unmatched_cost', 'auction_epsilon', 'invalid_cost'):
            if key not in values:
                raise KeyError(f'{path} missing required key {key!r}')
        merged = {**PROVISIONAL, **values}
        source = str(path)
        is_frozen = True
    else:
        if require_frozen:
            raise FileNotFoundError(
                'No frozen_parameters.json found. Pass --parameters PATH, or omit\n'
                '--require-frozen for an explicitly PROVISIONAL run using documented defaults.'
            )
        merged = dict(PROVISIONAL)
        source = 'PROVISIONAL PACKAGE DEFAULTS (NOT SYNTHETIC-DEV TUNED)'
        is_frozen = False
        print('WARNING: frozen_parameters.json not found. Using PROVISIONAL parameters.')
        print('WARNING: Do not present this run as the frozen-parameter FINAL comparison.')
    return merged, source, is_frozen


def read_rows(folder: Path) -> tuple[dict, list[dict]]:
    groups = defaultdict(list)
    counts = {}
    files = sorted(folder.glob('assignment_input_*.csv'))
    if not files:
        raise FileNotFoundError(f'No assignment_input_*.csv found in {folder}')
    for path in files:
        with path.open('r', encoding='utf-8-sig', newline='') as fp:
            reader = csv.DictReader(fp)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
            rows = list(reader)
        counts[path.name] = len(rows)
        for row in rows:
            key = (row['gating_algorithm'], row['similarity_algorithm'],
                   row['source_frame'], row['target_frame'])
            groups[key].append(row)
    return counts, groups


def sort_plane_ids(values):
    def key(value):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)
    return sorted(set(values), key=key)


def make_case(case_id, key, rows, AssignmentCase):
    gating, method, source_frame, target_frame = key
    source_ids = sort_plane_ids(r['source_plane_id'] for r in rows)
    target_ids = sort_plane_ids(r['target_plane_id'] for r in rows)
    source_map = {v: i for i, v in enumerate(source_ids)}
    target_map = {v: j for j, v in enumerate(target_ids)}
    sim = np.zeros((len(source_ids), len(target_ids)), dtype=np.float64)
    mask = np.zeros(sim.shape, dtype=bool)
    gt_labels = np.full(sim.shape, -1, dtype=np.int8)
    gt_matches = set()
    for row in rows:
        i, j = source_map[row['source_plane_id']], target_map[row['target_plane_id']]
        if mask[i, j]:
            raise ValueError(f'Duplicate pair in {key}: {row["source_plane_id"]}, {row["target_plane_id"]}')
        score = float(row['score'])
        if not math.isfinite(score) or not (0 <= score <= 1):
            raise ValueError(f'Invalid score: {score} in {key}')
        direction = row['higher_is_better'].strip().lower()
        if direction not in {'true', 'false', '1', '0'}:
            raise ValueError(f'Unrecognised higher_is_better={direction!r}')
        if direction in {'false', '0'}:
            score = 1.0 - score
        mask[i, j] = True
        sim[i, j] = score
        label = row['gt_match'].strip().lower()
        if label in {'', '-1', 'unknown', 'nan', 'none'}:
            gt_labels[i, j] = -1
        elif label in {'1', 'true'}:
            gt_labels[i, j] = 1
            gt_matches.add((i, j))
        elif label in {'0', 'false'}:
            gt_labels[i, j] = 0
        else:
            raise ValueError(f'Unrecognised GT label {label!r}')
    # Unknown GT pairs remain unscored. No GT information is used by solvers.
    meta = {
        'gating_algorithm': gating, 'similarity_algorithm': method,
        'source_frame': source_frame, 'target_frame': target_frame,
        'ground_truth_coverage': 'incomplete; labels of -1 are unknown',
    }
    case = AssignmentCase(case_id=case_id, current_ids=source_ids,
                          reference_ids=target_ids, similarity=sim,
                          candidate_mask=mask, gt_matches=gt_matches, metadata=meta)
    return case, gt_labels


def call_solver(algorithm, runner, case, params):
    unmatched = float(params['shared_unmatched_cost'])
    invalid = float(params['invalid_cost'])
    if algorithm == 'hungarian':
        return runner(case=case, unmatched_cost=unmatched, invalid_cost=invalid)
    if algorithm == 'auction':
        return runner(case=case, unmatched_cost=unmatched,
                      epsilon=float(params['auction_epsilon']), invalid_cost=invalid)
    if algorithm == 'jonker_volgenant':
        return runner(case=case, unmatched_cost=unmatched, invalid_cost=invalid)
    if algorithm == 'cost_scaling':
        return runner(case=case, unmatched_cost=unmatched, invalid_cost=invalid,
                      cost_scale=int(params['cost_scaling_cost_scale']),
                      epsilon_factor=float(params['cost_scaling_epsilon_factor']))
    raise KeyError(algorithm)


def validate_result(case, result):
    from assignment_evaluation import validate_assignment_result
    validate_assignment_result(case, result)
    # Defensive check: no exact duplicate returned match edges.
    if len(result.matches) != len(set(result.matches)):
        raise ValueError(f'{case.case_id}: returned duplicate matches')


def objective(case, result, unmatched):
    return (
        sum(1.0 - float(case.similarity[i, j]) for i, j in result.matches)
        + unmatched * (len(result.unmatched_current) + len(result.unmatched_reference))
    )


def save_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else ['status']
    with path.open('w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def make_results(args):
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else ROOT / 'input_csv'
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / 'results'
    counts, groups = read_rows(input_dir)
    algorithm_dir = find_algorithm_dir(args.algorithms_dir)
    runners, AssignmentCase, save_assignment_case = load_solvers(algorithm_dir)
    params, param_source, param_is_frozen = load_parameters(args.parameters, args.require_frozen)
    if args.repeats < 1:
        raise ValueError('--repeats must be >=1')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases').mkdir(exist_ok=True)
    case_rows = []
    matches = []
    unmatched = []
    case_manifest = []
    cost_comparison = []
    n_cases = len(groups)
    print(f'Loaded {sum(counts.values())} candidate records in {n_cases} cases')
    print(f'Using original algorithms from: {algorithm_dir}')
    print(f'Parameters: {param_source}')
    print(f'Running four solvers x {n_cases} cases (repeats={args.repeats})')
    for case_index, (key, pair_rows) in enumerate(sorted(groups.items())):
        case_id = f'case_{case_index:04d}'
        case, gt_labels = make_case(case_id, key, pair_rows, AssignmentCase)
        save_assignment_case(case, out / 'cases' / f'{case_id}.npz')
        gating, method, source_frame, target_frame = key
        base = {
            'case_id': case_id, 'gating_algorithm': gating,
            'similarity_algorithm': method, 'source_frame': source_frame,
            'target_frame': target_frame,
        }
        manifest = dict(base, n_source_planes=case.n_current, n_target_planes=case.n_reference,
                        candidate_pairs=int(np.count_nonzero(case.candidate_mask)),
                        known_gt_positive=int(np.count_nonzero(case.candidate_mask & (gt_labels == 1))),
                        known_gt_negative=int(np.count_nonzero(case.candidate_mask & (gt_labels == 0))),
                        unknown_gt=int(np.count_nonzero(case.candidate_mask & (gt_labels == -1))))
        case_manifest.append(manifest)
        by_alg = {}
        for algorithm in DEFAULT_ALGORITHMS:
            runner = runners[algorithm]
            runtime_samples = []
            result = None
            for repeat in range(args.repeats):
                trial = call_solver(algorithm, runner, case, params)
                validate_result(case, trial)
                if result is not None and set(trial.matches) != set(result.matches):
                    raise RuntimeError(f'{algorithm}/{case_id}: different matches across timing repeats')
                result = trial
                runtime_samples.append(float(trial.runtime_ms))
            assert result is not None
            chosen = set(result.matches)
            tp = sum(gt_labels[i, j] == 1 for i, j in chosen)
            fp = sum(gt_labels[i, j] == 0 for i, j in chosen)
            unknown = sum(gt_labels[i, j] == -1 for i, j in chosen)
            known_positives = set(case.gt_matches)
            fn = len(known_positives - chosen)
            has_gt = bool(np.count_nonzero(case.candidate_mask & (gt_labels != -1)))
            # An all-unknown reference is not zero accuracy; it is unmeasurable.
            precision = tp / (tp + fp) if has_gt and tp + fp > 0 else None
            recall = tp / (tp + fn) if has_gt and tp + fn > 0 else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall > 0
                  else None)
            c = objective(case, result, float(params['shared_unmatched_cost']))
            by_alg[algorithm] = (c, chosen)
            case_rows.append({**base, 'algorithm': algorithm,
                              'n_source_planes': case.n_current,
                              'n_target_planes': case.n_reference,
                              'candidate_pairs': manifest['candidate_pairs'],
                              'selected_matches': len(chosen),
                              'unmatched_source': len(result.unmatched_current),
                              'unmatched_target': len(result.unmatched_reference),
                              'tp_known': tp if has_gt else None,
                              'fp_known': fp if has_gt else None,
                              'fn_known_reachable': fn if has_gt else None,
                              'selected_unknown': int(unknown),
                              'precision_known': precision,
                              'recall_reachable_known': recall,
                              'f1_known': f1,
                              'runtime_mean_ms': float(np.mean(runtime_samples)),
                              'runtime_median_ms': float(np.median(runtime_samples)),
                              'runtime_std_ms': float(np.std(runtime_samples)),
                              'total_augmented_cost': c,
                              'parameter_status': 'FROZEN' if param_is_frozen else 'PROVISIONAL'})
            for i, j in sorted(chosen):
                matches.append({**base, 'algorithm': algorithm,
                                'source_plane_id': case.current_ids[i],
                                'target_plane_id': case.reference_ids[j],
                                'similarity_score': float(case.similarity[i, j]),
                                'gt_label': int(gt_labels[i, j])})
            for i in result.unmatched_current:
                unmatched.append({**base, 'algorithm': algorithm, 'side': 'source',
                                  'plane_id': case.current_ids[i]})
            for j in result.unmatched_reference:
                unmatched.append({**base, 'algorithm': algorithm, 'side': 'target',
                                  'plane_id': case.reference_ids[j]})
        reference_cost, reference_matches = by_alg['hungarian']
        for algorithm in DEFAULT_ALGORITHMS[1:]:
            candidate_cost, candidate_matches = by_alg[algorithm]
            cost_comparison.append({**base, 'reference_algorithm': 'hungarian',
                                    'compared_algorithm': algorithm,
                                    'reference_cost': reference_cost,
                                    'compared_cost': candidate_cost,
                                    'absolute_cost_gap': abs(candidate_cost - reference_cost),
                                    'same_match_set': int(candidate_matches == reference_matches)})
        print(f'  [{case_index + 1:02d}/{n_cases:02d}] {gating} / {method} / {source_frame}->{target_frame}')
    summaries = []
    for algorithm in DEFAULT_ALGORITHMS:
        group = [r for r in case_rows if r['algorithm'] == algorithm]
        summaries.append({
            'algorithm': algorithm, 'n_cases': len(group),
            'total_selected_matches': sum(r['selected_matches'] for r in group),
            'total_selected_unknown': sum(r['selected_unknown'] for r in group),
            'mean_runtime_ms': statistics.mean(r['runtime_mean_ms'] for r in group),
            'median_runtime_ms': statistics.median(r['runtime_median_ms'] for r in group),
            'mean_total_augmented_cost': statistics.mean(r['total_augmented_cost'] for r in group),
            'accuracy_note': 'UNAVAILABLE - no independent GT' if all(r['precision_known'] is None for r in group)
                             else 'See per_case_results.csv; partial evaluable GT only',
            'parameter_status': 'FROZEN' if param_is_frozen else 'PROVISIONAL',
        })
    save_csv(out / 'case_manifest.csv', case_manifest)
    save_csv(out / 'per_case_results.csv', case_rows)
    save_csv(out / 'assignment_matches.csv', matches)
    save_csv(out / 'assignment_unmatched.csv', unmatched)
    save_csv(out / 'solver_agreement.csv', cost_comparison)
    save_csv(out / 'summary.csv', summaries)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'algorithms': list(DEFAULT_ALGORITHMS),
        'algorithm_implementation_directory': str(algorithm_dir),
        'parameters': params, 'parameter_source': param_source,
        'parameters_frozen': param_is_frozen,
        'input_csv_record_counts': counts, 'n_cases': n_cases,
        'n_solver_case_evaluations': len(case_rows), 'timing_repeats': args.repeats,
        'gt_policy': 'Missing GT is unknown (-1), never treated as a false positive.',
        'evaluation_caveat': 'Input CSV plane sets include only planes appearing in candidate rows; they may omit planes with zero candidates.',
        'score_caveat': 'Higher is better is taken from CSV. Similarity algorithms may not have calibrated scores; shared unmatched penalty is not guaranteed comparable across methods.',
    }
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Finished {len(case_rows)} solver-case evaluations. Results: {out}')
    if all(r['precision_known'] is None for r in case_rows):
        print('NOTE: No independent GT labels provided. Accuracy/Recall/F1 are intentionally blank.')


def main():
    parser = argparse.ArgumentParser(description='Run original 4 Assignment solvers on all real CSV inputs using ONE command.')
    parser.add_argument('--algorithms-dir', help='Path to your ORIGINAL algorithms/ folder (auto-detected if omitted)')
    parser.add_argument('--parameters', help='Path to your Synthetic-DEV frozen_parameters.json (auto-detected if omitted)')
    parser.add_argument('--require-frozen', action='store_true', help='Reject runs without actual frozen parameters')
    parser.add_argument('--input-dir', help='Folder containing four assignment_input_*.csv')
    parser.add_argument('--output-dir', help='Results output folder (default: results/)')
    parser.add_argument('--repeats', type=int, default=10, help='Timing repeats per solver/case, default: 10')
    args = parser.parse_args()
    make_results(args)


if __name__ == '__main__':
    main()
