"""
assignment_common.py

Common data structures and utilities for the one-to-one plane
correspondence assignment stage.

The Assignment stage receives candidate plane correspondences and their
pairwise similarity scores from the previous Candidate Gating and
Similarity stages, and selects a globally consistent one-to-one set of
plane correspondences.

All assignment algorithms use the same AssignmentCase representation
to ensure a fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np


# ============================================================
# Data structures
# ============================================================

@dataclass
class AssignmentCase:
    """
    One complete assignment problem for a pair of frames.

    Parameters
    ----------
    case_id:
        Unique identifier of the frame pair / synthetic case.

    current_ids:
        Plane identifiers in the current frame.

    reference_ids:
        Plane identifiers in the reference frame.

    similarity:
        Matrix with shape (N_current, N_reference).

        similarity[i, j] is the pairwise similarity score between
        current plane i and reference plane j.

        Expected range:
            0 <= similarity <= 1

    candidate_mask:
        Boolean matrix with the same shape as similarity.

        True:
            candidate correspondence retained by Candidate Gating.

        False:
            correspondence is not allowed to enter Assignment.

    gt_matches:
        Ground-truth correspondences represented by index pairs:

            {(current_index, reference_index), ...}

        Example:
            {(0, 2), (1, 0), (3, 4)}

    metadata:
        Optional information, e.g.
        sequence, frame IDs, synthetic scenario, noise level, etc.
    """

    case_id: str

    current_ids: list[str]
    reference_ids: list[str]

    similarity: np.ndarray
    candidate_mask: np.ndarray

    gt_matches: set[tuple[int, int]] = field(default_factory=set)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_current(self) -> int:
        return len(self.current_ids)

    @property
    def n_reference(self) -> int:
        return len(self.reference_ids)


@dataclass
class AssignmentResult:
    """
    Standardised output returned by every assignment algorithm.
    """

    algorithm: str
    case_id: str

    matches: list[tuple[int, int]]

    unmatched_current: list[int]
    unmatched_reference: list[int]

    runtime_ms: float

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Validation
# ============================================================

def validate_assignment_case(case: AssignmentCase) -> None:
    """
    Validate the consistency of an AssignmentCase.

    Raises
    ------
    ValueError
        If the case is internally inconsistent.
    """

    similarity = np.asarray(case.similarity)
    mask = np.asarray(case.candidate_mask)

    expected_shape = (case.n_current, case.n_reference)

    if similarity.shape != expected_shape:
        raise ValueError(
            f"{case.case_id}: similarity shape {similarity.shape} "
            f"does not match expected {expected_shape}"
        )

    if mask.shape != expected_shape:
        raise ValueError(
            f"{case.case_id}: candidate_mask shape {mask.shape} "
            f"does not match expected {expected_shape}"
        )

    if mask.dtype != bool:
        raise ValueError(
            f"{case.case_id}: candidate_mask must have boolean dtype"
        )

    if not np.all(np.isfinite(similarity)):
        raise ValueError(
            f"{case.case_id}: similarity matrix contains NaN or Inf"
        )

    if np.any(similarity < 0.0) or np.any(similarity > 1.0):
        raise ValueError(
            f"{case.case_id}: similarity scores must lie in [0, 1]"
        )

    for i, j in case.gt_matches:
        if not (0 <= i < case.n_current):
            raise ValueError(
                f"{case.case_id}: invalid GT current-plane index {i}"
            )

        if not (0 <= j < case.n_reference):
            raise ValueError(
                f"{case.case_id}: invalid GT reference-plane index {j}"
            )

    # Ground truth itself must also be one-to-one
    gt_current = [i for i, _ in case.gt_matches]
    gt_reference = [j for _, j in case.gt_matches]

    if len(gt_current) != len(set(gt_current)):
        raise ValueError(
            f"{case.case_id}: one current plane has multiple GT matches"
        )

    if len(gt_reference) != len(set(gt_reference)):
        raise ValueError(
            f"{case.case_id}: one reference plane has multiple GT matches"
        )


# ============================================================
# Similarity / cost conversion
# ============================================================

def similarity_to_cost(
    similarity: np.ndarray,
) -> np.ndarray:
    """
    Convert similarity scores into assignment costs.

    The Assignment experiments use

        c_ij = 1 - s_ij

    where higher similarity therefore produces lower cost.
    """

    similarity = np.asarray(similarity, dtype=np.float64)

    return 1.0 - similarity


# ============================================================
# Augmented assignment matrix
# ============================================================

def build_augmented_cost_matrix(
    case: AssignmentCase,
    unmatched_cost: float = 0.50,
    invalid_cost: float = 1e6,
) -> np.ndarray:
    """
    Construct a square cost matrix supporting:

    - unequal numbers of planes;
    - unmatched current planes;
    - unmatched reference planes;
    - Candidate Gating restrictions.

    Matrix structure
    ----------------

                   Reference        Dummy-current
                 real columns         columns

    Current      [ real costs      unmatched current ]
    real rows

    Dummy        [ unmatched ref      dummy-dummy     ]
    reference
    rows


    The final matrix has size

        (N_current + N_reference)
            x
        (N_reference + N_current)

    which is always square.

    Parameters
    ----------
    unmatched_cost:
        Cost assigned when a plane is deliberately left unmatched.

    invalid_cost:
        Large cost applied to plane pairs rejected by Candidate Gating.

    Returns
    -------
    cost:
        Augmented square assignment matrix.
    """

    validate_assignment_case(case)

    n_c = case.n_current
    n_r = case.n_reference

    size = n_c + n_r

    cost = np.full(
        (size, size),
        invalid_cost,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Top-left:
    # real current planes -> real reference planes
    # --------------------------------------------------------

    real_cost = similarity_to_cost(case.similarity)

    real_cost = np.where(
        case.candidate_mask,
        real_cost,
        invalid_cost,
    )

    cost[:n_c, :n_r] = real_cost

    # --------------------------------------------------------
    # Top-right:
    # current plane -> its own dummy column
    #
    # This represents an unmatched current plane.
    # --------------------------------------------------------

    for i in range(n_c):
        dummy_col = n_r + i
        cost[i, dummy_col] = unmatched_cost

    # --------------------------------------------------------
    # Bottom-left:
    # dummy row -> reference plane
    #
    # This represents an unmatched reference plane.
    # --------------------------------------------------------

    for j in range(n_r):
        dummy_row = n_c + j
        cost[dummy_row, j] = unmatched_cost

    # --------------------------------------------------------
    # Bottom-right:
    # dummy -> dummy
    #
    # No physical correspondence is represented here.
    # Zero cost allows completion of the perfect assignment.
    # --------------------------------------------------------

    cost[n_c:, n_r:] = 0.0

    return cost


# ============================================================
# Decode solver output
# ============================================================

def decode_augmented_assignment(
    case: AssignmentCase,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    list[int],
    list[int],
]:
    """
    Decode the solution of an augmented assignment matrix.

    Parameters
    ----------
    row_indices, col_indices:
        Solver output describing selected matrix entries.

    Returns
    -------
    matches:
        Real current -> real reference correspondences.

    unmatched_current:
        Current-plane indices that remain unmatched.

    unmatched_reference:
        Reference-plane indices that remain unmatched.
    """

    n_c = case.n_current
    n_r = case.n_reference

    matches: list[tuple[int, int]] = []

    matched_current: set[int] = set()
    matched_reference: set[int] = set()

    for row, col in zip(row_indices, col_indices):

        # Only top-left entries correspond to real matches.
        if row < n_c and col < n_r:

            # Safety check:
            # rejected candidate pairs must never be returned.
            if not case.candidate_mask[row, col]:
                continue

            matches.append((int(row), int(col)))

            matched_current.add(int(row))
            matched_reference.add(int(col))

    unmatched_current = [
        i
        for i in range(n_c)
        if i not in matched_current
    ]

    unmatched_reference = [
        j
        for j in range(n_r)
        if j not in matched_reference
    ]

    matches.sort()
    unmatched_current.sort()
    unmatched_reference.sort()

    return (
        matches,
        unmatched_current,
        unmatched_reference,
    )


# ============================================================
# Ground-truth helpers
# ============================================================

def get_gt_unmatched_current(
    case: AssignmentCase,
) -> set[int]:
    """
    Return current planes with no GT correspondence.
    """

    matched = {
        i
        for i, _ in case.gt_matches
    }

    return set(range(case.n_current)) - matched


def get_gt_unmatched_reference(
    case: AssignmentCase,
) -> set[int]:
    """
    Return reference planes with no GT correspondence.
    """

    matched = {
        j
        for _, j in case.gt_matches
    }

    return set(range(case.n_reference)) - matched


# ============================================================
# Case persistence
# ============================================================

def save_assignment_case(
    case: AssignmentCase,
    path: str | Path,
) -> None:
    """
    Save one AssignmentCase as a compressed NPZ file.
    """

    validate_assignment_case(case)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gt_array = np.asarray(
        sorted(case.gt_matches),
        dtype=np.int64,
    )

    if gt_array.size == 0:
        gt_array = np.empty((0, 2), dtype=np.int64)

    np.savez_compressed(
        path,
        case_id=np.asarray(case.case_id),
        current_ids=np.asarray(case.current_ids, dtype=str),
        reference_ids=np.asarray(case.reference_ids, dtype=str),
        similarity=np.asarray(
            case.similarity,
            dtype=np.float64,
        ),
        candidate_mask=np.asarray(
            case.candidate_mask,
            dtype=bool,
        ),
        gt_matches=gt_array,
        metadata=np.asarray(
            json.dumps(case.metadata),
            dtype=str,
        ),
    )


def load_assignment_case(
    path: str | Path,
) -> AssignmentCase:
    """
    Load an AssignmentCase saved by save_assignment_case().
    """

    path = Path(path)

    with np.load(path, allow_pickle=False) as data:

        gt_array = np.asarray(
            data["gt_matches"],
            dtype=np.int64,
        )

        gt_matches = {
            (int(i), int(j))
            for i, j in gt_array
        }

        metadata = json.loads(
            str(data["metadata"].item())
        )

        case = AssignmentCase(
            case_id=str(data["case_id"].item()),
            current_ids=data["current_ids"].astype(str).tolist(),
            reference_ids=data["reference_ids"].astype(str).tolist(),
            similarity=np.asarray(
                data["similarity"],
                dtype=np.float64,
            ),
            candidate_mask=np.asarray(
                data["candidate_mask"],
                dtype=bool,
            ),
            gt_matches=gt_matches,
            metadata=metadata,
        )

    validate_assignment_case(case)

    return case


# ============================================================
# Basic case summary
# ============================================================

def summarise_assignment_case(
    case: AssignmentCase,
) -> dict[str, Any]:
    """
    Produce a compact summary useful for debugging and experiment logs.
    """

    validate_assignment_case(case)

    n_candidates = int(
        np.count_nonzero(case.candidate_mask)
    )

    n_possible = (
        case.n_current
        * case.n_reference
    )

    gt_unmatched_current = get_gt_unmatched_current(case)
    gt_unmatched_reference = get_gt_unmatched_reference(case)

    return {
        "case_id": case.case_id,
        "n_current": case.n_current,
        "n_reference": case.n_reference,
        "n_candidates": n_candidates,
        "n_possible_pairs": n_possible,
        "candidate_density": (
            n_candidates / n_possible
            if n_possible > 0
            else 0.0
        ),
        "n_gt_matches": len(case.gt_matches),
        "n_gt_unmatched_current": len(
            gt_unmatched_current
        ),
        "n_gt_unmatched_reference": len(
            gt_unmatched_reference
        ),
        **case.metadata,
    }