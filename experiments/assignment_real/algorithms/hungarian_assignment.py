"""
hungarian_assignment.py

Hungarian (Kuhn-Munkres) algorithm for one-to-one plane
correspondence assignment.

The algorithm operates on the common augmented cost matrix defined in
assignment_common.py, allowing:

- one-to-one plane correspondence;
- unequal current/reference plane-set sizes;
- explicitly unmatched planes;
- Candidate Gating restrictions.

The measured runtime includes only the assignment optimisation itself.
Common matrix construction and result decoding are excluded so that
runtime comparisons between Assignment algorithms remain fair.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from assignment_common import (
    AssignmentCase,
    AssignmentResult,
    build_augmented_cost_matrix,
    decode_augmented_assignment,
    validate_assignment_case,
)


# ============================================================
# Core Hungarian algorithm
# ============================================================

def _hungarian_square(
    cost_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a square linear assignment problem using the
    Hungarian / Kuhn-Munkres algorithm.

    Parameters
    ----------
    cost_matrix:
        Square finite cost matrix of shape (N, N).

    Returns
    -------
    row_indices:
        Row indices of the selected assignment.

    col_indices:
        Corresponding selected column indices.

    Notes
    -----
    This implementation uses the classical primal-dual Hungarian
    formulation with augmenting paths.

    Complexity:
        O(N^3)
    """

    cost = np.asarray(
        cost_matrix,
        dtype=np.float64,
    )

    if cost.ndim != 2:
        raise ValueError(
            "cost_matrix must be two-dimensional"
        )

    n_rows, n_cols = cost.shape

    if n_rows != n_cols:
        raise ValueError(
            "Hungarian implementation requires a square matrix"
        )

    if n_rows == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    if not np.all(np.isfinite(cost)):
        raise ValueError(
            "cost_matrix contains NaN or Inf"
        )

    n = n_rows

    # --------------------------------------------------------
    # 1-indexed arrays following the standard Hungarian
    # primal-dual formulation.
    #
    # u: row potentials
    # v: column potentials
    # p: current column-to-row matching
    # way: predecessor columns for augmenting paths
    # --------------------------------------------------------

    u = np.zeros(
        n + 1,
        dtype=np.float64,
    )

    v = np.zeros(
        n + 1,
        dtype=np.float64,
    )

    p = np.zeros(
        n + 1,
        dtype=np.int64,
    )

    way = np.zeros(
        n + 1,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Add each row to the matching
    # --------------------------------------------------------

    for i in range(1, n + 1):

        p[0] = i

        minv = np.full(
            n + 1,
            np.inf,
            dtype=np.float64,
        )

        used = np.zeros(
            n + 1,
            dtype=bool,
        )

        j0 = 0

        # ----------------------------------------------------
        # Search for an augmenting path
        # ----------------------------------------------------

        while True:

            used[j0] = True

            i0 = p[j0]

            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):

                if used[j]:
                    continue

                current_cost = (
                    cost[i0 - 1, j - 1]
                    - u[i0]
                    - v[j]
                )

                if current_cost < minv[j]:
                    minv[j] = current_cost
                    way[j] = j0

                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            if not np.isfinite(delta):
                raise RuntimeError(
                    "Hungarian algorithm could not find "
                    "an augmenting path"
                )

            # ------------------------------------------------
            # Update dual variables
            # ------------------------------------------------

            for j in range(n + 1):

                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta

                else:
                    minv[j] -= delta

            j0 = j1

            # Free column found
            if p[j0] == 0:
                break

        # ----------------------------------------------------
        # Augment the current matching
        # ----------------------------------------------------

        while True:

            j1 = way[j0]

            p[j0] = p[j1]

            j0 = j1

            if j0 == 0:
                break

    # --------------------------------------------------------
    # Convert column-to-row representation into conventional
    # row -> column assignment.
    # --------------------------------------------------------

    row_to_col = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for j in range(1, n + 1):

        i = p[j]

        if i != 0:
            row_to_col[i - 1] = j - 1

    if np.any(row_to_col < 0):
        raise RuntimeError(
            "Hungarian algorithm returned an incomplete assignment"
        )

    row_indices = np.arange(
        n,
        dtype=np.int64,
    )

    col_indices = row_to_col

    return (
        row_indices,
        col_indices,
    )


# ============================================================
# Public Assignment interface
# ============================================================

def run_hungarian_assignment(
    case: AssignmentCase,
    unmatched_cost: float = 0.50,
    invalid_cost: float = 1e6,
) -> AssignmentResult:
    """
    Run Hungarian Assignment on one AssignmentCase.

    Parameters
    ----------
    case:
        Complete frame-pair Assignment problem.

    unmatched_cost:
        Cost of deliberately leaving one plane unmatched.

        Since

            cost = 1 - similarity,

        unmatched_cost = 0.50 corresponds approximately to preferring
        a real match only when its similarity is sufficiently high
        relative to the unmatched alternative.

        This parameter should later be tuned using Synthetic DEV and
        frozen before Synthetic FINAL evaluation.

    invalid_cost:
        Large cost assigned to correspondences rejected by Candidate
        Gating.

    Returns
    -------
    AssignmentResult
        Standardised Assignment result.
    """

    validate_assignment_case(case)

    if unmatched_cost < 0:
        raise ValueError(
            "unmatched_cost must be non-negative"
        )

    if invalid_cost <= unmatched_cost:
        raise ValueError(
            "invalid_cost must be larger than unmatched_cost"
        )

    # --------------------------------------------------------
    # Common preprocessing
    #
    # Excluded from measured solver runtime.
    # --------------------------------------------------------

    cost_matrix = build_augmented_cost_matrix(
        case=case,
        unmatched_cost=unmatched_cost,
        invalid_cost=invalid_cost,
    )

    # --------------------------------------------------------
    # Hungarian optimisation
    # --------------------------------------------------------

    start = perf_counter()

    row_indices, col_indices = (
        _hungarian_square(
            cost_matrix
        )
    )

    end = perf_counter()

    runtime_ms = (
        end - start
    ) * 1000.0

    # --------------------------------------------------------
    # Common decoding
    #
    # Also excluded from solver runtime.
    # --------------------------------------------------------

    (
        matches,
        unmatched_current,
        unmatched_reference,
    ) = decode_augmented_assignment(
        case=case,
        row_indices=row_indices,
        col_indices=col_indices,
    )

    # --------------------------------------------------------
    # Cost statistics
    # --------------------------------------------------------

    total_augmented_cost = float(
        np.sum(
            cost_matrix[
                row_indices,
                col_indices,
            ]
        )
    )

    real_match_cost = float(
        sum(
            1.0 - case.similarity[i, j]
            for i, j in matches
        )
    )

    return AssignmentResult(
        algorithm="hungarian",
        case_id=case.case_id,
        matches=matches,
        unmatched_current=unmatched_current,
        unmatched_reference=unmatched_reference,
        runtime_ms=runtime_ms,
        metadata={
            "unmatched_cost":
                float(unmatched_cost),

            "invalid_cost":
                float(invalid_cost),

            "augmented_matrix_size":
                int(cost_matrix.shape[0]),

            "total_augmented_cost":
                total_augmented_cost,

            "real_match_cost":
                real_match_cost,
        },
    )


# ============================================================
# Optional diagnostic helper
# ============================================================

def hungarian_diagnostics(
    case: AssignmentCase,
    result: AssignmentResult,
) -> dict[str, Any]:
    """
    Generate a compact diagnostic summary.
    """

    selected_scores = [
        float(case.similarity[i, j])
        for i, j in result.matches
    ]

    return {
        "algorithm":
            result.algorithm,

        "case_id":
            case.case_id,

        "n_current":
            case.n_current,

        "n_reference":
            case.n_reference,

        "n_matches":
            len(result.matches),

        "n_unmatched_current":
            len(result.unmatched_current),

        "n_unmatched_reference":
            len(result.unmatched_reference),

        "runtime_ms":
            float(result.runtime_ms),

        "mean_match_similarity":
            (
                float(
                    np.mean(selected_scores)
                )
                if selected_scores
                else 0.0
            ),

        **result.metadata,
    }