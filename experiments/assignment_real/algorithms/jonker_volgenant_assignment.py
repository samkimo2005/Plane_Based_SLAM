"""
jonker_volgenant_assignment.py

Jonker-Volgenant (JV) assignment method for one-to-one plane
correspondence Assignment.

The implementation uses scipy.optimize.linear_sum_assignment(),
which implements a modified Jonker-Volgenant algorithm.

The same augmented cost matrix is used as for the other Assignment
methods, allowing:

- one-to-one plane correspondences;
- unmatched planes;
- unequal current/reference plane-set sizes;
- Candidate Gating restrictions.

Only the optimisation stage is included in the measured runtime.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "SciPy is required for the Jonker-Volgenant implementation. "
        "Install it using: pip install scipy"
    ) from exc

from assignment_common import (
    AssignmentCase,
    AssignmentResult,
    build_augmented_cost_matrix,
    decode_augmented_assignment,
    validate_assignment_case,
)


# ============================================================
# Core JV solver
# ============================================================

def _jonker_volgenant_square(
    cost_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a square linear assignment problem using SciPy's
    modified Jonker-Volgenant algorithm.

    Parameters
    ----------
    cost_matrix:
        Square finite cost matrix of shape (N, N).

    Returns
    -------
    row_indices:
        Selected row indices.

    col_indices:
        Corresponding selected column indices.
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
            "JV implementation requires a square matrix"
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

    row_indices, col_indices = (
        linear_sum_assignment(cost)
    )

    return (
        np.asarray(
            row_indices,
            dtype=np.int64,
        ),
        np.asarray(
            col_indices,
            dtype=np.int64,
        ),
    )


# ============================================================
# Public Assignment interface
# ============================================================

def run_jonker_volgenant_assignment(
    case: AssignmentCase,
    unmatched_cost: float = 0.50,
    invalid_cost: float = 1e6,
) -> AssignmentResult:
    """
    Run Jonker-Volgenant Assignment on one AssignmentCase.

    Parameters
    ----------
    case:
        Complete frame-pair Assignment problem.

    unmatched_cost:
        Cost of deliberately leaving one plane unmatched.

        This parameter will later be tuned using Synthetic DEV and
        frozen before Synthetic FINAL evaluation.

    invalid_cost:
        Large cost assigned to correspondences rejected by
        Candidate Gating.

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
    # JV optimisation
    # --------------------------------------------------------

    start = perf_counter()

    row_indices, col_indices = (
        _jonker_volgenant_square(
            cost_matrix
        )
    )

    end = perf_counter()

    runtime_ms = (
        end - start
    ) * 1000.0

    # --------------------------------------------------------
    # Common decoding
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
            1.0
            - case.similarity[i, j]
            for i, j in matches
        )
    )

    return AssignmentResult(
        algorithm="jonker_volgenant",
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

            "implementation":
                (
                    "scipy.optimize.linear_sum_assignment "
                    "(modified Jonker-Volgenant)"
                ),
        },
    )


# ============================================================
# Diagnostic helper
# ============================================================

def jonker_volgenant_diagnostics(
    case: AssignmentCase,
    result: AssignmentResult,
) -> dict[str, Any]:
    """
    Return a compact diagnostic summary.
    """

    selected_scores = [
        float(
            case.similarity[i, j]
        )
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
            len(
                result.unmatched_current
            ),

        "n_unmatched_reference":
            len(
                result.unmatched_reference
            ),

        "runtime_ms":
            float(result.runtime_ms),

        "mean_match_similarity":
            (
                float(
                    np.mean(
                        selected_scores
                    )
                )
                if selected_scores
                else 0.0
            ),

        **result.metadata,
    }