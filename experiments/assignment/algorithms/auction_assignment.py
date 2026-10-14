"""
auction_assignment.py

Auction algorithm for one-to-one plane correspondence Assignment.

Reference:
    D. P. Bertsekas,
    "The auction algorithm: A distributed relaxation method
    for the assignment problem,"
    Annals of Operations Research, 1988.

The algorithm uses the same augmented assignment matrix as the other
Assignment methods so that:

- one-to-one correspondences are enforced;
- unmatched planes are supported;
- unequal plane-set sizes are supported;
- Candidate Gating restrictions are preserved.

Only the optimisation stage is included in the measured runtime.
"""

from __future__ import annotations

from collections import deque
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
# Core Auction algorithm
# ============================================================

def _auction_square(
    cost_matrix: np.ndarray,
    epsilon: float = 1e-4,
    max_iterations: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a square linear assignment problem using the Auction algorithm.

    Parameters
    ----------
    cost_matrix:
        Square cost matrix of shape (N, N).

    epsilon:
        Minimum bidding increment.

        Smaller epsilon generally improves optimality but may increase
        the number of iterations.

    max_iterations:
        Safety limit preventing non-terminating optimisation.

    Returns
    -------
    row_indices:
        Row indices of the final assignment.

    col_indices:
        Assigned column for each row.

    Notes
    -----
    Auction is naturally formulated as a maximisation problem.

    Therefore the assignment cost matrix is converted to values using

        value_ij = -cost_ij

    and the algorithm maximises the total value.
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
            "Auction implementation requires a square matrix"
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

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive"
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be positive"
        )

    n = n_rows

    # --------------------------------------------------------
    # Convert minimisation costs to maximisation values
    # --------------------------------------------------------

    values = -cost

    # Price associated with each reference / column
    prices = np.zeros(
        n,
        dtype=np.float64,
    )

    # assignment[i] = column currently assigned to row i
    assignment = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    # owner[j] = row currently owning column j
    owner = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    # Initially every row is unassigned
    unassigned = deque(
        range(n)
    )

    iterations = 0

    # ========================================================
    # Auction bidding process
    # ========================================================

    while unassigned:

        iterations += 1

        if iterations > max_iterations:
            raise RuntimeError(
                "Auction algorithm exceeded max_iterations. "
                "Consider increasing epsilon."
            )

        i = unassigned.popleft()

        # ----------------------------------------------------
        # Net value:
        #
        #     v_ij - price_j
        #
        # The row prefers the column with the largest net value.
        # ----------------------------------------------------

        net_values = (
            values[i]
            - prices
        )

        # Best column
        best_j = int(
            np.argmax(net_values)
        )

        best_value = float(
            net_values[best_j]
        )

        # ----------------------------------------------------
        # Second-best net value
        # ----------------------------------------------------

        if n == 1:

            second_best_value = (
                best_value
                - epsilon
            )

        else:

            # Avoid allocating a full sorted array.
            temp = net_values.copy()

            temp[best_j] = -np.inf

            second_best_value = float(
                np.max(temp)
            )

        # ----------------------------------------------------
        # Bid increment
        #
        # The winning row raises the price enough that the
        # chosen column is only epsilon better than its
        # second-best alternative.
        # ----------------------------------------------------

        bid_increment = (
            best_value
            - second_best_value
            + epsilon
        )

        prices[best_j] += (
            bid_increment
        )

        # Previous owner, if any
        previous_owner = int(
            owner[best_j]
        )

        # Assign best_j to row i
        owner[best_j] = i
        assignment[i] = best_j

        # ----------------------------------------------------
        # If another row previously owned this column,
        # it becomes unassigned and must bid again.
        # ----------------------------------------------------

        if previous_owner != -1:

            assignment[
                previous_owner
            ] = -1

            unassigned.append(
                previous_owner
            )

    # --------------------------------------------------------
    # Verify complete assignment
    # --------------------------------------------------------

    if np.any(assignment < 0):
        raise RuntimeError(
            "Auction algorithm returned an incomplete assignment"
        )

    if len(set(assignment.tolist())) != n:
        raise RuntimeError(
            "Auction algorithm returned duplicate column assignments"
        )

    row_indices = np.arange(
        n,
        dtype=np.int64,
    )

    col_indices = assignment

    return (
        row_indices,
        col_indices,
    )


# ============================================================
# Public interface
# ============================================================

def run_auction_assignment(
    case: AssignmentCase,
    unmatched_cost: float = 0.50,
    epsilon: float = 1e-4,
    invalid_cost: float = 1e6,
    max_iterations: int = 1_000_000,
) -> AssignmentResult:
    """
    Run Auction Assignment on one AssignmentCase.

    Parameters
    ----------
    case:
        Complete frame-pair Assignment problem.

    unmatched_cost:
        Cost of deliberately leaving one plane unmatched.

        This parameter will later be tuned using Synthetic DEV and
        frozen before Synthetic FINAL evaluation.

    epsilon:
        Auction bidding parameter.

        Smaller values generally improve approximation to the minimum
        cost solution but can increase runtime.

        This parameter must also be selected using Synthetic DEV only.

    invalid_cost:
        Large cost assigned to pairs rejected by Candidate Gating.

    max_iterations:
        Safety limit for the bidding process.

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

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive"
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
    # Auction optimisation
    # --------------------------------------------------------

    start = perf_counter()

    row_indices, col_indices = (
        _auction_square(
            cost_matrix=cost_matrix,
            epsilon=epsilon,
            max_iterations=max_iterations,
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
        algorithm="auction",
        case_id=case.case_id,
        matches=matches,
        unmatched_current=unmatched_current,
        unmatched_reference=unmatched_reference,
        runtime_ms=runtime_ms,
        metadata={
            "unmatched_cost":
                float(unmatched_cost),

            "epsilon":
                float(epsilon),

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
# Diagnostic helper
# ============================================================

def auction_diagnostics(
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