"""
cost_scaling_assignment.py

Cost-scaling min-cost-flow method for one-to-one plane correspondence
Assignment.

Literature basis:
    H. N. Gabow and R. E. Tarjan,
    "Faster scaling algorithms for network problems,"
    SIAM Journal on Computing, 1989.

The assignment problem is represented as a unit-capacity bipartite
min-cost-flow network:

    source
      ->
    current planes
      ->
    reference planes
      ->
    sink

A cost-scaling / epsilon-refinement procedure is used to obtain the
minimum-cost assignment.

The same augmented cost matrix is used as for Hungarian, Auction and
Jonker-Volgenant, allowing:

- one-to-one correspondences;
- unmatched planes;
- unequal plane-set sizes;
- Candidate Gating restrictions.

Only the optimisation stage is included in measured runtime.
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
# Residual-network edge
# ============================================================

class _Edge:
    """
    Residual-network edge.
    """

    __slots__ = (
        "to",
        "rev",
        "capacity",
        "cost",
    )

    def __init__(
        self,
        to: int,
        rev: int,
        capacity: int,
        cost: int,
    ) -> None:

        self.to = int(to)
        self.rev = int(rev)
        self.capacity = int(capacity)
        self.cost = int(cost)


# ============================================================
# Integer cost conversion
# ============================================================

def _quantise_cost_matrix(
    cost_matrix: np.ndarray,
    cost_scale: int,
) -> np.ndarray:
    """
    Convert floating-point assignment costs into integer costs.

    Cost-scaling methods operate most naturally with integer costs.

    Parameters
    ----------
    cost_matrix:
        Floating-point cost matrix.

    cost_scale:
        Multiplication factor before rounding.

        Example:
            0.123456 -> 123456
            when cost_scale = 1,000,000
    """

    if cost_scale <= 0:
        raise ValueError(
            "cost_scale must be positive"
        )

    cost = np.asarray(
        cost_matrix,
        dtype=np.float64,
    )

    scaled = np.rint(
        cost * float(cost_scale)
    )

    int64_limit = np.iinfo(
        np.int64
    ).max

    if np.any(
        np.abs(scaled)
        >= int64_limit
    ):
        raise OverflowError(
            "Scaled assignment costs exceed int64 range"
        )

    return scaled.astype(
        np.int64
    )


# ============================================================
# Core cost-scaling solver
# ============================================================

def _cost_scaling_square(
    integer_cost_matrix: np.ndarray,
    epsilon_factor: float = 4.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, int | float],
]:
    """
    Solve a square assignment problem using cost-scaling
    min-cost-flow optimisation.

    Parameters
    ----------
    integer_cost_matrix:
        Square integer cost matrix.

    epsilon_factor:
        Factor by which epsilon is reduced between refinement phases.

        A value of 4 is used as the fixed implementation setting.

    Returns
    -------
    row_indices:
        Assignment row indices.

    col_indices:
        Assigned column for every row.

    diagnostics:
        Solver statistics.

    Notes
    -----
    Each row and column has unit capacity.

    Reduced cost is

        c_p(u,v) = c(u,v) + p(u) - p(v)

    where p is the node potential.

    During each epsilon-refinement phase, residual arcs with negative
    reduced cost are admissible. Node potentials are adjusted until a
    feasible epsilon-optimal flow is obtained.
    """

    cost = np.asarray(
        integer_cost_matrix,
        dtype=np.int64,
    )

    if cost.ndim != 2:
        raise ValueError(
            "integer_cost_matrix must be two-dimensional"
        )

    n_rows, n_cols = cost.shape

    if n_rows != n_cols:
        raise ValueError(
            "Cost-scaling implementation requires a square matrix"
        )

    if n_rows == 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
            {
                "pushes": 0,
                "relabels": 0,
                "epsilon_phases": 0,
            },
        )

    if epsilon_factor <= 1.0:
        raise ValueError(
            "epsilon_factor must be greater than 1"
        )

    n = n_rows

    # --------------------------------------------------------
    # Network layout
    #
    # 0                  = source
    # 1 ... n            = row/current nodes
    # n+1 ... 2n         = column/reference nodes
    # 2n+1               = sink
    # --------------------------------------------------------

    source = 0
    row_start = 1
    col_start = 1 + n
    sink = 2 * n + 1

    n_vertices = (
        2 * n + 2
    )

    graph: list[list[_Edge]] = [
        []
        for _ in range(
            n_vertices
        )
    ]

    # Store row -> column edge locations so that the final
    # flow can be decoded efficiently.
    assignment_edges: list[
        list[tuple[int, int]]
    ] = [
        [
            (-1, -1)
            for _ in range(n)
        ]
        for _ in range(n)
    ]

    # --------------------------------------------------------
    # Residual edge helper
    # --------------------------------------------------------

    def add_edge(
        u: int,
        v: int,
        capacity: int,
        edge_cost: int,
    ) -> int:

        forward_index = len(
            graph[u]
        )

        reverse_index = len(
            graph[v]
        )

        graph[u].append(
            _Edge(
                to=v,
                rev=reverse_index,
                capacity=capacity,
                cost=edge_cost,
            )
        )

        graph[v].append(
            _Edge(
                to=u,
                rev=forward_index,
                capacity=0,
                cost=-edge_cost,
            )
        )

        return forward_index

    # --------------------------------------------------------
    # Source -> row nodes
    # --------------------------------------------------------

    for i in range(n):

        add_edge(
            source,
            row_start + i,
            capacity=1,
            edge_cost=0,
        )

    # --------------------------------------------------------
    # Row -> column nodes
    # --------------------------------------------------------

    for i in range(n):

        u = (
            row_start
            + i
        )

        for j in range(n):

            v = (
                col_start
                + j
            )

            edge_index = add_edge(
                u,
                v,
                capacity=1,
                edge_cost=int(
                    cost[i, j]
                ),
            )

            assignment_edges[i][j] = (
                u,
                edge_index,
            )

    # --------------------------------------------------------
    # Column -> sink nodes
    # --------------------------------------------------------

    for j in range(n):

        add_edge(
            col_start + j,
            sink,
            capacity=1,
            edge_cost=0,
        )

    # ========================================================
    # Initial pseudoflow
    # ========================================================

    # excess[v] =
    #     supply + inflow - outflow
    #
    # Source supplies n units and sink consumes n units.

    excess = np.zeros(
        n_vertices,
        dtype=np.int64,
    )

    excess[source] = n
    excess[sink] = -n

    # Node potentials
    potential = np.zeros(
        n_vertices,
        dtype=np.float64,
    )

    push_count = 0
    relabel_count = 0
    epsilon_phases = 0

    # ========================================================
    # Residual helpers
    # ========================================================

    def reduced_cost(
        u: int,
        edge: _Edge,
    ) -> float:

        return (
            float(edge.cost)
            + potential[u]
            - potential[edge.to]
        )

    def push(
        u: int,
        edge_index: int,
        amount: int,
    ) -> None:

        nonlocal push_count

        edge = graph[u][
            edge_index
        ]

        v = edge.to

        reverse = graph[v][
            edge.rev
        ]

        edge.capacity -= amount
        reverse.capacity += amount

        excess[u] -= amount
        excess[v] += amount

        push_count += 1

    # ========================================================
    # One epsilon-refinement phase
    # ========================================================

    def refine(
        epsilon: float,
    ) -> None:

        nonlocal relabel_count

        tolerance = 1e-12

        # ----------------------------------------------------
        # Saturate residual arcs having negative reduced cost.
        # ----------------------------------------------------

        for u in range(
            n_vertices
        ):

            for edge_index, edge in enumerate(
                graph[u]
            ):

                if (
                    edge.capacity > 0
                    and reduced_cost(
                        u,
                        edge,
                    ) < -tolerance
                ):

                    push(
                        u,
                        edge_index,
                        edge.capacity,
                    )

        # ----------------------------------------------------
        # Queue all active nodes.
        # ----------------------------------------------------

        active = deque()

        in_queue = np.zeros(
            n_vertices,
            dtype=bool,
        )

        for u in range(
            n_vertices
        ):

            if excess[u] > 0:

                active.append(u)
                in_queue[u] = True

        current_edge = np.zeros(
            n_vertices,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Push/relabel discharge
        # ----------------------------------------------------

        while active:

            u = active.popleft()

            in_queue[u] = False

            while excess[u] > 0:

                # --------------------------------------------
                # Search for an admissible residual arc.
                # --------------------------------------------

                while (
                    current_edge[u]
                    < len(graph[u])
                ):

                    edge_index = int(
                        current_edge[u]
                    )

                    edge = graph[u][
                        edge_index
                    ]

                    if (
                        edge.capacity > 0
                        and reduced_cost(
                            u,
                            edge,
                        ) < -tolerance
                    ):

                        v = edge.to

                        previous_excess = int(
                            excess[v]
                        )

                        amount = min(
                            int(excess[u]),
                            edge.capacity,
                        )

                        push(
                            u,
                            edge_index,
                            amount,
                        )

                        # Newly active node
                        if (
                            previous_excess <= 0
                            and excess[v] > 0
                            and not in_queue[v]
                        ):

                            active.append(v)

                            in_queue[v] = True

                        if excess[u] == 0:
                            break

                    else:

                        current_edge[u] += 1

                if excess[u] == 0:
                    break

                # --------------------------------------------
                # No admissible residual arc remains.
                # Relabel node potential.
                # --------------------------------------------

                available_costs = [
                    reduced_cost(
                        u,
                        edge,
                    )
                    for edge in graph[u]
                    if edge.capacity > 0
                ]

                if not available_costs:
                    raise RuntimeError(
                        "Cost-scaling solver encountered "
                        "an active node without a residual edge"
                    )

                minimum_reduced_cost = min(
                    available_costs
                )

                # Shift p(u) so that the best outgoing residual
                # arc becomes epsilon-admissible.
                potential[u] -= (
                    minimum_reduced_cost
                    + epsilon
                )

                relabel_count += 1

                current_edge[u] = 0

    # ========================================================
    # Epsilon scaling
    # ========================================================

    maximum_cost = int(
        np.max(
            np.abs(cost)
        )
    )

    epsilon = float(
        max(
            1,
            maximum_cost,
        )
    )

    # For integer costs, an epsilon-optimal feasible flow with
    # epsilon < 1 / |V| is optimal.
    stopping_epsilon = (
        1.0
        / float(n_vertices)
    )

    while True:

        refine(
            epsilon
        )

        epsilon_phases += 1

        if (
            epsilon
            < stopping_epsilon
        ):
            break

        epsilon /= (
            epsilon_factor
        )

    # ========================================================
    # Verify feasibility
    # ========================================================

    if np.any(
        excess != 0
    ):
        raise RuntimeError(
            "Cost-scaling solver did not produce "
            "a feasible final flow"
        )

    # ========================================================
    # Decode row -> column flow
    # ========================================================

    assignment = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for i in range(n):

        for j in range(n):

            u, edge_index = (
                assignment_edges[i][j]
            )

            edge = graph[u][
                edge_index
            ]

            reverse_edge = graph[
                edge.to
            ][
                edge.rev
            ]

            # Reverse residual capacity of one means that one
            # unit of flow was sent through this row-column arc.
            if reverse_edge.capacity > 0:

                if assignment[i] != -1:
                    raise RuntimeError(
                        "Cost-scaling solver returned multiple "
                        "columns for one row"
                    )

                assignment[i] = j

    if np.any(
        assignment < 0
    ):
        raise RuntimeError(
            "Cost-scaling solver returned an incomplete assignment"
        )

    if (
        len(
            set(
                assignment.tolist()
            )
        )
        != n
    ):
        raise RuntimeError(
            "Cost-scaling solver returned duplicate "
            "column assignments"
        )

    row_indices = np.arange(
        n,
        dtype=np.int64,
    )

    col_indices = assignment

    diagnostics = {
        "pushes":
            int(push_count),

        "relabels":
            int(relabel_count),

        "epsilon_phases":
            int(epsilon_phases),

        "final_epsilon":
            float(epsilon),
    }

    return (
        row_indices,
        col_indices,
        diagnostics,
    )


# ============================================================
# Public Assignment interface
# ============================================================

def run_cost_scaling_assignment(
    case: AssignmentCase,
    unmatched_cost: float = 0.50,
    invalid_cost: float = 1e6,
    cost_scale: int = 1_000_000,
    epsilon_factor: float = 4.0,
) -> AssignmentResult:
    """
    Run Cost-Scaling Assignment on one AssignmentCase.

    Parameters
    ----------
    case:
        Complete frame-pair Assignment problem.

    unmatched_cost:
        Cost of deliberately leaving one plane unmatched.

        This will later be selected using Synthetic DEV.

    invalid_cost:
        Cost assigned to correspondences rejected by Candidate Gating.

    cost_scale:
        Numerical scaling used to convert floating-point costs into
        integer costs.

        This is an implementation precision setting rather than an
        experimental model parameter.

    epsilon_factor:
        Fixed epsilon reduction factor used by the cost-scaling solver.

    Returns
    -------
    AssignmentResult
        Standardised Assignment result.
    """

    validate_assignment_case(
        case
    )

    if unmatched_cost < 0:
        raise ValueError(
            "unmatched_cost must be non-negative"
        )

    if invalid_cost <= unmatched_cost:
        raise ValueError(
            "invalid_cost must be larger than unmatched_cost"
        )

    if cost_scale <= 0:
        raise ValueError(
            "cost_scale must be positive"
        )

    # --------------------------------------------------------
    # Common preprocessing
    #
    # Excluded from measured optimisation runtime.
    # --------------------------------------------------------

    cost_matrix = (
        build_augmented_cost_matrix(
            case=case,
            unmatched_cost=unmatched_cost,
            invalid_cost=invalid_cost,
        )
    )

    integer_cost_matrix = (
        _quantise_cost_matrix(
            cost_matrix=cost_matrix,
            cost_scale=cost_scale,
        )
    )

    # --------------------------------------------------------
    # Cost-scaling optimisation
    # --------------------------------------------------------

    start = perf_counter()

    (
        row_indices,
        col_indices,
        solver_diagnostics,
    ) = _cost_scaling_square(
        integer_cost_matrix=
            integer_cost_matrix,

        epsilon_factor=
            epsilon_factor,
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
        algorithm="cost_scaling",
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

            "cost_scale":
                int(cost_scale),

            "epsilon_factor":
                float(epsilon_factor),

            "augmented_matrix_size":
                int(
                    cost_matrix.shape[0]
                ),

            "total_augmented_cost":
                total_augmented_cost,

            "real_match_cost":
                real_match_cost,

            **solver_diagnostics,
        },
    )


# ============================================================
# Diagnostic helper
# ============================================================

def cost_scaling_diagnostics(
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
            float(
                result.runtime_ms
            ),

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