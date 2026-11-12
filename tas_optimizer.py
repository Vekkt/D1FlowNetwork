#!/usr/bin/env python3
"""TAS allocation: expected residual hedge plus full-completion penalty.

Install: python -m pip install numpy "scipy>=1.11"
Run:     python tas_optimizer.py --kappa 20 --all-edges
         python tas_optimizer.py --twelve-contract-demo --all-edges
         python tas_optimizer.py --two-contract-demo
         python tas_optimizer.py --six-contract-demo
         python tas_optimizer.py --all-filled-demo --plot all_filled.svg
         python tas_optimizer.py --plot network.svg --all-edges
         python tas_optimizer.py --self-test

Model
-----
q[e]       = nonnegative INTEGER lots submitted on directed edge e.
A @ q = b  = the full TAS-eligible hedge, if every submitted order fills.
L[e, k]    = alpha[e] * historical daily TAS volume in observation k.
pi[e, k]   = normalized weight of comparable historical observation k.
F[e](q)    = min(q[e], L[e]); random, potentially partial order fill.
p[e](q)    = weighted fraction of observations with L[e, k] >= q.
m[e](q)    = weighted average of min(q, L[e, k]); reporting only.
P(q)       = product_e p[e](q[e]), with p[e](0) = 1.
R(q)       = E[sum_i abs(b[i] - (A @ F(q))[i])]; maturities only, no OR.
J(q)       = R(q) + kappa * (1 - P(q)). Minimize J, then brokerage.
kappa      = penalty in residual-lot units for failing full completion.
             Default 20 is illustrative, NOT a calibrated trading parameter.
             One percentage point of P is worth 0.01*kappa residual lots.

Fills on different edges are assumed independent. Partial fills are allowed.
Only actual contracts appear in A's rows. None denotes the virtual OR node
when constructing edges; it is not an instrument, position, or order variable.
Both directions of each active instrument can be included. No order-size
constraint is imposed from forecast liquidity. Selected routes must contain
no directed round trips, including through OR. This execution-policy rule is
now explicitly enforced inside the optimization. Brokerage is a secondary
tie-break on J; probability no longer has absolute priority over partial fills.
Futures execution and slippage are excluded.

Default worked example: 12 futures maturities, nonzero hedge targets on five,
TAS instruments involving only four. A closed TAS maturity has no trading
edge: its target cannot satisfy A @ q = b. The example explicitly separates
b_full = b_TAS + r_unavailable, solves A @ q = b_TAS, and reserves
r_unavailable for futures. This is preprocessing, not futures optimization.
The reported TAS-order probability applies only to the TAS-eligible target;
the probability of completing the ENTIRE hedge using TAS is zero when the
unavailable reserve is nonzero. Core optimize_tas remains strict.

Solver
------
An interval MILP represents integer q, min(q,L), and each stepwise p(q).
Binary/continuous product linearizations represent P exactly, including zeros.
Expected absolute residuals are integrated locally at each maturity, using
independent liquidity distributions. Equal-index historical rows on different
instruments are NOT treated as correlated future outcomes.

Local combinations <= exact_state_limit (default 4096) are enumerated exactly.
Above that limit, fixed seeded Monte Carlo draws approximate R, while P stays
exact. residual_samples (default 2048) controls draws per approximated node.
Results state whether R is exact or approximate. All integration rows share
ONE submitted order vector; there are no later conditional decisions.
The worked examples use exact integration. SciPy/HiGHS solves the resulting
MILP to its numerical tolerances, then minimizes brokerage at the best J.
Timeouts raise instead of presenting an unproved optimum. This is not a
classical linear per-unit-cost transshipment solver.

R takes absolute values BEFORE averaging; abs(b - A @ mean_fill) is only a
lower bound and can hide intermediate-maturity exposure. When all P are zero,
J differs between plans through R. A 99-lot fill of a 100-lot order therefore
scores better than an 80-lot fill, despite both full-fill probabilities being 0.

Brokerage is dollars per executed instrument lot, including BOTH legs when
applicable. make_edges() defaults to illustrative fees of $1 per outright lot
and $2 per spread lot. Supply your actual fee schedule. The secondary objective
prices the entire submitted plan as if fully filled; expected actual fees
sum(beta * mean_fill) are separately reported. These coincide at 100% fills.
There is no per-ticket fixed charge, so fewer tickets need not mean lower fees.

A structural search bound is max(total required buys, total required sells).
It is valid for the explicitly acyclic execution policy, not a liquidity cap.
No post-optimization cycle removal is performed: it could change R and J.
For acyclic plans, all-order completion and exact net-hedge completion coincide.

Historical volumes provide heuristic probabilities, not calibrated own-order
fill probabilities. Seasonality uses caller-supplied exchange trading-day
distances, rather than approximating exchange calendars with weekdays.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from itertools import product
from math import exp, fsum, isfinite, log
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class FillForecast:
    """Weighted accessible-volume observations for ONE directed edge.

    Use from_volumes() for raw daily volume. Weights may be unnormalized.
    The samples need not be aligned across edges: the joint objective uses
    the independence assumption, not a shared historical scenario index.
    """

    accessible_volume: Sequence[float]
    weights: Sequence[float]

    def __post_init__(self) -> None:
        capacity = np.asarray(self.accessible_volume, dtype=float)
        weight = np.asarray(self.weights, dtype=float)
        if capacity.ndim != 1 or capacity.size == 0 or weight.shape != capacity.shape:
            raise ValueError("Provide equally sized, nonempty 1D volumes and weights.")
        if not np.all(np.isfinite(capacity)) or np.any(capacity < 0):
            raise ValueError("Accessible volumes must be finite and nonnegative.")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0) or not np.any(weight > 0):
            raise ValueError("Weights must be finite, nonnegative, and have positive sum.")
        keep = weight > 0
        self.accessible_volume = capacity[keep].copy()
        weight = weight[keep] / np.max(weight[keep])
        self.weights = weight / fsum(weight)

    @classmethod
    def from_volumes(cls, volumes: Sequence[float], weights: Sequence[float],
                     alpha: float = 0.01) -> FillForecast:
        if not isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        return cls(alpha * np.asarray(volumes, dtype=float), weights)

    @staticmethod
    def _quantity(q: int) -> int:
        if not isfinite(q) or q < 0 or int(q) != q:
            raise ValueError("Order size must be a nonnegative integer.")
        return int(q)

    def probability(self, q: int) -> float:
        """p(q): probability the entire order fills; p(0) = 1."""
        q = self._quantity(q)
        if q == 0:
            return 1.0
        enough_volume = self.accessible_volume >= q
        if np.all(enough_volume):
            return 1.0
        return min(1.0, fsum(self.weights[enough_volume]))

    def mean_fill(self, q: int) -> float:
        """m(q): average modeled filled lots, NOT full-fill probability."""
        q = self._quantity(q)
        return float(np.dot(self.weights, np.minimum(q, self.accessible_volume)))

    def positive_bands(self, max_q: int) -> list[tuple[int, int, float]]:
        """Inclusive integer intervals [lower, upper] with constant positive p.

        A capacity L stops supporting full fills at integer floor(L) + 1.
        Separate q=0 initially, then merge intervals with identical p.
        """
        starts = {0, 1, max_q + 1}
        for value in self.accessible_volume:
            if value < max_q:
                starts.add(int(np.floor(value)) + 1)
        starts = sorted(x for x in starts if 0 <= x <= max_q + 1)
        bands: list[tuple[int, int, float]] = []
        for lower, stop in zip(starts, starts[1:]):
            upper = stop - 1
            probability = self.probability(lower)
            if probability == 0:
                continue
            if bands and bands[-1][1] + 1 == lower and bands[-1][2] == probability:
                bands[-1] = (bands[-1][0], upper, probability)
            else:
                bands.append((lower, upper, probability))
        return bands


@dataclass(frozen=True)
class HistoricalVolume:
    """One daily observation of the same commodity/instrument family.

    delivery_months: ((year, month),) for an outright, or the ordered pair
    ((near_year, near_month), (far_year, far_month)) for a calendar spread.
    days_to_tas_end: exchange trading-day distance to this instrument's actual
    final TAS session, calculated upstream with the exchange calendar.
    """

    trade_date: date
    delivery_months: tuple[tuple[int, int], ...]
    days_to_tas_end: int
    volume: float | None
    available: bool = True


def seasonal_forecast(
    history: Sequence[HistoricalVolume],
    target_delivery_months: tuple[tuple[int, int], ...],
    target_days_to_tas_end: int,
    as_of: date,
    *, years: int = 5, window: int = 5, decay: float = 0.8,
    alpha: float = 0.01,
) -> FillForecast:
    """Match delivery months, year gaps and lifecycle; return fixed SOD curve.

    Every historical leg must be shifted back by the same number of years.
    This preserves cross-year spreads, e.g. Dec 2026 / Jan 2027.
    Weight EACH selected daily observation by decay**(year_age - 1), then
    normalize. Actual zero-volume days stay; missing/unavailable days do not.
    No same-day or later observations are used at start of day.
    """
    if (not target_delivery_months or years < 1 or window < 0
            or not 0 < decay <= 1):
        raise ValueError("Invalid seasonal matching parameters.")
    if (not isfinite(target_days_to_tas_end) or target_days_to_tas_end < 0
            or int(target_days_to_tas_end) != target_days_to_tas_end):
        raise ValueError("Target lifecycle distance must be a nonnegative integer.")
    for year, month in target_delivery_months:
        if not 1 <= month <= 12 or year < 1:
            raise ValueError("Invalid delivery year/month.")
    volumes, weights = [], []
    for obs in history:
        if not obs.available or obs.volume is None or obs.trade_date >= as_of:
            continue
        if len(obs.delivery_months) != len(target_delivery_months):
            continue
        if any(current[1] != past[1]
               for current, past in zip(target_delivery_months, obs.delivery_months)):
            continue
        ages = [current[0] - past[0]
                for current, past in zip(target_delivery_months, obs.delivery_months)]
        age = ages[0]
        if len(set(ages)) != 1 or not 1 <= age <= years:
            continue
        if (not isfinite(obs.days_to_tas_end) or obs.days_to_tas_end < 0
                or int(obs.days_to_tas_end) != obs.days_to_tas_end):
            raise ValueError("Historical lifecycle distance must be a nonnegative integer.")
        if abs(obs.days_to_tas_end - target_days_to_tas_end) > window:
            continue
        if not isfinite(obs.volume) or obs.volume < 0:
            raise ValueError("A selected observed volume is invalid; use None for missing.")
        volumes.append(obs.volume)
        weights.append(decay ** (age - 1))
    if not volumes:
        raise ValueError("Insufficient history: no comparable valid observations.")
    return FillForecast.from_volumes(volumes, weights, alpha)


@dataclass(frozen=True)
class Edge:
    name: str
    source: str | None       # None is OR. No OR row in A.
    destination: str | None
    forecast: FillForecast
    brokerage_per_lot: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.brokerage_per_lot) or self.brokerage_per_lot < 0:
            raise ValueError("Brokerage per instrument lot must be finite and nonnegative.")


def make_edges(
    contracts: Sequence[str],
    outright_forecasts: Mapping[str, FillForecast],
    spread_forecasts: Mapping[tuple[str, str], FillForecast],
    *, outright_brokerage: float | Mapping[str, float] = 1.0,
    spread_brokerage: float | Mapping[tuple[str, str], float] = 2.0,
) -> list[Edge]:
    """Build BOTH directions of every supplied outright and active spread.

    A spread key (i, j) means its buy direction buys i and sells j.
    No spread is inferred merely because both contracts exist.
    Daily volume has no side information, so this helper uses the same
    initial curve for both directions. Construct Edge objects directly for
    side-specific curves, or to allow only selected directions.
    Brokerage schedules can be scalars or mappings with the same keys as the
    forecasts. Spread fees must include all charged legs; do not double them
    again. All rates must be in the same currency (dollars in the reports).
    """
    if len(set(contracts)) != len(contracts) or any(not x for x in contracts):
        raise ValueError("Contract names must be unique, nonempty strings.")
    names = set(contracts)
    if set(outright_forecasts) - names:
        raise ValueError("An outright references an unknown contract.")
    edges = []

    def fee(schedule, key):
        return float(schedule[key] if isinstance(schedule, Mapping) else schedule)

    for contract in contracts:
        if contract in outright_forecasts:
            forecast = outright_forecasts[contract]
            rate = fee(outright_brokerage, contract)
            edges.extend([Edge(f"Buy {contract}", None, contract, forecast, rate),
                          Edge(f"Sell {contract}", contract, None, forecast, rate)])
    seen = set()
    for (buy, sell), forecast in spread_forecasts.items():
        pair = frozenset((buy, sell))
        if buy == sell or buy not in names or sell not in names or pair in seen:
            raise ValueError("Spread pairs must be distinct, known, and not duplicated.")
        seen.add(pair)
        rate = fee(spread_brokerage, (buy, sell))
        edges.extend([Edge(f"Buy {buy} / sell {sell}", sell, buy, forecast, rate),
                      Edge(f"Sell {buy} / buy {sell}", buy, sell, forecast, rate)])
    return edges


def incidence_matrix(contracts: Sequence[str], edges: Sequence[Edge]) -> np.ndarray:
    """N contract rows, E directed-edge columns; no tradable OR quantity."""
    if len(set(contracts)) != len(contracts):
        raise ValueError("Contract rows must be unique.")
    row = {contract: i for i, contract in enumerate(contracts)}
    A = np.zeros((len(contracts), len(edges)), dtype=np.int64)
    for e, edge in enumerate(edges):
        if edge.source == edge.destination:
            raise ValueError("Self-loops are not trading instruments.")
        for node, sign in ((edge.source, -1), (edge.destination, 1)):
            if node is not None:
                if node not in row:
                    raise ValueError(f"Unknown contract {node!r} on edge {edge.name!r}.")
                A[row[node], e] = sign
    return A


@dataclass
class OptimizationResult:
    contracts: tuple[str, ...]
    edges: tuple[Edge, ...]
    A: np.ndarray
    b: np.ndarray
    q: np.ndarray
    full_fill_probabilities: np.ndarray
    mean_fills: np.ndarray
    h_hat: np.ndarray
    joint_probability: float
    log_joint_probability: float
    kappa: float
    residual_by_maturity: np.ndarray
    expected_residual: float
    objective_value: float
    residual_is_exact: bool
    residual_state_count: int
    residual_samples: int
    seed: int
    solve_seconds: float
    probability_band_count: int
    brokerage_rates: np.ndarray
    brokerage_if_fully_filled: float
    expected_brokerage: float


@dataclass
class ResidualRule:
    """Weighted integration rule for ONE maturity's absolute residual.

    Rows are liquidity outcomes used to evaluate an expectation. They do not
    create scenario-dependent orders: every row uses the SAME decision q.
    """
    edge_indices: tuple[int, ...]
    capacities: np.ndarray
    weights: np.ndarray
    exact: bool


def _is_acyclic(edges: Sequence[Edge], q: Sequence[int]) -> bool:
    """The selected orders must not contain a directed round trip, including OR."""
    nodes = {node for edge in edges for node in (edge.source, edge.destination)}
    outgoing = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge, size in zip(edges, q):
        if size > 0:
            outgoing[edge.source].append(edge.destination)
            indegree[edge.destination] += 1
    stack = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        for destination in outgoing[node]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                stack.append(destination)
    return visited == len(nodes)


def _distribution(forecast: FillForecast, bound: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge equal outcomes, clipping above a structural order bound only."""
    values, inverse = np.unique(np.minimum(forecast.accessible_volume, bound), return_inverse=True)
    weights = np.bincount(inverse, weights=forecast.weights)
    weights /= weights.sum()
    return values, weights


def _residual_rules(
    A: np.ndarray, edges: Sequence[Edge], bound: int,
    *, exact_state_limit: int, residual_samples: int, seed: int,
) -> list[ResidualRule]:
    """Integrate locally at each node instead of enumerating global outcomes.

    Pair opposite directions of the same link. Only one can be selected in
    an acyclic plan, so a common uniform quantile for that pair does not
    alter the assumed independence of ANY simultaneously submitted orders.
    Different links have independent uniforms. Unequal directional curves
    are supported by taking the union of their CDF breakpoints. Parallel
    edges in the same direction remain independent, separate groups.
    """
    distributions = [_distribution(edge.forecast, bound) for edge in edges]
    remaining = set(range(len(edges)))
    groups = []
    for e, edge in enumerate(edges):
        if e not in remaining:
            continue
        remaining.remove(e)
        reverse = next((j for j in sorted(remaining)
                        if edges[j].source == edge.destination
                        and edges[j].destination == edge.source), None)
        members = (e,) if reverse is None else (e, reverse)
        if reverse is not None:
            remaining.remove(reverse)
        cdfs = []
        for j in members:
            cdf = np.cumsum(distributions[j][1])
            cdf[-1] = 1.0
            cdfs.append(cdf)
        knots = np.unique(np.r_[0.0, *cdfs])
        weights = np.diff(knots)
        midpoints = (knots[:-1] + knots[1:]) / 2
        capacities = np.column_stack([
            distributions[j][0][np.searchsorted(cdf, midpoints, side="right")]
            for j, cdf in zip(members, cdfs)])
        # Quantiles can create duplicate rows after clipping; aggregate them.
        capacities, inverse = np.unique(capacities, axis=0, return_inverse=True)
        weights = np.bincount(inverse, weights=weights)
        weights /= weights.sum()
        groups.append((members, capacities, weights))

    rules = []
    for i in range(A.shape[0]):
        incident = tuple(int(e) for e in np.flatnonzero(A[i]))
        relevant = [group for group in groups if any(A[i, e] for e in group[0])]
        count = 1
        for _, capacities, _ in relevant:
            count *= len(capacities)
        exact = count <= exact_state_limit or count == 1
        if exact:
            choices = np.array(list(product(*(range(len(group[1])) for group in relevant))),
                               dtype=int).reshape(count, len(relevant))
            weights = np.ones(count)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            choices = np.column_stack([
                rng.choice(len(group[1]), size=residual_samples, p=group[2])
                for group in relevant])
            weights = np.full(residual_samples, 1.0 / residual_samples)
        capacities = np.zeros((len(choices), len(incident)))
        column = {e: j for j, e in enumerate(incident)}
        for g, (members, values, probabilities) in enumerate(relevant):
            selected = choices[:, g]
            if exact:
                weights *= probabilities[selected]
            for j, e in enumerate(members):
                if e in column:
                    capacities[:, column[e]] = values[selected, j]
        # Keep identical sampled outcomes only once, with combined weights.
        capacities, inverse = np.unique(capacities, axis=0, return_inverse=True)
        weights = np.bincount(inverse, weights=weights)
        weights /= weights.sum()
        rules.append(ResidualRule(incident, capacities, weights, exact))
    return rules


def _allocation_metrics(A, b, edges, q, rules):
    probabilities = np.array([edge.forecast.probability(int(size))
                              for edge, size in zip(edges, q)])
    means = np.array([edge.forecast.mean_fill(int(size)) for edge, size in zip(edges, q)])
    log_joint = -np.inf if np.any(probabilities == 0) else fsum(log(p) for p in probabilities)
    joint = exp(log_joint) if isfinite(log_joint) else 0.0
    residual = np.zeros(len(b))
    for i, rule in enumerate(rules):
        indices = list(rule.edge_indices)
        fills = np.minimum(q[indices], rule.capacities)
        residual[i] = rule.weights @ np.abs(b[i] - fills @ A[i, indices])
    return probabilities, means, log_joint, joint, residual


def evaluate_allocation(
    contracts: Sequence[str], edges: Sequence[Edge], hedge: Sequence[int], q: Sequence[int],
    *, kappa: float = 20.0, exact_state_limit: int = 4096,
    residual_samples: int = 2048, seed: int = 2026,
) -> dict:
    """Score a fixed feasible, acyclic allocation without optimizing it."""
    contracts, edges = tuple(contracts), tuple(edges)
    A = incidence_matrix(contracts, edges)
    b, q = np.asarray(hedge, dtype=float), np.asarray(q, dtype=float)
    if (b.shape != (len(contracts),) or q.shape != (len(edges),)
            or not np.all(np.isfinite(b)) or not np.all(np.isfinite(q))
            or not np.all(b == np.rint(b)) or not np.all(q == np.rint(q))
            or np.any(q < 0) or not np.array_equal(A @ q, b)):
        raise ValueError("Provide finite integer hedge/orders with q>=0 and Aq=b.")
    if not _is_acyclic(edges, q):
        raise ValueError("The selected allocation contains a directed round trip.")
    _validate_settings(kappa, exact_state_limit, residual_samples, seed, None)
    b, q = b.astype(np.int64), q.astype(np.int64)
    bound = int(max(b[b > 0].sum(), -b[b < 0].sum()))
    rules = _residual_rules(A, edges, bound, exact_state_limit=exact_state_limit,
                            residual_samples=residual_samples, seed=seed)
    p, means, log_joint, joint, residual = _allocation_metrics(A, b, edges, q, rules)
    return dict(joint_probability=joint, expected_residual=float(residual.sum()),
                residual_by_maturity=residual, mean_fills=means, h_hat=A @ means,
                objective_value=float(residual.sum() + kappa * (1 - joint)),
                residual_is_exact=all(rule.exact for rule in rules),
                brokerage_if_fully_filled=float(sum(edge.brokerage_per_lot * size
                                                     for edge, size in zip(edges, q))))


def _validate_settings(kappa, exact_state_limit, residual_samples, seed, time_limit):
    if not isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative (in residual-lot units).")
    for value, name, minimum in ((exact_state_limit, "exact_state_limit", 0),
                                 (residual_samples, "residual_samples", 1), (seed, "seed", 0)):
        if not isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if time_limit is not None and (not isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be positive.")


class _LinearModel:
    """Small sparse MILP builder. All objective coefficients start at zero."""
    def __init__(self):
        self.lower, self.upper, self.integer, self.cost = [], [], [], []
        self.rows, self.columns, self.values = [], [], []
        self.row_lower, self.row_upper = [], []

    def variable(self, lower=0.0, upper=np.inf, integer=False, cost=0.0):
        index = len(self.lower)
        self.lower.append(lower)
        self.upper.append(upper)
        self.integer.append(int(integer))
        self.cost.append(cost)
        return index

    def constrain(self, coefficients, lower=-np.inf, upper=np.inf):
        row = len(self.row_lower)
        for column, value in coefficients.items():
            if value:
                self.rows.append(row)
                self.columns.append(column)
                self.values.append(value)
        self.row_lower.append(lower)
        self.row_upper.append(upper)

    def constraint(self):
        matrix = coo_matrix((self.values, (self.rows, self.columns)),
                            shape=(len(self.row_lower), len(self.lower))).tocsc()
        return LinearConstraint(matrix, self.row_lower, self.row_upper)


def optimize_tas(
    contracts: Sequence[str], edges: Sequence[Edge], hedge: Sequence[int],
    *, kappa: float = 20.0, exact_state_limit: int = 4096,
    residual_samples: int = 2048, seed: int = 2026,
    time_limit: float | None = None,
) -> OptimizationResult:
    """Minimize R(q)+kappa*(1-P(q)), then full-fill brokerage, or raise.

    A single fixed integer order vector covers the target: Aq=b. No liquidity
    cap is placed on submissions. Selected routes are explicitly acyclic,
    including OR; unlike the previous probability-only solver, cycles are
    NEVER removed after optimizing. This is an execution-policy constraint.

    P is exact for the supplied empirical marginal curves. R is integrated
    exactly per maturity when local combinations <= exact_state_limit;
    otherwise reproducible Monte Carlo integration uses residual_samples
    independent draws per maturity. Approximation is explicitly reported.
    Optimality is relative to that integration rule and solver tolerances.
    A finite time limit applies separately to each optimization phase.
    """
    started = perf_counter()
    contracts, edges = tuple(contracts), tuple(edges)
    A = incidence_matrix(contracts, edges)
    raw_b = np.asarray(hedge, dtype=float)
    if (raw_b.shape != (len(contracts),) or not np.all(np.isfinite(raw_b))
            or not np.all(raw_b == np.rint(raw_b))):
        raise ValueError("hedge must contain one finite integer per contract row.")
    _validate_settings(kappa, exact_state_limit, residual_samples, seed, time_limit)
    b = raw_b.astype(np.int64)
    N, E = A.shape
    fees = np.array([edge.brokerage_per_lot for edge in edges], dtype=float)
    # In an acyclic feasible flow, no edge exceeds total source flow.
    bound = int(max(b[b > 0].sum(), -b[b < 0].sum()))
    rules = _residual_rules(A, edges, bound, exact_state_limit=exact_state_limit,
                            residual_samples=residual_samples, seed=seed)
    band_count = 0
    primary_score = None
    if bound == 0:
        q = np.zeros(E, dtype=np.int64)
    elif E == 0:
        raise ValueError("No trading edges can execute the nonzero hedge.")
    else:
        model = _LinearModel()
        q_vars = [model.variable(upper=bound, integer=True) for _ in edges]
        nodes = [None, *contracts]
        ranks = {node: model.variable(upper=len(nodes) - 1) for node in nodes}
        bands = []
        for e, edge in enumerate(edges):
            starts = sorted({0, 1, bound + 1, *(int(np.floor(value)) + 1
                            for value in edge.forecast.accessible_volume if value < bound)})
            edge_bands = []
            for lo, stop in zip(starts, starts[1:]):
                hi = stop - 1
                z = model.variable(upper=1, integer=True)
                x = model.variable(upper=hi)
                model.constrain({x: 1, z: -lo}, lower=0)
                model.constrain({x: 1, z: -hi}, upper=0)
                edge_bands.append((lo, hi, edge.forecast.probability(lo), z, x))
            bands.append(edge_bands)
            model.constrain({band[3]: 1 for band in edge_bands}, lower=1, upper=1)
            model.constrain({q_vars[e]: 1, **{band[4]: -1 for band in edge_bands}}, lower=0, upper=0)
            # q>0 iff zero-size band is not selected. Topological ranks then
            # increase strictly along each selected route, excluding cycles.
            z_zero = edge_bands[0][3]
            model.constrain({ranks[edge.source]: 1, ranks[edge.destination]: -1,
                             z_zero: -len(nodes)}, upper=-1)
        band_count = sum(len(edge_bands) for edge_bands in bands)
        for i in range(N):
            model.constrain({q_vars[e]: int(A[i, e]) for e in np.flatnonzero(A[i])},
                            lower=int(b[i]), upper=int(b[i]))

        # Exact product of discrete band probabilities. For each selected
        # band z, w = previous_product*z has an exact binary linearization.
        previous = None
        for edge_bands in bands:
            current = model.variable(upper=1)
            expression = {current: 1}
            for lo, hi, probability, z, x in edge_bands:
                if probability == 0:
                    continue
                if previous is None:
                    w = z  # Initial product is one.
                else:
                    w = model.variable(upper=1)
                    model.constrain({w: 1, z: -1}, upper=0)
                    model.constrain({w: 1, previous: -1}, upper=0)
                    model.constrain({w: 1, previous: -1, z: -1}, lower=-1)
                expression[w] = -probability
            model.constrain(expression, lower=0, upper=0)
            previous = current
        model.cost[previous] = -float(kappa)

        # min(q,L) is affine within each integer band. Reuse fill variables
        # between integration rows and both endpoints of a spread.
        fill_variables = {}
        def fill_variable(e, capacity):
            key = (e, float(capacity))
            if key not in fill_variables:
                variable = model.variable(upper=min(bound, capacity))
                expression = {variable: 1}
                for lo, hi, probability, z, x in bands[e]:
                    if hi <= capacity:
                        expression[x] = -1
                    elif lo > capacity:
                        expression[z] = -float(capacity)
                    else:
                        raise RuntimeError("An integer band straddles a fill breakpoint.")
                model.constrain(expression, lower=0, upper=0)
                fill_variables[key] = variable
            return fill_variables[key]

        for i, rule in enumerate(rules):
            if not rule.edge_indices:
                continue  # Nonzero isolated target is rejected by Aq=b.
            for capacities, weight in zip(rule.capacities, rule.weights):
                residual = model.variable(cost=float(weight))
                plus, minus = {residual: 1}, {residual: 1}
                for e, capacity in zip(rule.edge_indices, capacities):
                    variable = fill_variable(e, capacity)
                    plus[variable] = int(A[i, e])
                    minus[variable] = -int(A[i, e])
                model.constrain(plus, lower=int(b[i]))
                model.constrain(minus, lower=-int(b[i]))

        options = {"mip_rel_gap": 0.0}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        base_constraint = model.constraint()
        objective = np.asarray(model.cost)
        bounds = Bounds(model.lower, model.upper)
        integrality = np.asarray(model.integer)
        first = milp(objective, integrality=integrality, bounds=bounds,
                     constraints=base_constraint, options=options)
        if first.status == 2:
            raise ValueError("The available TAS network cannot execute the full target without round trips.")
        if first.status != 0 or first.x is None:
            raise RuntimeError(f"Combined-objective optimality not established: {first.message}")
        first_q = np.rint(first.x[q_vars]).astype(np.int64)
        if not np.array_equal(A @ first_q, b) or not _is_acyclic(edges, first_q):
            raise RuntimeError("Primary solution failed rounded feasibility checks.")
        _, _, _, first_probability, first_residual = _allocation_metrics(A, b, edges, first_q, rules)
        primary_score = float(first_residual.sum() + kappa * (1 - first_probability))
        if abs(primary_score - (first.fun + kappa)) > max(1e-5, 1e-8 * abs(primary_score)):
            raise RuntimeError("Primary solver objective disagrees with recomputed fills.")
        secondary = np.zeros_like(objective)
        secondary[q_vars] = fees
        locked_score = LinearConstraint(objective.reshape(1, -1), -np.inf, primary_score - kappa + 1e-7)
        secondary_upper = np.array(model.upper)
        if kappa > 0 and primary_score == 0 and first_probability == 1:
            # Do not let numerical score tolerances trade certain execution
            # for a slightly less probable but cheaper order.
            for edge_bands in bands:
                for lo, hi, probability, z, x in edge_bands:
                    if probability < 1:
                        secondary_upper[z] = 0
        secondary_bounds = Bounds(model.lower, secondary_upper)
        second = milp(secondary, integrality=integrality, bounds=secondary_bounds,
                      constraints=[base_constraint, locked_score], options=options)
        if second.status == 2:
            # The primary integer plan is a feasible witness for this score
            # lock. Tight objective rows can trigger a numerical presolve
            # infeasibility in HiGHS; retry the same MILP without presolve.
            second = milp(secondary, integrality=integrality, bounds=secondary_bounds,
                          constraints=[base_constraint, locked_score],
                          options={**options, "presolve": False})
        if second.status != 0 or second.x is None:
            raise RuntimeError(f"Brokerage tie-break optimality not established: {second.message}")
        raw_q = second.x[q_vars]
        if np.max(np.abs(raw_q - np.rint(raw_q))) > 1e-5:
            raise RuntimeError("Solver returned non-integral order quantities.")
        q = np.rint(raw_q).astype(np.int64)

    if np.any(q < 0) or not np.array_equal(A @ q, b) or not _is_acyclic(edges, q):
        raise RuntimeError("Result failed Aq=b, nonnegative quantity, or no-round-trip checks.")
    p, means, log_joint, joint, residual = _allocation_metrics(A, b, edges, q, rules)
    expected_residual = float(residual.sum())
    score = expected_residual + kappa * (1 - joint)
    if primary_score is not None and abs(score - primary_score) > max(1e-5, 1e-8 * abs(primary_score)):
        raise RuntimeError("Recomputed combined score disagrees with the optimized score.")
    return OptimizationResult(
        contracts=contracts, edges=edges, A=A, b=b, q=q,
        full_fill_probabilities=p, mean_fills=means, h_hat=A @ means,
        joint_probability=joint, log_joint_probability=log_joint, kappa=float(kappa),
        residual_by_maturity=residual, expected_residual=expected_residual,
        objective_value=score, residual_is_exact=all(rule.exact for rule in rules),
        residual_state_count=sum(len(rule.weights) for rule in rules if rule.edge_indices),
        residual_samples=residual_samples, seed=seed,
        solve_seconds=perf_counter() - started, probability_band_count=band_count,
        brokerage_rates=fees, brokerage_if_fully_filled=float(fees @ q),
        expected_brokerage=float(fees @ means))


def print_result(result: OptimizationResult) -> None:
    """Display actual contract targets; never report a tradable OR balance."""
    print("A: contract rows, directed-edge columns")
    print(result.A)
    print("Columns:", "; ".join(f"{i}: {edge.name}" for i, edge in enumerate(result.edges)))
    print("b =", result.b.tolist())
    print("q =", result.q.tolist())
    print("Aq =", (result.A @ result.q).tolist())
    print("\nTAS orders at zero:")
    print(f"{'Order':42} {'Submit':>8} {'Mean fill':>10} {'Full-fill p':>12} {'$/lot':>9}")
    for edge, q, mean, probability, fee in zip(result.edges, result.q,
            result.mean_fills, result.full_fill_probabilities, result.brokerage_rates):
        if q:
            print(f"{edge.name:42} {q:8d} {mean:10.2f} {probability:12.2%} {fee:9.2f}")
    print("m(q) =", np.round(result.mean_fills, 4).tolist())
    print("h_hat(q) =", dict(zip(result.contracts, np.round(result.h_hat, 4).tolist())))
    print(f"Joint full-completion probability: {result.joint_probability:.4%}")
    print(f"Log joint probability: {result.log_joint_probability:.8g}")
    print(f"Expected absolute residual R: {result.expected_residual:.6f} maturity-lots")
    print("E[absolute residual] by maturity =",
          dict(zip(result.contracts, np.round(result.residual_by_maturity, 6).tolist())))
    print(f"Combined score J = R + kappa*(1-P): {result.objective_value:.6f}; kappa={result.kappa:g}")
    method = "EXACT empirical expectation" if result.residual_is_exact else "SAMPLED expectation approximation"
    print(f"Residual integration: {method}; {result.residual_state_count} local outcomes")
    if not result.residual_is_exact:
        print(f"Samples per approximated maturity: {result.residual_samples}; seed={result.seed}")
    print(f"Brokerage if all orders fill (tie-break): ${result.brokerage_if_fully_filled:.2f}")
    print(f"Expected actual brokerage: ${result.expected_brokerage:.2f}")
    print(f"Solver time: {result.solve_seconds:.4f} s; probability bands: {result.probability_band_count}")
    if result.joint_probability == 0:
        print("Selected plan has P=0; its expected partial fills still contribute through R.")


@dataclass
class TASExecutionPlan:
    """Full hedge plus an explicit reserve on maturities without any TAS edge."""
    result: OptimizationResult
    full_target: np.ndarray
    tas_available: np.ndarray
    futures_only: np.ndarray

    @property
    def full_hedge_tas_probability(self) -> float:
        """Unavailable exposure cannot be executed in TAS, regardless of fills."""
        return 0.0 if np.any(self.futures_only) else self.result.joint_probability

    @property
    def expected_futures_balance(self) -> np.ndarray:
        """Mean residual; actual futures orders use actual realized TAS fills."""
        return self.full_target - self.result.h_hat

    @property
    def expected_total_remaining_lots(self) -> float:
        """Expected absolute residual plus the deterministic unavailable reserve."""
        return self.result.expected_residual + float(np.abs(self.futures_only).sum())


def plan_with_unavailable_maturities(
    contracts: Sequence[str], edges: Sequence[Edge], hedge: Sequence[int],
    *, time_limit: float | None = None, **optimizer_options,
) -> TASExecutionPlan:
    """Reserve only isolated maturity targets, then solve the strict TAS model.

    A maturity is TAS-available when it is an endpoint of an actual outright
    or spread edge. Availability is not inferred from the futures listing or
    from volume. An active zero-volume edge remains available: its entire
    assigned target is still submitted. No target is silently dropped.
    If the remaining active network cannot execute its target, still raise;
    this helper does not relax balances in disconnected spread-only groups.
    """
    contracts, edges = tuple(contracts), tuple(edges)
    A = incidence_matrix(contracts, edges)
    raw_b = np.asarray(hedge, dtype=float)
    if (raw_b.shape != (len(contracts),) or not np.all(np.isfinite(raw_b))
            or not np.all(raw_b == np.rint(raw_b))):
        raise ValueError("hedge must contain one finite integer per contract row.")
    full_target = raw_b.astype(np.int64)
    available = np.any(A != 0, axis=1)
    tas_target = np.where(available, full_target, 0)
    reserve = full_target - tas_target
    result = optimize_tas(contracts, edges, tas_target, time_limit=time_limit, **optimizer_options)
    if not np.array_equal(result.A @ result.q + reserve, full_target):
        raise RuntimeError("TAS orders plus unavailable reserve must cover the full hedge.")
    return TASExecutionPlan(result, full_target, available, reserve)


def print_execution_plan(plan: TASExecutionPlan) -> None:
    """Keep the total target and the unavailable reserve visible together."""
    result = plan.result
    print("12-maturity worked example" if len(result.contracts) == 12 else "Execution plan")
    print(f"{len(result.contracts)} futures maturities; "
          f"{np.count_nonzero(plan.full_target)} nonzero hedge targets; "
          f"{np.count_nonzero(plan.tas_available)} TAS-available maturities.")
    print(f"Gross target: {np.abs(plan.full_target).sum():,} maturity-lots. "
          "Buy is positive; sell is negative.")
    print(f"{'Contract':10} {'Target':>8} {'TAS':>7} {'TAS target':>11} "
          f"{'Mean TAS':>10} {'No-TAS reserve':>15} {'Mean residual':>14}")
    for i, contract in enumerate(result.contracts):
        print(f"{contract:10} {plan.full_target[i]:+8d} "
              f"{'open' if plan.tas_available[i] else 'closed':>7} "
              f"{result.b[i]:+11d} {result.h_hat[i]:+10.2f} "
              f"{plan.futures_only[i]:+15d} {plan.expected_futures_balance[i]:+14.2f}")
    print("\nFull target b =", plan.full_target.tolist())
    print("Unavailable reserve r =", plan.futures_only.tolist())
    print("Aq + r =", (result.A @ result.q + plan.futures_only).tolist())
    print(f"Entire hedge completed by TAS: {plan.full_hedge_tas_probability:.4%}")
    print(f"Expected total remaining hedge, including unavailable reserve: {plan.expected_total_remaining_lots:.4f} maturity-lots")
    print("The reserve needs futures; actual futures balance is b - A @ realized_TAS_fills.")
    print("Futures routing, brokerage and slippage are not optimized here.\n")
    print_result(result)
    print("Probability above applies to the TAS-eligible target only; fees are TAS-only.")


def export_network_svg(result, path, *, show_unused=False, reference_quantity=100,
                       full_target=None, tas_available=None, futures_only=None, title=None):
    """Export a self-contained, dependency-free SVG of the optimized network.

    Selected arrows report submitted lots and modeled full-fill probability.
    With more than eight selected edges, arrows carry IDs and the table below
    supplies exact quantities, probabilities, mean fills and brokerage.
    Optional unused edges are faint; their tooltips evaluate probability at
    reference_quantity, never the uninformative p(0)=1.
    Optional full_target/tas_available/futures_only distinguish unavailable TAS
    maturities from zero-sized targets. Unavailable nodes have no TAS edges.
    """
    from html import escape
    from math import atan2, cos, hypot, pi, sin
    from pathlib import Path

    if int(reference_quantity) != reference_quantity or reference_quantity <= 0:
        raise ValueError("reference_quantity must be a positive integer.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contracts, edges = list(result.contracts), list(result.edges)
    selected = [i for i, quantity in enumerate(result.q) if quantity > 0]
    drawn = list(range(len(edges))) if show_unused else selected
    separate_closed = tas_available is not None and not all(tas_available)
    available = [True] * len(contracts) if tas_available is None else list(tas_available)
    target = list(result.b) if full_target is None else list(full_target)
    reserved = [0] * len(contracts) if futures_only is None else list(futures_only)
    if any(len(values) != len(contracts) for values in (available, target, reserved)):
        raise ValueError("Visualization vectors must have one entry per maturity.")
    if any(abs(float(t) - float(b) - float(r)) > 1e-8
           for t, b, r in zip(target, result.b, reserved)):
        raise ValueError("full_target must equal the TAS target plus futures_only.")
    closed = [name for name, is_open in zip(contracts, available) if not is_open]
    active_contracts = [name for name, is_open in zip(contracts, available) if is_open]
    if any(edge.source in closed or edge.destination in closed for edge in edges):
        raise ValueError("TAS-closed maturities cannot have TAS edges.")
    dense = len(selected) > 8 or separate_closed
    rates = list(getattr(result, "brokerage_rates", [0.0] * len(edges)))
    full_brokerage = float(getattr(result, "brokerage_if_fully_filled",
        sum(float(rate) * int(q) for rate, q in zip(rates, result.q))))
    mean_brokerage = float(getattr(result, "expected_brokerage",
        sum(float(rate) * float(fill) for rate, fill in zip(rates, result.mean_fills))))

    def fmt(value, signed=False):
        value = float(value)
        text = f"{value:+,.2f}" if signed else f"{value:,.2f}"
        return text.rstrip("0").rstrip(".")

    def short(value, limit=43):
        value = str(value)
        return value if len(value) <= limit else value[:limit - 1] + "…"

    width = 1100
    graph_top = 240 if separate_closed else 185
    graph_bottom = 820 if separate_closed else (785 if len(contracts) > 3 else 625)
    closed_rows = (len(closed) + 3) // 4
    closed_top = graph_bottom + 120
    notes_top = closed_top + closed_rows * 120 + 10 if separate_closed else graph_bottom + 35
    table_top = notes_top + (73 if separate_closed else 60)
    table_rows = max(len(selected), 1)
    height = table_top + 64 + table_rows * 36 + 72
    nodes = [None] + contracts
    graph_nodes = [None] + active_contracts if separate_closed else nodes
    if separate_closed and len(active_contracts) == 4:
        position = {None: (550, 310), active_contracts[0]: (210, 505),
                    active_contracts[1]: (890, 505), active_contracts[2]: (280, 750),
                    active_contracts[3]: (820, 750)}
    elif len(graph_nodes) == 3:
        # Align the sample's February and March with their natural reading order.
        position = {None: (550, 245), graph_nodes[1]: (220, 530), graph_nodes[2]: (880, 530)}
    elif len(graph_nodes) == 2:
        position = {None: (280, 400), graph_nodes[1]: (820, 400)}
    elif len(graph_nodes) == 1:
        position = {None: (550, 400)}
    else:
        center_y = (graph_top + graph_bottom) / 2 + 20
        radius_x, radius_y = 405, (graph_bottom - graph_top) / 2 - 50
        position = {}
        for i, node in enumerate(graph_nodes):
            angle = -pi / 2 + 2 * pi * i / len(graph_nodes)
            position[node] = (550 + radius_x * cos(angle), center_y + radius_y * sin(angle))

    if separate_closed:
        for i, name in enumerate(closed):
            position[name] = (167.5 + (i % 4) * 255, closed_top + (i // 4) * 120)
    node_width, node_height = (210, 104) if separate_closed else (184, 76)
    colors = {"ink": "#16324a", "secondary": "#526678", "muted": "#9aaab5",
              "line": "#d9e3e9", "teal": "#087f8c", "light": "#edf8f7"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="network-title network-desc">',
           '<title id="network-title">TAS order allocation network</title>',
           '<desc id="network-desc">Contract nodes show required and forecast net trades. '
           'Directed edges show submitted TAS orders. Full data appears in the order table.</desc>',
           '<defs><marker id="selected-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
           'markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#087f8c"/></marker>',
           '<marker id="unused-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
           'markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c7d2db"/></marker></defs>',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<g font-family="Arial, Helvetica, sans-serif">',
           f'<text x="45" y="42" fill="#16324a" font-size="25" font-weight="700">{escape(title or "TAS execution network")}</text>',
           '<text x="45" y="70" fill="#526678" font-size="14">'
           'q = submitted lots · p = probability of full order fill · m = forecast filled lots</text>']

    metric_values = [
        (45, "SELECTED TAS ORDERS: ALL FILL", f"{float(result.joint_probability):.2%}"),
        (310, "EXPECTED TAS RESIDUAL R", f"{fmt(result.expected_residual)} lots"),
        (575, "COMBINED SCORE J", fmt(result.objective_value)),
        (840, "TAS BROKERAGE IF ALL FILL", f"${full_brokerage:,.2f}"),
    ]
    for x, label, value in metric_values:
        svg.append(f'<text x="{x}" y="105" fill="#526678" font-size="11">{label}</text>')
        svg.append(f'<text x="{x}" y="133" fill="#16324a" font-size="23" font-weight="700">{value}</text>')
    integration = 'exact empirical expectation' if result.residual_is_exact else 'sampled expectation approximation'
    summary = (f'J = R + kappa x (1 - P); kappa = {fmt(result.kappa)} lots. '
               f'R: {integration}. Expected TAS brokerage: ${mean_brokerage:,.2f}.')
    svg.append(f'<text x="45" y="163" fill="#526678" font-size="13">{escape(summary)}</text>')
    if separate_closed:
        reserve_text = "; ".join(f"{name} {fmt(lots, True)} lots"
                                 for name, lots in zip(contracts, reserved) if lots)
        message = (f"Futures-only reserve: {reserve_text}. Entire-hedge TAS completion: 0%."
                   if reserve_text else "All hedge targets are reachable through the TAS-open maturities.")
        svg.extend([
            '<rect x="45" y="183" width="1010" height="37" rx="7" fill="#fff6e5"/>',
            f'<text x="60" y="207" fill="#875b10" font-size="14">{escape(message)}</text>',
            f'<text x="45" y="{graph_bottom + 25}" fill="#16324a" font-size="18" font-weight="700">'
            f'TAS closed — {len(closed)} futures maturities, no TAS edges</text>',
        ])

    def boundary(center, toward):
        dx, dy = toward[0] - center[0], toward[1] - center[1]
        scale = min((node_width / 2 + 5) / abs(dx) if dx else float("inf"),
                    (node_height / 2 + 5) / abs(dy) if dy else float("inf"))
        return center[0] + dx * scale, center[1] + dy * scale

    # Draw non-selected structure first. Edge IDs are stable (one-based A columns).
    labels = []
    for e in sorted(drawn, key=lambda i: bool(result.q[i])):
        edge = edges[e]
        source, destination = position[edge.source], position[edge.destination]
        dx, dy = destination[0] - source[0], destination[1] - source[1]
        distance = hypot(dx, dy)
        if not distance:
            continue
        has_reverse = any(other.source == edge.destination and other.destination == edge.source
                          for j, other in enumerate(edges) if j in drawn and j != e)
        bend = (0.15 if has_reverse else 0.045) * distance
        control = ((source[0] + destination[0]) / 2 - dy / distance * bend,
                   (source[1] + destination[1]) / 2 + dx / distance * bend)
        start, end = boundary(source, control), boundary(destination, control)
        active = bool(result.q[e])
        color = colors["teal"] if active else "#c7d2db"
        marker = "selected-arrow" if active else "unused-arrow"
        thickness = 2.3 if active else 1.0
        d = f'M {start[0]:.2f},{start[1]:.2f} Q {control[0]:.2f},{control[1]:.2f} {end[0]:.2f},{end[1]:.2f}'
        if active:
            tooltip = (f"E{e + 1}: {edge.name}; submitted={int(result.q[e])} lots; "
                       f"full-fill probability={float(result.full_fill_probabilities[e]):.4%}; "
                       f"mean fill={float(result.mean_fills[e]):.4f} lots; "
                       f"brokerage=${float(rates[e]):.4f}/lot")
        else:
            probability = edge.forecast.probability(int(reference_quantity))
            tooltip = (f"E{e + 1}: {edge.name}; unused (q=0); "
                       f"full-fill probability at {reference_quantity} lots={probability:.4%}; "
                       f"brokerage=${float(rates[e]):.4f}/lot")
        svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{thickness}" '
                   f'marker-end="url(#{marker})"><title>{escape(tooltip)}</title></path>')
        if active:
            # Vary label location in dense networks to reduce central collisions.
            t = (0.29 + 0.14 * (e % 4)) if dense else 0.5
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
            labels.append((e, x, y, tooltip))

    for e, x, y, tooltip in labels:
        if dense:
            svg.append(f'<g><title>{escape(tooltip)}</title><rect x="{x - 19:.2f}" y="{y - 12:.2f}" '
                       'width="38" height="24" rx="5" fill="#ffffff" stroke="#d9e3e9"/>')
            svg.append(f'<text x="{x:.2f}" y="{y + 4:.2f}" text-anchor="middle" '
                       f'fill="#087f8c" font-size="12" font-weight="700">E{e + 1}</text></g>')
        else:
            label_width = 250
            svg.append(f'<g><title>{escape(tooltip)}</title><rect x="{x - label_width / 2:.2f}" '
                       f'y="{y - 30:.2f}" width="{label_width}" height="60" rx="7" '
                       'fill="#ffffff" stroke="#d9e3e9"/>')
            svg.append(f'<text x="{x:.2f}" y="{y - 9:.2f}" text-anchor="middle" '
                       f'fill="#16324a" font-size="14" font-weight="700">E{e + 1} · q={int(result.q[e])} '
                       f'· p={float(result.full_fill_probabilities[e]):.1%}</text>')
            svg.append(f'<text x="{x:.2f}" y="{y + 14:.2f}" text-anchor="middle" '
                       f'fill="#526678" font-size="13">m={fmt(result.mean_fills[e])} lots '
                       f'· ${float(rates[e]):.2f}/lot</text></g>')

    for node in nodes:
        x, y = position[node]
        label = "OR" if node is None else str(node)
        i = None if node is None else contracts.index(node)
        fill = "#f2f5f7" if node is None or node in closed else colors["light"]
        if i is not None and reserved[i]:
            fill = "#fff6e5"
        svg.append(f'<g><title>{escape(label)}</title><rect x="{x - node_width / 2:.2f}" '
                   f'y="{y - node_height / 2:.2f}" width="{node_width}" height="{node_height}" '
                   f'rx="10" fill="{fill}" stroke="#d9e3e9"/>')
        name_offset = 28 if separate_closed else 15
        svg.append(f'<text x="{x:.2f}" y="{y - name_offset:.2f}" text-anchor="middle" fill="#16324a" '
                   f'font-size="17" font-weight="700">{escape(short(label, 22))}</text>')
        if node is None:
            lines = ["Accounting hub", "No tradable OR position"]
        else:
            lines = [f"Target {fmt(target[i], True)} lots", f"TAS forecast {fmt(result.h_hat[i], True)} lots"]
            if separate_closed:
                lines.append(f"Futures only {fmt(reserved[i], True)} lots" if reserved[i]
                             else ("TAS open" if available[i] else "TAS closed"))
        for offset, text in zip((-3, 17, 37) if separate_closed else (7, 26), lines):
            svg.append(f'<text x="{x:.2f}" y="{y + offset:.2f}" text-anchor="middle" '
                       f'fill="#526678" font-size="12">{escape(text)}</text>')
        svg.append('</g>')

    svg.extend([
        f'<text x="45" y="{notes_top}" fill="#526678" font-size="13">'
        'Arrows follow A: −1 at the source, +1 at the destination. Node quantities exclude OR.</text>',
        f'<text x="45" y="{notes_top + 21}" fill="#526678" font-size="13">'
        + (f'Faint arrows are unused. Hover them for p at {reference_quantity} lots.'
           if show_unused else 'Only selected edges are shown. Edge IDs match columns of A (one-based).') + '</text>',
        f'<text x="45" y="{table_top}" fill="#16324a" font-size="20" font-weight="700">Selected TAS orders</text>',
    ])
    if separate_closed:
        svg.append(f'<text x="45" y="{notes_top + 42}" fill="#526678" font-size="13">'
                   'Full target = submitted TAS exposure + futures-only reserve. The futures balance is not optimized here.</text>')
    headings = [(52, "ID", "start"), (107, "Order", "start"), (580, "Submitted", "end"),
                (695, "p(full)", "end"), (805, "Mean fill", "end"),
                (915, "$/lot", "end"), (1050, "Fee if full", "end")]
    for x, text, anchor in headings:
        svg.append(f'<text x="{x}" y="{table_top + 30}" text-anchor="{anchor}" '
                   f'fill="#526678" font-size="12">{text}</text>')
    for row, e in enumerate(selected):
        y = table_top + 58 + row * 36
        svg.append(f'<line x1="45" x2="1055" y1="{y - 20}" y2="{y - 20}" stroke="#e8eef2"/>')
        cells = [(52, f"E{e + 1}", "start"), (107, short(edges[e].name, 51), "start"),
                 (580, fmt(result.q[e]), "end"),
                 (695, f"{float(result.full_fill_probabilities[e]):.2%}", "end"),
                 (805, fmt(result.mean_fills[e]), "end"),
                 (915, f"${float(rates[e]):.2f}", "end"),
                 (1050, f"${float(rates[e]) * int(result.q[e]):,.2f}", "end")]
        svg.append(f'<g><title>{escape(edges[e].name)}</title>')
        for x, text, anchor in cells:
            svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                       f'fill="#16324a" font-size="13">{escape(text)}</text>')
        svg.append('</g>')
    if not selected:
        svg.append(f'<text x="52" y="{table_top + 61}" fill="#526678" font-size="13">No orders required.</text>')
    svg.append(f'<text x="45" y="{height - 28}" fill="#526678" font-size="12">'
               'Probabilities are volume-based estimates; brokerage rates are supplied per executed edge-lot.</text>')
    svg.extend(['</g>', '</svg>'])
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return path


def twelve_contract_demo(*, all_filled: bool = False,
                         spread_fee: float = 1.5, **optimizer_options) -> TASExecutionPlan:
    """12 listed futures, five nonzero targets, only four TAS-enabled months.

    The 4,000 gross maturity-lots include a +500 Aug target with no TAS route.
    Reserve it explicitly for futures. Only Jan-Apr have TAS outrights and
    the six calendar spreads among them (20 directed routes in total).
    The supplied volumes are synthetic comparable-day observations, already
    seasonally matched. At alpha=.01, accessible lots are 1% of those volumes.
    The all-filled variant sets every route's accessible volume to 4,000;
    it changes forecasts only, never opens TAS on the other eight contracts.
    """
    contracts = [f"{month}27" for month in
                 ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
    full_target = [1200, -900, 600, -800, 0, 0, 0, 500, 0, 0, 0, 0]
    weights = [0.2, 0.5, 0.3]
    outright_volumes = {
        "Jan27": [30000, 60000, 150000],
        "Feb27": [20000, 50000, 120000],
        "Mar27": [10000, 40000, 90000],
        "Apr27": [10000, 50000, 100000],
    }
    spread_volumes = {
        ("Jan27", "Feb27"): [60000, 80000, 120000],
        ("Jan27", "Mar27"): [5000, 10000, 30000],
        ("Jan27", "Apr27"): [20000, 40000, 80000],
        ("Feb27", "Mar27"): [10000, 20000, 60000],
        ("Feb27", "Apr27"): [5000, 10000, 30000],
        ("Mar27", "Apr27"): [40000, 60000, 100000],
    }

    def forecast(volumes):
        return FillForecast.from_volumes([400000] if all_filled else volumes,
                                         [1.0] if all_filled else weights, alpha=0.01)

    outright = {contract: forecast(volumes) for contract, volumes in outright_volumes.items()}
    spreads = {pair: forecast(volumes) for pair, volumes in spread_volumes.items()}
    edges = make_edges(contracts, outright, spreads,
                       outright_brokerage=1.0, spread_brokerage=spread_fee)
    optimizer_options.setdefault("time_limit", 60)
    return plan_with_unavailable_maturities(contracts, edges, full_target, **optimizer_options)


def toy_problem() -> tuple[list[str], list[Edge], list[int]]:
    """Earlier two-contract example, now with ALL SIX directions available.

    The 20/50/30 weights are synthetic, already aggregated historical weights.
    They are marginal distributions; matching row indices across instruments
    does NOT create correlated future fills in the optimization.
    """
    contracts = ["Feb27", "Mar27"]
    weights = [0.2, 0.5, 0.3]
    forecasts = {
        "Feb27": FillForecast.from_volumes([5000, 6000, 10000], weights),
        "Mar27": FillForecast.from_volumes([4000, 6000, 9000], weights),
    }
    spreads = {("Feb27", "Mar27"):
               FillForecast.from_volumes([2000, 4000, 7000], weights)}
    return contracts, make_edges(contracts, forecasts, spreads), [100, -100]


def six_contract_demo(**optimizer_options) -> OptimizationResult:
    """Synthetic 4,000 gross lots over six contracts and all 42 directions."""
    rng = np.random.default_rng(2026)
    contracts = [f"M{i + 1}" for i in range(6)]
    # Positive probability for large orders, plus lower-liquidity outcomes.
    weights = [0.2, 0.5, 0.3]
    outright = {name: FillForecast.from_volumes(
        [int(rng.integers(20000, 40001)), int(rng.integers(50000, 90001)), 220000], weights)
        for name in contracts}
    spreads = {(contracts[i], contracts[j]): FillForecast.from_volumes(
        [int(rng.integers(5000, 15001)), int(rng.integers(15000, 40001)), 120000], weights)
        for i in range(6) for j in range(i + 1, 6)}
    optimizer_options.setdefault("time_limit", 60)
    return optimize_tas(contracts, make_edges(contracts, outright, spreads),
                        [800, -700, 600, -500, 600, -800], **optimizer_options)


def all_filled_demo(spread_fee: float = 1.5, **optimizer_options) -> OptimizationResult:
    """Every edge supports the entire target with probability 1.

    At $1 per outright lot and $1.50 per spread lot, 100 Feb/Mar spreads
    cost $150, versus $200 for the two outright orders. Setting spread_fee
    to $3 instead makes the two outright orders the cheaper allocation.
    """
    contracts = ["Feb27", "Mar27"]
    certain = FillForecast.from_volumes([20000], [1.0])
    edges = make_edges(contracts, {name: certain for name in contracts},
                       {("Feb27", "Mar27"): certain},
                       outright_brokerage=1.0, spread_brokerage=spread_fee)
    return optimize_tas(contracts, edges, [100, -100], **optimizer_options)


def self_test() -> None:
    """Focused model tests, including brute-force comparison on tiny networks."""
    checks = 0
    contracts, edges, b = toy_problem()
    result = optimize_tas(contracts, edges, b)
    assert np.array_equal(result.q, [60, 0, 0, 60, 40, 0])
    assert np.allclose(result.mean_fills, [58, 0, 0, 56, 36, 0])
    assert np.allclose(result.h_hat, [94, -92])
    assert np.isclose(result.joint_probability, 0.512)
    assert result.brokerage_if_fully_filled == 200
    assert np.isclose(result.expected_brokerage, 186)
    assert np.isclose(result.expected_residual, 14)
    assert np.isclose(result.objective_value, 23.76) and result.residual_is_exact
    checks += 1

    low = FillForecast.from_volumes([5000], [1])
    single = optimize_tas(["Feb27"], [Edge("Buy Feb27", None, "Feb27", low, 3)], [100])
    assert single.q.tolist() == [100] and single.mean_fills.tolist() == [50]
    assert single.joint_probability == 0 and single.expected_residual == 50
    assert single.brokerage_if_fully_filled == 300 and single.expected_brokerage == 150
    checks += 1
    zero = optimize_tas(contracts, edges, [0, 0])
    assert not zero.q.any() and zero.joint_probability == 1 and zero.objective_value == 0
    checks += 1

    cheap_spread = all_filled_demo(1.5)
    assert cheap_spread.q.tolist() == [0, 0, 0, 0, 100, 0]
    assert cheap_spread.joint_probability == 1
    assert cheap_spread.brokerage_if_fully_filled == cheap_spread.expected_brokerage == 150
    expensive_spread = all_filled_demo(3.0)
    assert expensive_spread.q.tolist() == [100, 0, 0, 100, 0, 0]
    assert expensive_spread.joint_probability == 1
    assert expensive_spread.brokerage_if_fully_filled == 200
    checks += 2

    certain = FillForecast([100], [1])
    almost_certain = FillForecast([0, 100], [0.01, 0.99])
    priority_edges = [Edge("Certain expensive A", None, "A", certain, 10),
                      Edge("Cheaper B", None, "B", almost_certain, 0.1),
                      Edge("B to A", "B", "A", certain, 0.1)]
    priority = optimize_tas(["A", "B"], priority_edges, [100, 0])
    assert priority.q.tolist() == [100, 0, 0]
    assert priority.joint_probability == 1 and priority.brokerage_if_fully_filled == 1000
    checks += 1

    empty = FillForecast([0], [1])
    zero_edges = [Edge("Direct A", None, "A", empty, 1),
                  Edge("Via B", None, "B", empty, 0.1),
                  Edge("B to A", "B", "A", empty, 0.1)]
    zero_fee_choice = optimize_tas(["A", "B"], zero_edges, [100, 0])
    assert zero_fee_choice.joint_probability == 0 and zero_fee_choice.expected_residual == 100
    assert zero_fee_choice.q.tolist() == [0, 100, 100]
    assert np.isclose(zero_fee_choice.brokerage_if_fully_filled, 20)
    checks += 1

    try:
        Edge("Bad fee", None, "A", certain, -1)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Negative brokerage must be rejected.")

    fractional = FillForecast([0, 1.5, 1.5, 3], [0.1, 0.2, 0.2, 0.5])
    assert np.isclose(fractional.probability(1), 0.9)
    assert np.isclose(fractional.probability(2), 0.5)
    assert np.isclose(fractional.mean_fill(2), 1.6)
    for q in range(6):
        bands = [p for lo, hi, p in fractional.positive_bands(5) if lo <= q <= hi]
        assert np.isclose(bands[0] if bands else 0, fractional.probability(q))
    checks += 1

    rows = [HistoricalVolume(date(2025, 11, 10), ((2026, 2),), 50, 10000),
            HistoricalVolume(date(2024, 11, 10), ((2025, 2),), 53, 0),
            HistoricalVolume(date(2025, 11, 10), ((2026, 3),), 50, 999999),
            HistoricalVolume(date(2025, 11, 10), ((2026, 2),), 56, 999999),
            HistoricalVolume(date(2025, 11, 10), ((2026, 2),), 50, None),
            HistoricalVolume(date(2025, 11, 10), ((2026, 2),), 50, 999999, False),
            HistoricalVolume(date(2026, 11, 11), ((2026, 2),), 50, 999999)]
    seasonal = seasonal_forecast(rows, ((2027, 2),), 50, date(2026, 11, 11))
    assert len(seasonal.weights) == 2
    assert np.isclose(seasonal.probability(100), 1 / 1.8)
    checks += 1

    invalid_lifecycle = [HistoricalVolume(date(2025, 11, 10), ((2026, 2),), np.nan, 10000)]
    for bad_rows, target_distance in [(invalid_lifecycle, 50), (rows, np.nan)]:
        try:
            seasonal_forecast(bad_rows, ((2027, 2),), target_distance, date(2026, 11, 11))
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid lifecycle distances must not silently match.")

    cross_year = [HistoricalVolume(date(2024, 6, 1), ((2024, 12), (2025, 1)), 50, 10000),
                  HistoricalVolume(date(2024, 6, 1), ((2024, 12), (2024, 1)), 50, 999999)]
    matched = seasonal_forecast(cross_year, ((2026, 12), (2027, 1)), 50, date(2026, 6, 1))
    assert len(matched.weights) == 1 and matched.mean_fill(200) == 100
    checks += 1

    try:
        optimize_tas(["A", "B"], [Edge("Buy A", None, "A", low)], [0, 1])
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Disconnected hedge should be infeasible.")

    plan = twelve_contract_demo()
    assert len(plan.result.contracts) == 12
    assert np.count_nonzero(plan.full_target) == 5 and plan.tas_available.sum() == 4
    assert plan.result.A.shape == (12, 20)
    assert np.array_equal(plan.futures_only, [0, 0, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0])
    assert np.array_equal(plan.result.A @ plan.result.q + plan.futures_only, plan.full_target)
    assert np.isclose(plan.result.joint_probability, 0.8)
    assert plan.full_hedge_tas_probability == 0
    assert np.isclose(plan.result.brokerage_if_fully_filled, 2800)
    assert np.isclose(plan.result.expected_residual, 40)
    assert np.isclose(plan.result.objective_value, 44) and plan.result.residual_is_exact
    assert np.allclose(plan.result.h_hat, [1180, -900, 600, -780, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(plan.expected_futures_balance,
                       [20, 0, 0, -20, 0, 0, 0, 500, 0, 0, 0, 0])
    assert not np.any(plan.result.A[~plan.tas_available])
    checks += 1
    try:
        optimize_tas(plan.result.contracts, plan.result.edges, plan.full_target)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("The full five-maturity hedge cannot be submitted in TAS.")

    twelve_certain = twelve_contract_demo(all_filled=True)
    assert twelve_certain.result.joint_probability == 1
    assert twelve_certain.full_hedge_tas_probability == 0
    assert twelve_certain.result.brokerage_if_fully_filled == 2650
    assert np.allclose(twelve_certain.expected_futures_balance, twelve_certain.futures_only)
    expensive_twelve = twelve_contract_demo(all_filled=True, spread_fee=3)
    assert expensive_twelve.result.brokerage_if_fully_filled == 3500
    assert not any(size for edge, size in zip(expensive_twelve.result.edges, expensive_twelve.result.q)
                   if edge.source is not None and edge.destination is not None)
    checks += 2

    no_tas = plan_with_unavailable_maturities(["A", "B"], [], [100, -50])
    assert no_tas.result.q.size == 0 and no_tas.result.joint_probability == 1
    assert no_tas.full_hedge_tas_probability == 0
    assert no_tas.futures_only.tolist() == [100, -50]
    no_target = plan_with_unavailable_maturities(["A", "B"], [], [0, 0])
    assert no_target.full_hedge_tas_probability == 1
    checks += 2

    near = optimize_tas(["A"], [Edge("Buy A", None, "A", FillForecast([99], [1]), 1)], [100])
    poor = optimize_tas(["A"], [Edge("Buy A", None, "A", FillForecast([80], [1]), 1)], [100])
    assert near.q.tolist() == [100] and near.joint_probability == poor.joint_probability == 0
    assert near.objective_value == 21 and poor.objective_value == 40
    checks += 1

    # Integer order size is one; fractional accessible liquidity lets this
    # test isolate the exact 99% vs 90%/100% trade-off without order splitting.
    tradeoff = [Edge("Nearly full", None, "A", FillForecast([0.99], [1]), 1),
                Edge("Sometimes full", None, "A", FillForecast([0.9, 1], [0.9, 0.1]), 1)]
    low_kappa = optimize_tas(["A"], tradeoff, [1], kappa=0.2)
    high_kappa = optimize_tas(["A"], tradeoff, [1], kappa=2)
    assert low_kappa.q.tolist() == [1, 0] and np.isclose(low_kappa.objective_value, .21)
    assert high_kappa.q.tolist() == [0, 1] and np.isclose(high_kappa.objective_value, 1.89)
    assert optimize_tas(["A"], tradeoff, [1], kappa=0).q.tolist() == [1, 0]
    checks += 2

    equal_score = [Edge("More partial, dearer", None, "A", FillForecast([.9], [1]), 2),
                   Edge("More variable, cheaper", None, "A", FillForecast([.6, 1], [.5, .5]), 1)]
    tied = optimize_tas(["A"], equal_score, [1], kappa=.2)
    assert tied.q.tolist() == [0, 1] and np.isclose(tied.objective_value, .3)
    assert tied.brokerage_if_fully_filled == 1
    checks += 1

    partial = FillForecast([90, 100], [.5, .5])
    path = [Edge("Buy B", None, "B", partial, 1), Edge("B to A", "B", "A", partial, 1)]
    path_result = optimize_tas(["A", "B"], path, [100, 0])
    assert np.isclose(path_result.expected_residual, 10)
    assert np.isclose(np.abs(path_result.b - path_result.h_hat).sum(), 5)
    assert np.isclose(path_result.joint_probability, .25)
    checks += 1
    sampled1 = optimize_tas(["A", "B"], path, [100, 0], exact_state_limit=0, residual_samples=1024, seed=9)
    sampled2 = evaluate_allocation(["A", "B"], path, [100, 0], [100, 100],
                                  exact_state_limit=0, residual_samples=1024, seed=9)
    assert not sampled1.residual_is_exact and not sampled2["residual_is_exact"]
    assert sampled1.expected_residual == sampled2["expected_residual"]
    assert sampled1.joint_probability == .25  # P remains exact even when R is sampled.
    assert abs(sampled1.expected_residual - 10) < 1
    checks += 1
    for bad in [-1, np.nan, np.inf]:
        try:
            optimize_tas(["A"], tradeoff, [1], kappa=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid kappa must be rejected.")
    checks += 1

    # Independent reference calculation enumerates outcomes of selected
    # directed edges; it does not use the solver's integration/grouping code.
    def brute_score(edge_list, A, b, q, kappa):
        selected = np.flatnonzero(q)
        probability = float(np.prod([edge.forecast.probability(int(size))
                                     for edge, size in zip(edge_list, q)]))
        residual = 0.0
        for outcomes in product(*(range(len(edge_list[e].forecast.weights)) for e in selected)):
            weight, fills = 1.0, []
            for e, k in zip(selected, outcomes):
                forecast = edge_list[e].forecast
                weight *= forecast.weights[k]
                fills.append(min(q[e], forecast.accessible_volume[k]))
            residual += weight * np.abs(b - A[:, selected] @ fills).sum()
        return residual + kappa * (1 - probability)

    rng = np.random.default_rng(17)
    for _ in range(8):
        tiny_edges = []
        endpoints = [(None, "A"), ("A", None), (None, "B"), ("B", None),
                     ("B", "A"), ("A", "B")]
        for i, (source, destination) in enumerate(endpoints):
            forecast = FillForecast(rng.choice([0, 0.5, 1, 1.5, 2, 3], 4),
                                    rng.uniform(0.1, 1, 4))
            tiny_edges.append(Edge(str(i), source, destination, forecast,
                                   float(rng.integers(0, 5)) / 2))
        target = np.array([2, -1])
        A = incidence_matrix(["A", "B"], tiny_edges)
        kappa = float(rng.choice([0, .5, 2, 20]))
        best, best_fee = np.inf, np.inf
        for allocation in product(range(3), repeat=6):
            q = np.array(allocation)
            if np.array_equal(A @ q, target) and _is_acyclic(tiny_edges, q):
                score = brute_score(tiny_edges, A, target, q, kappa)
                fee = sum(edge.brokerage_per_lot * int(size)
                          for edge, size in zip(tiny_edges, q))
                if score < best - 1e-10:
                    best, best_fee = score, fee
                elif np.isclose(score, best, rtol=1e-12, atol=1e-10):
                    best_fee = min(best_fee, fee)
        solved = optimize_tas(["A", "B"], tiny_edges, target, kappa=kappa)
        assert np.array_equal(solved.A @ solved.q, target)
        assert np.isclose(solved.objective_value, best, rtol=1e-8, atol=1e-6)
        assert np.isclose(solved.brokerage_if_fully_filled, best_fee)
        assert _is_acyclic(tiny_edges, solved.q)
        checks += 1
    print(f"Passed {checks} model checks, including 8 exhaustive tiny-network comparisons.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="run model verification checks")
    demos = parser.add_mutually_exclusive_group()
    demos.add_argument("--twelve-contract-demo", action="store_true", help="12 maturities, targets on five, TAS on four (default)")
    demos.add_argument("--two-contract-demo", action="store_true", help="run the earlier two-contract toy example")
    demos.add_argument("--six-contract-demo", action="store_true", help="solve synthetic 4,000-lot example")
    demos.add_argument("--all-filled-demo", action="store_true", help="12-maturity example with 100% fills on the four TAS-enabled maturities")
    parser.add_argument("--plot", default="tas_network.svg", help="SVG output path (default: tas_network.svg)")
    parser.add_argument("--no-plot", action="store_true", help="skip SVG output")
    parser.add_argument("--all-edges", action="store_true", help="also draw unused routes with reference-size tooltips")
    parser.add_argument("--reference-quantity", type=int, default=100, help="order size for unused-edge probability tooltips")
    parser.add_argument("--kappa", type=float, default=20.0, help="full-completion penalty in residual lots (illustrative default: 20)")
    parser.add_argument("--exact-state-limit", type=int, default=4096, help="maximum exact liquidity combinations per maturity")
    parser.add_argument("--residual-samples", type=int, default=2048, help="draws per maturity when exact integration exceeds the limit")
    parser.add_argument("--seed", type=int, default=2026, help="reproducible residual sampling seed")
    parser.add_argument("--time-limit", type=float, default=60, help="seconds per solver phase; timeouts raise")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        plan = None
        settings = dict(kappa=args.kappa, exact_state_limit=args.exact_state_limit,
                        residual_samples=args.residual_samples, seed=args.seed, time_limit=args.time_limit)
        if args.six_contract_demo:
            result = six_contract_demo(**settings)
        elif args.two_contract_demo:
            result = optimize_tas(*toy_problem(), **settings)
        else:
            plan = twelve_contract_demo(all_filled=args.all_filled_demo, **settings)
            result = plan.result
        if plan is None:
            print_result(result)
        else:
            print_execution_plan(plan)
        if not args.no_plot:
            context = {} if plan is None else dict(full_target=plan.full_target,
                                                   tas_available=plan.tas_available,
                                                   futures_only=plan.futures_only)
            output = export_network_svg(result, args.plot, show_unused=args.all_edges,
                                        reference_quantity=args.reference_quantity, **context)
            print(f"Network visualization: {output.resolve()}")
