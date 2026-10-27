"""Firms: one class, the same rules for all 120, different states.

Production needs two complementary tasks. Task A is routine and partly
automatable; task B is not. A machine adds A capacity, but never more than the
automatable share θ of A work: the remaining (1 − θ) has to be done by people.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import payroll

A, B = 0, 1
TOL = 1e-9


def ceil_tol(x):
    """Ceiling that ignores floating noise: 2.0000000001 posts is 2 posts."""
    return np.maximum(np.ceil(np.asarray(x, dtype=float) - TOL), 0).astype(int)


def effective_machine(machine, staff_a, theta):
    """Machine capacity the A staff can actually use: at most θ of all A work."""
    if theta >= 1:
        return np.asarray(machine, dtype=float)
    return np.minimum(machine, theta / (1 - theta) * np.asarray(staff_a, dtype=float))


def capacity(staff_a, staff_b, machine, a, b, theta):
    """q = min((L_A + M_eff) / a, L_B / b); an inactive machine counts as zero."""
    return np.minimum((staff_a + effective_machine(machine, staff_a, theta)) / a, staff_b / b)


def requirements(q, machine, a, b, theta):
    """Smallest whole staff that makes output q feasible.

    L_A must cover what the machine cannot do (a·q − M) and the part of A work
    that cannot be automated at all ((1 − θ)·a·q). Planner, layoffs and
    investment projections all go through this one function.
    """
    q = np.asarray(q, dtype=float)
    need_b = ceil_tol(b * q)
    if machine > 0:
        need_a = ceil_tol(np.maximum(a * q - machine, (1 - theta) * a * q))
    else:
        need_a = ceil_tol(a * q)
    if need_a.ndim == 0:
        return int(need_a), int(need_b)
    return need_a, need_b


class BandLadder:
    """Order in which the posts of one task fill the wage bands.

    The k-th post goes to the band furthest below its share of k posts (ties to
    the lower band). The band mix of n posts then depends on n alone: a firm
    that hires, loses and rehires keeps its wage structure, and the cost of n
    posts is a lookup rather than a guess from an average wage.
    """

    def __init__(self, shares):
        self.shares = np.asarray(shares, dtype=float)
        self.sequence = []
        self._counts = [(0,) * len(self.shares)]

    def extend(self, n):
        shares = self.shares.tolist()
        bands = range(len(shares))
        counts = list(self._counts[-1])
        while len(self.sequence) < n:
            k = len(self.sequence) + 1
            g = max(bands, key=lambda j: (shares[j] * k - counts[j], -j))
            counts[g] += 1
            self.sequence.append(g)
            self._counts.append(tuple(counts))

    def counts(self, n):
        self.extend(n)
        return np.array(self._counts[n])

    def cumulative(self, per_band, n):
        """Amounts of the first 0, 1, …, n posts."""
        self.extend(n)
        values = np.asarray(per_band, dtype=float)[np.array(self.sequence[:n], dtype=int)]
        return np.concatenate([[0.0], np.cumsum(values)])


@dataclass
class CostTables:
    """Cumulative quarterly amounts of the first n posts, by task."""

    cost: list
    gross: list
    contributions: list
    reduction: list


def accounts(price, output, unit_ci, fixed_cost, maintenance, labor_cost):
    """Operating account of a quarter. CI is split into a variable and a fixed part."""
    revenue = price * output
    variable_ci = unit_ci * output
    value_added = revenue - variable_ci - fixed_cost - maintenance
    return {
        "revenue": revenue,
        "variable_ci": variable_ci,
        "value_added": value_added,
        "surplus": value_added - labor_cost,
    }


@dataclass
class Firm:
    id: int
    market: int
    size: int
    replicate: int
    profile: payroll.Profile
    a: float                  # A posts per unit of output
    b: float                  # B posts per unit of output
    theta: float              # automatable share of A work
    ladders: list             # one BandLadder per task
    staff: np.ndarray         # posts by task, during the current quarter
    price: float
    ref_price: float
    markup: float             # over the full unit cost, calibrated on the initial state
    unit_ci: float            # variable intermediate consumption per unit
    fixed_cost: float         # fixed external charges per quarter
    cash: float
    machine_capacity: float   # size of the firm's single technology option, in A posts
    initial_output: float
    target_margin: float = 0.0  # EBE / VA drawn for the initial state
    demand_history: list = field(default_factory=list)
    band_costs: payroll.BandCosts | None = None
    tables: CostTables | None = None

    @property
    def cells(self):
        """Posts by task × band: the eight numbers of the spec."""
        return np.stack([self.ladders[t].counts(int(self.staff[t])) for t in (A, B)])

    def set_costs(self, band_costs):
        self.band_costs = band_costs
        self.tables = None
        self.table("cost", A, 2 * int(self.staff.max()) + 10)

    def table(self, name, task, n):
        """Cumulative amounts for at least n posts of a task, extended on demand."""
        if self.tables is None or len(self.tables.cost[task]) <= n:
            size = n if self.tables is None else max(2 * n, len(self.tables.cost[A]) - 1)
            bc = self.band_costs
            self.tables = CostTables(*(
                [ladder.cumulative(getattr(bc, attr), size) for ladder in self.ladders]
                for attr in ("cost", "gross", "contributions", "reduction")
            ))
        return getattr(self.tables, name)[task]

    def amount(self, name, staff=None):
        """Quarterly amount for a staff vector, e.g. amount("cost") is the labor cost."""
        staff = self.staff if staff is None else staff
        return sum(self.table(name, t, int(staff[t]))[int(staff[t])] for t in (A, B))
