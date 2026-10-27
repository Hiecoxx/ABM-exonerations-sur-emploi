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
    credit_ceiling: float = 0.0  # loan + overdraft may not exceed it
    cash_buffer: float = 0.0     # cash the firm keeps before borrowing
    demand_history: list = field(default_factory=list)
    due: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))  # notices leaving at end of quarter
    planned_output: float = 0.0
    distress: int = 0            # consecutive quarters beyond the credit ceiling
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

    def financing_available(self):
        """What the firm could still pay out: cash above its buffer, then its credit line.

        Negative cash is already a draw on the line, so it is not counted twice.
        """
        return self.cash - self.cash_buffer + self.credit_ceiling

    def debt(self):
        """Overdraft."""
        return max(0.0, -self.cash)


@dataclass(frozen=True)
class Plan:
    """Staffing decided at the end of quarter t."""

    output: float           # planned output
    target: np.ndarray      # staff from t+2, once notices have run out
    next_staff: np.ndarray  # staff in t+1
    hires: np.ndarray       # vacancies posted now
    cancel: np.ndarray      # due notices withdrawn
    execute: np.ndarray     # due notices carried out now
    notify: np.ndarray      # new notices, leaving at the end of t+1
    value: float


def candidate_outputs(demand, machines, a, b, theta):
    """Output levels worth comparing: each point where one more A or B post is needed, and demand."""
    levels = [np.arange(int(ceil_tol(b * demand)) + 1) / b, [demand]]
    staff_a = np.arange(int(ceil_tol(a * demand)) + 1, dtype=float)
    for m in {machines[0], machines[min(1, len(machines) - 1)]}:
        levels.append((staff_a + effective_machine(m, staff_a, theta)) / a)
    return np.unique(np.clip(np.concatenate(levels), 0.0, demand))


def plan_staff(firm, demand, margin, machines, adjustment, allow_hiring=True):
    """Choose output, then the bundle of A and B posts it takes, over a short horizon.

    Every candidate output, from zero to expected demand, comes with the staff
    `requirements` gives for it, so complementary posts are valued together.
    Each candidate is valued with the real dates: hires work from t+1; notified
    posts are still paid, and still produce, in t+1 and are gone from t+2; each
    quarter counts only the machine capacity active in it (`machines[k]` for
    t+1+k), and staff from t+2 is sized on the capacity active in t+2. Value = Σ [(p − c_v) × min(demand, capacity) − labor cost], minus a
    hiring cost per hire and severance per layoff. Quits left unreplaced cost
    nothing, so a shrinking firm tends to shrink by attrition first. A level
    below demand is allowed: nothing forces a firm to keep losing money.
    """
    a, b, theta = firm.a, firm.b, firm.theta
    hiring_months, severance_months = adjustment
    q = candidate_outputs(demand, machines, a, b, theta)
    soon = np.stack(requirements(q, machines[0], a, b, theta))   # staff needed in t+1
    m_later = machines[min(1, len(machines) - 1)]
    later = np.stack(requirements(q, m_later, a, b, theta))       # staff needed from t+2
    staff = firm.staff[:, None]
    due = firm.due[:, None]
    next_staff = np.maximum(soon, staff - due)
    hires = np.maximum(0, soon - staff)
    cancel = np.minimum(due, np.maximum(0, soon - (staff - due)))
    execute = due - cancel
    target = np.minimum(next_staff, later)
    notify = next_staff - target

    top = int(max(next_staff.max(), staff.max())) + 1
    cost = [firm.table("cost", t, top) for t in (A, B)]
    gross = [firm.table("gross", t, top) for t in (A, B)]
    value = np.zeros(len(q))
    for k, m in enumerate(machines):
        s = next_staff if k == 0 else target
        out = np.minimum(demand, capacity(s[A], s[B], m, a, b, theta))
        value += margin * out - cost[A][s[A]] - cost[B][s[B]]
    for t in (A, B):
        s0 = int(firm.staff[t])
        value -= hiring_months / 3 * (gross[t][s0 + hires[t]] - gross[t][s0])
        value -= severance_months / 3 * (gross[t][s0] - gross[t][s0 - execute[t]] + gross[t][next_staff[t]] - gross[t][target[t]])
    if not allow_hiring:
        value = np.where(hires.sum(axis=0) > 0, -np.inf, value)

    best = value.max()
    i = int(np.flatnonzero(value >= best - 1e-9 * max(1.0, abs(best)))[-1])
    plan = Plan(float(q[i]), target[:, i], next_staff[:, i], hires[:, i], cancel[:, i], execute[:, i],
                notify[:, i], float(value[i]))
    if (capacity(*plan.next_staff, machines[0], a, b, theta) < plan.output - 1e-6
            or capacity(*plan.target, m_later, a, b, theta) < plan.output - 1e-6):
        raise RuntimeError(f"firm {firm.id}: planned staff cannot produce the planned output")
    return plan


def price_target(firm, plan, maintenance):
    """(1 + markup) × full unit cost at the planned staff; unchanged if nothing is planned."""
    if plan.output <= 0:
        return firm.price
    labor = firm.amount("cost", plan.target)
    return (1 + firm.markup) * (firm.unit_ci + (labor + firm.fixed_cost + maintenance) / plan.output)
