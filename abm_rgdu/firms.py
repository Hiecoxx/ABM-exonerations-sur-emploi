"""Firms: one class, the same rules for all 120, different states.

Production needs two complementary tasks. Task A is routine and partly
automatable; task B is not. A machine adds A capacity, but never more than the
automatable share θ of A work: the remaining (1 − θ) has to be done by people.
"""

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from . import markets, payroll

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
class Machine:
    """The firm's single technology option, once bought.

    Ordered at the end of quarter `ordered_at`, installing until `active_from`.
    Capacity, maintenance and the transition cost all start in `active_from`.
    """

    capacity: float     # A work it can do, in posts
    cost: float         # K, paid once, at the order date
    maintenance: float  # per quarter, from activation
    transition: float   # once, in the first active quarter
    ordered_at: int
    active_from: int
    origin: str = "new"  # or "pre-existing"

    def active(self, t):
        return t >= self.active_from


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
    machine: Machine | None = None
    loan: float = 0.0            # investment loan outstanding
    loan_payment: float = 0.0    # principal due each quarter
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
        return self.cash - self.cash_buffer + self.credit_ceiling - self.loan

    def debt(self):
        """Investment loan plus overdraft: what the credit ceiling bounds."""
        return self.loan + max(0.0, -self.cash)

    def machine_at(self, t):
        """Machine capacity usable in quarter t: zero while it is not installed."""
        return self.machine.capacity if self.machine is not None and self.machine.active(t) else 0.0

    def maintenance_at(self, t):
        return self.machine.maintenance if self.machine is not None and self.machine.active(t) else 0.0


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


def candidate_outputs(demand, soon, later, a, b, theta):
    """Output levels worth comparing: each point where one more A or B post is needed, and demand.

    Duplicates do no harm and are left in; levels above demand are cut back to it.
    """
    staff_a = np.arange(int(ceil_tol(a * demand)) + 1, dtype=float)
    levels = [np.arange(int(ceil_tol(b * demand)) + 1) / b, [demand], staff_a / a if soon == 0 else
              (staff_a + effective_machine(soon, staff_a, theta)) / a]
    if later != soon:
        levels.append((staff_a + effective_machine(later, staff_a, theta)) / a)
    return np.minimum(np.concatenate(levels), demand)


def _need_a(q, machine, a, theta):
    x = np.maximum(a * q - machine, (1 - theta) * a * q) if machine > 0 else a * q
    return np.maximum(np.ceil(x - TOL), 0).astype(np.int64)


def _capacity(staff_a, staff_b, machine, a, b, theta):
    """`capacity` for arrays of whole staff, without the conversions."""
    if machine > 0:
        staff_m = staff_a + (np.minimum(machine, theta / (1 - theta) * staff_a) if theta < 1 else machine)
        return np.minimum(staff_m / a, staff_b / b)
    return np.minimum(staff_a / a, staff_b / b)


def plan_staff(firm, demand, margin, machines, adjustment, allow_hiring=True):
    """Choose output, then the bundle of A and B posts it takes, over a short horizon.

    Every candidate output, from zero to expected demand, comes with the staff
    `requirements` gives for it, so complementary posts are valued together.
    Each candidate is valued with the real dates: hires work from t+1; notified
    posts are still paid, and still produce, in t+1 and are gone from t+2; each
    quarter counts only the machine capacity active in it (`machines[k]` for
    t+1+k), and staff from t+2 is sized on the capacity active in t+2.
    Value = Σ [(p − c_v) × min(demand, capacity) − labor cost], minus a hiring
    cost per hire and severance per layoff. Quits left unreplaced cost nothing,
    so a shrinking firm tends to shrink by attrition first. A level below
    demand is allowed: nothing forces a firm to keep losing money. Among equal
    values the larger output wins.
    """
    a, b, theta = firm.a, firm.b, firm.theta
    hiring_months, severance_months = adjustment
    m_soon = machines[0]
    m_later = machines[1] if len(machines) > 1 else m_soon
    q = candidate_outputs(demand, m_soon, m_later, a, b, theta)
    need_b = np.maximum(np.ceil(b * q - TOL), 0).astype(np.int64)
    need_a = _need_a(q, m_soon, a, theta)                                          # in t+1
    need_a_later = need_a if m_later == m_soon else _need_a(q, m_later, a, theta)  # from t+2

    sa, sb = int(firm.staff[A]), int(firm.staff[B])
    da, db = int(firm.due[A]), int(firm.due[B])
    next_a, next_b = np.maximum(need_a, sa - da), np.maximum(need_b, sb - db)
    hires_a, hires_b = np.maximum(need_a - sa, 0), np.maximum(need_b - sb, 0)
    cancel_a = np.minimum(np.maximum(need_a - (sa - da), 0), da)
    cancel_b = np.minimum(np.maximum(need_b - (sb - db), 0), db)
    exec_a, exec_b = da - cancel_a, db - cancel_b
    target_a, target_b = np.minimum(next_a, need_a_later), need_b

    top = max(int(next_a.max()), int(next_b.max()), sa, sb) + 1
    cost_a, cost_b = firm.table("cost", A, top), firm.table("cost", B, top)
    gross_a, gross_b = firm.table("gross", A, top), firm.table("gross", B, top)
    value = margin * np.minimum(_capacity(next_a, next_b, m_soon, a, b, theta), demand) - cost_a[next_a] - cost_b[next_b]
    later_cost = cost_a[target_a] + cost_b[target_b]
    counts = {}
    for m in machines[1:]:
        counts[m] = counts.get(m, 0) + 1
    for m, k in counts.items():
        value += k * (margin * np.minimum(_capacity(target_a, target_b, m, a, b, theta), demand) - later_cost)
    value -= hiring_months / 3 * (gross_a[sa + hires_a] + gross_b[sb + hires_b] - gross_a[sa] - gross_b[sb])
    value -= severance_months / 3 * (
        gross_a[sa] - gross_a[sa - exec_a] + gross_a[next_a] - gross_a[target_a]
        + gross_b[sb] - gross_b[sb - exec_b] + gross_b[next_b] - gross_b[target_b]
    )
    if not allow_hiring:
        value = np.where(hires_a + hires_b > 0, -np.inf, value)

    best = value.max()
    i = int(np.argmax(np.where(value >= best - 1e-9 * max(1.0, abs(best)), q, -1.0)))
    plan = Plan(float(q[i]), np.array([target_a[i], target_b[i]]), np.array([next_a[i], next_b[i]]),
                np.array([hires_a[i], hires_b[i]]), np.array([cancel_a[i], cancel_b[i]]),
                np.array([exec_a[i], exec_b[i]]), np.array([next_a[i] - target_a[i], next_b[i] - target_b[i]]), float(value[i]))
    if (_capacity(plan.next_staff[A], plan.next_staff[B], m_soon, a, b, theta) < plan.output - 1e-6
            or _capacity(plan.target[A], plan.target[B], m_later, a, b, theta) < plan.output - 1e-6):
        raise RuntimeError(f"firm {firm.id}: planned staff cannot produce the planned output")
    return plan


def price_target(firm, plan, maintenance):
    """(1 + markup) × full unit cost at the planned staff; unchanged if nothing is planned."""
    if plan.output <= 0:
        return firm.price
    labor = firm.amount("cost", plan.target)
    return (1 + firm.markup) * (firm.unit_ci + (labor + firm.fixed_cost + maintenance) / plan.output)


def new_machine(firm, t, unit_cost, p):
    tech = p["technology"]
    cost = unit_cost * firm.machine_capacity
    return Machine(firm.machine_capacity, cost, tech["maintenance_rate"] * cost, tech["transition_rate"] * cost,
                   ordered_at=t, active_from=t + tech["delay"])


@dataclass(frozen=True)
class MarketView:
    """What a firm takes as given about its market when it looks ahead."""

    budget: float
    base_share: float
    rest_weight: float  # weight of its rivals at their current prices
    eta: float


@dataclass(frozen=True)
class Evaluation:
    npv_with: float
    npv_without: float
    feasible: bool
    cost: float

    @property
    def gain(self):
        return self.npv_with - self.npv_without


def project(firm, t, machine, view, p):
    """Cash flows of the decision quarter t and of quarters t+1 … t+H, with or without a machine.

    The projection runs the same planner as the simulation, with the same dates,
    the same `requirements` and the same adjustment costs; vacancies are assumed
    filled and quits are their expected value. The firm's own price follows its
    target at speed λ and its own demand answers through the market-share formula,
    rivals' prices held fixed. Capex is in the flow of quarter t, once. The machine
    keeps a linear residual value at the horizon, returned separately because it
    is not cash. No interest and no loan flow enter here: they are financing.
    """
    tech, planning, dem, lab = p["technology"], p["planning"], p["demand"], p["labor"]
    horizon, h = tech["npv_horizon"], planning["horizon"]
    adjustment = (planning["hiring_cost_months"], planning["severance_months"])
    shadow = copy.copy(firm)
    shadow.staff, shadow.due = firm.staff.copy(), firm.due.copy()
    shadow.machine = machine if machine is not None else firm.machine
    history = list(firm.demand_history[-dem["history"]:])
    price = firm.price
    flows = np.zeros(horizon + 1)

    def decide(tq):
        """End-of-quarter decisions in the projection; returns their cost."""
        nonlocal price
        expected = float(np.mean(history[-dem["history"]:]))
        machines = [shadow.machine_at(tq + 1 + k) for k in range(h)]
        plan = plan_staff(shadow, expected, price - shadow.unit_ci, machines, adjustment)
        cost = 0.0
        for task in (A, B):
            s = int(shadow.staff[task])
            gross = shadow.table("gross", task, s + int(plan.hires[task]))
            cost += adjustment[1] / 3 * (gross[s] - gross[s - int(plan.execute[task])])
            cost += adjustment[0] / 3 * (gross[s + int(plan.hires[task])] - gross[s])
        shadow.staff = shadow.staff - plan.execute + plan.hires
        shadow.due = plan.notify.copy()
        target = price_target(shadow, plan, shadow.maintenance_at(tq + 2))
        price = markets.adjust_price(price, target, dem["lambda"])
        return cost

    flows[0] = -(machine.cost if machine is not None else 0.0) - decide(t)
    for k in range(1, horizon + 1):
        tq = t + k
        demand = markets.own_demand(price, firm.ref_price, view.base_share, view.rest_weight, view.budget, view.eta)
        out = min(demand, float(capacity(shadow.staff[A], shadow.staff[B], shadow.machine_at(tq),
                                         firm.a, firm.b, firm.theta)))
        surplus = (price - firm.unit_ci) * out - firm.fixed_cost - shadow.maintenance_at(tq) - shadow.amount("cost")
        if machine is not None and tq == machine.active_from:
            surplus -= machine.transition
        staying = shadow.staff - shadow.due
        shadow.staff = shadow.staff - np.floor(lab["quit_rate"] * staying + 0.5).astype(int)
        history.append(demand)
        flows[k] = surplus - decide(tq)
    residual = 0.0
    if machine is not None:
        used = t + horizon - machine.active_from + 1
        residual = machine.cost * max(0.0, 1 - used / tech["lifetime"])
    return flows, residual


def financing_path_ok(firm, machine, flows, p):
    """Can the firm pay K now and service the loan along the projected path?

    Cash above the buffer pays first; the loan covers the rest, from the order
    date. From the next quarter, interest on loan and overdraft, and equal
    principal installments, leave cash. Loan + overdraft must stay within the
    ceiling throughout.
    """
    fin = p["finance"]
    if firm.financing_available() < machine.cost - 1e-6:
        return False
    used = max(0.0, min(machine.cost, firm.cash - firm.cash_buffer))
    loan = firm.loan + machine.cost - used
    payment = firm.loan_payment + (machine.cost - used) / fin["loan_quarters"]
    cash = firm.cash + flows[0] + (machine.cost - used)
    for k in range(1, len(flows)):
        interest = fin["loan_rate"] * (loan + max(0.0, -cash))
        principal = min(loan, payment)
        cash += flows[k] - interest - principal
        loan -= principal
        if loan + max(0.0, -cash) > firm.credit_ceiling + 1e-6:
            return False
    return True


def evaluate_machine(firm, t, unit_cost, view, p):
    """Economic profitability and financial feasibility, kept apart."""
    machine = new_machine(firm, t, unit_cost, p)
    r = p["finance"]["discount_rate"]
    discount = (1 + r) ** -np.arange(p["technology"]["npv_horizon"] + 1)
    with_flows, residual = project(firm, t, machine, view, p)
    without_flows, _ = project(firm, t, None, view, p)
    npv_with = float(with_flows @ discount + residual * discount[-1])
    npv_without = float(without_flows @ discount)
    return Evaluation(npv_with, npv_without, financing_path_ok(firm, machine, with_flows, p), machine.cost), machine


def order_machine(firm, machine, p):
    """Buy at the order date: cash above the buffer first, the loan for the rest."""
    used = max(0.0, min(machine.cost, firm.cash - firm.cash_buffer))
    borrowed = machine.cost - used
    firm.cash += borrowed - machine.cost
    firm.loan += borrowed
    firm.loan_payment += borrowed / p["finance"]["loan_quarters"]
    firm.machine = machine
    return borrowed
