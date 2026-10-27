"""The quarterly loop, with its dates stated once.

During quarter t a firm produces, sells, and pays wages, intermediate
consumption and interest. At the end of t it learns its quits, decides on
technology and staff, sets its price for t+1 and posts vacancies, which are
matched at once: hires work from t+1. A layoff notified at the end of t is
paid and works through t+1, and leaves at the end of t+1.

Order within a quarter:
     1. costs under α           2. budgets               3. demand and spillover
     4. production              5. accounts, financing   6. quits, then layoffs due
     7. machine decision        8. labor plan            9. price update
    10. matching               11. ageing of the pools  12. partial public balance
    13. records and invariant checks
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from . import firms as fm
from . import labor_pool, payroll, population
from . import markets as mk
from .firms import A, B


@dataclass(frozen=True)
class Branch:
    alpha: float = 1.0        # share of the RGDU kept; 1 is the reference
    technology: bool = False  # may firms buy their machine?
    feedback: bool = False    # do budgets follow labor income and ARE?


@dataclass
class Shocks:
    """Primitive draws of a seed, shared by every branch."""

    quits: np.ndarray     # (quarters, firms, tasks)
    match: np.ndarray
    tiebreak: np.ndarray


@dataclass
class State:
    params: dict
    seed: int
    firms: list
    markets: list
    pools: list
    shocks: Shocks
    t: int                    # last quarter completed
    income_ref: float = 0.0   # labor income + ARE in the last quarter before the reform
    income_last: float = 0.0  # the same, in the last quarter completed


@dataclass
class Records:
    quarters: list = field(default_factory=list)
    firm: dict = field(default_factory=dict)
    pool: dict = field(default_factory=dict)
    market: dict = field(default_factory=dict)
    public: dict = field(default_factory=dict)

    def add(self, t, firm, pool, market, public):
        self.quarters.append(t)
        for store, values in ((self.firm, firm), (self.pool, pool), (self.market, market), (self.public, public)):
            for name, value in values.items():
                store.setdefault(name, []).append(np.asarray(value))

    def get(self, kind, name):
        """Stacked series, quarter first."""
        return np.stack(getattr(self, kind)[name])


class InvariantError(AssertionError):
    pass


def initial_state(p, seed):
    sim = p["simulation"]
    pop_seq, shock_seq = np.random.SeedSequence(seed).spawn(2)
    firms = population.build_firms(p, np.random.default_rng(pop_seq))
    rng = np.random.default_rng(shock_seq)
    shape = (sim["burn_in"] + sim["quarters"], len(firms), 2)
    shocks = Shocks(rng.random(shape), rng.random(shape), rng.random(shape))
    state = State(p, seed, firms, mk.build_markets(firms, p), labor_pool.build_pools(firms, p), shocks,
                  t=-sim["burn_in"])
    state.income_ref = state.income_last = (
        sum(f.amount("gross") for f in firms) + sum(pool.are(p) for pool in state.pools)
    )
    return state


def burn_in(state):
    """Quarters before the reform, at α = 1 and with no investment, common to all branches."""
    records = Records()
    branch = Branch()
    _prepare(state, branch)
    while state.t < 0:
        step(state, branch, records, invest=False)
    state.income_ref = state.income_last
    return records


def run(state, branch):
    """Quarters 1 … T of one branch. The state is consumed: pass a copy."""
    records = Records()
    _prepare(state, branch)
    while state.t < state.params["simulation"]["quarters"]:
        step(state, branch, records, invest=branch.technology)
    return records


def run_branches(state, branches):
    """Every branch starts from the same state, with the same shocks."""
    return {branch: run(copy.deepcopy(state), branch) for branch in branches}


def _prepare(state, branch):
    for f in state.firms:
        f.set_costs(payroll.band_costs(branch.alpha, f.profile, state.params))


def step(state, branch, records, invest):
    p = state.params
    t = state.t + 1
    row = t + p["simulation"]["burn_in"] - 1
    firms, pools = state.firms, state.pools
    n = len(firms)
    dem, lab, planning, fin = p["demand"], p["labor"], p["planning"], p["finance"]

    staff0 = np.array([f.staff for f in firms])
    cash0 = np.array([f.cash for f in firms])
    machine_now = np.zeros(n)
    maintenance = np.zeros(n)

    # 2–4. Budgets, demand, production.
    budgets = np.array([m.budget for m in state.markets])
    demand, output, extra, cap, share = (np.zeros(n) for _ in range(5))
    lost, sales = np.zeros(len(state.markets)), np.zeros(len(state.markets))
    for m, budget in zip(state.markets, budgets):
        idx = m.firms
        prices = np.array([firms[i].price for i in idx])
        refs = np.array([firms[i].ref_price for i in idx])
        cap[idx] = [fm.capacity(firms[i].staff[A], firms[i].staff[B], machine_now[i], firms[i].a, firms[i].b,
                                firms[i].theta) for i in idx]
        share[idx] = mk.shares(prices, refs, m.base_share, dem["eta"])
        demand[idx] = mk.allocate(budget, prices, share[idx])
        output[idx], extra[idx], lost[m.index] = mk.spillover(demand[idx], cap[idx], prices, share[idx],
                                                              dem["spillover"])
        sales[m.index] = float((output[idx] * prices).sum())
    orders = demand + extra

    # 5. Accounts of the quarter, and interest on the overdraft carried into it.
    acc = {k: np.zeros(n) for k in ("price", "revenue", "variable_ci", "fixed_cost", "maintenance", "value_added",
                                    "labor_cost", "surplus", "gross_wages", "contributions", "reduction_full",
                                    "interest")}
    for i, f in enumerate(firms):
        a = fm.accounts(f.price, output[i], f.unit_ci, f.fixed_cost, maintenance[i], f.amount("cost"))
        acc["price"][i] = f.price
        acc["revenue"][i], acc["variable_ci"][i] = a["revenue"], a["variable_ci"]
        acc["fixed_cost"][i], acc["maintenance"][i] = f.fixed_cost, maintenance[i]
        acc["value_added"][i], acc["surplus"][i] = a["value_added"], a["surplus"]
        acc["labor_cost"][i] = f.amount("cost")
        acc["gross_wages"][i] = f.amount("gross")
        acc["contributions"][i] = f.amount("contributions")
        acc["reduction_full"][i] = f.amount("reduction")
        acc["interest"][i] = fin["loan_rate"] * f.debt()
        f.cash += a["surplus"] - acc["interest"][i]
    are = np.array([pool.are(p) for pool in pools])
    unemployed = np.array([pool.unemployed() for pool in pools])  # during t, like employment
    buckets = np.stack([pool.by_bucket() for pool in pools])

    # 6. Quits, among those not already leaving on notice.
    quits = np.zeros((n, 2), dtype=int)
    for i, f in enumerate(firms):
        staying = f.staff - f.due
        quits[i] = np.minimum(staying, labor_pool.stochastic_round(lab["quit_rate"] * staying, state.shocks.quits[row, i]))
        f.staff = f.staff - quits[i]

    # 8. Labor plan, executed at once; 9. price for t+1.
    adjustment = (planning["hiring_cost_months"], planning["severance_months"])
    fields = ("vacancies", "layoffs", "notices", "cancelled", "non_replacements")
    flows = {k: np.zeros((n, 2), dtype=int) for k in fields}
    severance, planned, expected_demand = np.zeros(n), np.zeros(n), np.zeros(n)
    allowed = np.zeros(n, dtype=bool)
    for i, f in enumerate(firms):
        f.demand_history.append(float(orders[i]))
        expected = float(np.mean(f.demand_history[-dem["history"]:]))
        machines = [0.0] * planning["horizon"]
        allowed[i] = f.financing_available() > 0
        plan = fm.plan_staff(f, expected, f.price - f.unit_ci, machines, adjustment, allowed[i])
        for task in (A, B):
            s = int(f.staff[task])
            gross = f.table("gross", task, s)
            severance[i] += planning["severance_months"] / 3 * (gross[s] - gross[s - int(plan.execute[task])])
        f.staff = f.staff - plan.execute
        f.due = plan.notify.copy()
        f.cash -= severance[i]
        f.planned_output = planned[i] = plan.output
        expected_demand[i] = expected
        flows["vacancies"][i], flows["layoffs"][i] = plan.hires, plan.execute
        flows["notices"][i], flows["cancelled"][i] = plan.notify, plan.cancel
        flows["non_replacements"][i] = np.maximum(0, quits[i] - plan.hires)
        f.price = mk.adjust_price(f.price, fm.price_target(f, plan, 0.0), dem["lambda"])

    # 10. Matching: separations join their pool, vacancies are filled from it.
    hires = np.zeros((n, 2), dtype=int)
    hiring_cost = np.zeros(n)
    separations = quits + flows["layoffs"]
    for task, pool in zip((A, B), pools):
        pool.receive(separations[:, task].sum())
        hires[:, task] = labor_pool.match(flows["vacancies"][:, task], lab["match_prob"],
                                          state.shocks.match[row, :, task], pool.available(),
                                          state.shocks.tiebreak[row, :, task])
        pool.hire(int(hires[:, task].sum()), lab["hire_weights"])
    for i, f in enumerate(firms):
        for task in (A, B):
            s = int(f.staff[task])
            gross = f.table("gross", task, s + int(hires[i, task]))
            hiring_cost[i] += planning["hiring_cost_months"] / 3 * (gross[s + hires[i, task]] - gross[s])
        f.staff = f.staff + hires[i]
        f.cash -= hiring_cost[i]
        f.distress = f.distress + 1 if f.debt() > f.credit_ceiling else 0

    # 11. Ageing; 12. partial public balance.
    for pool in pools:
        pool.age()
    reduction_applied = branch.alpha * acc["reduction_full"]
    balance = acc["contributions"].sum() - reduction_applied.sum() - are.sum()
    state.income_last = acc["gross_wages"].sum() + are.sum()

    # 13. Records and checks.
    firm_rec = {
        "employment": staff0, "quits": quits, "hires": hires, "exits": quits + flows["layoffs"], **flows,
        "demand": demand, "orders": orders, "spillover_in": extra, "output": output, "capacity": cap,
        "share": share, **acc, "reduction_applied": reduction_applied, "severance": severance,
        "hiring_cost": hiring_cost, "cash": np.array([f.cash for f in firms]), "cash_start": cash0,
        "debt": np.array([f.debt() for f in firms]), "distress": np.array([f.distress for f in firms]),
        "hiring_allowed": allowed, "planned_output": planned, "expected_demand": expected_demand,
        "machine_capacity": machine_now,
    }
    pool_rec = {"active": [pool.active for pool in pools], "unemployed": unemployed, "buckets": buckets,
                "are": are, "hires": hires.sum(axis=0), "separations": separations.sum(axis=0)}
    market_rec = {"budget": budgets, "sales": sales, "lost": lost}
    public = {"contributions": acc["contributions"].sum(), "reduction_applied": reduction_applied.sum(),
              "are": are.sum(), "balance": balance}
    records.add(t, firm_rec, pool_rec, market_rec, public)
    state.t = t
    if p["simulation"]["check_invariants"]:
        check(state, firm_rec, pool_rec, market_rec)


def check(state, rec, pool_rec, market_rec):
    """Identities that must hold whatever the parameters."""
    firms, pools = state.firms, state.pools
    staff = np.array([f.staff for f in firms])
    if not np.array_equal(staff, rec["employment"] + rec["hires"] - rec["exits"]):
        raise InvariantError("hires − exits ≠ change in staff")
    for task, pool in zip((A, B), pools):
        if staff[:, task].sum() + pool.unemployed() != pool.active:
            raise InvariantError(f"pool {task}: employment + unemployment ≠ active population")
    if not np.array_equal(rec["employment"].sum(axis=0) + pool_rec["unemployed"], np.array(pool_rec["active"])):
        raise InvariantError("recorded employment + unemployment ≠ active population")
    va = rec["revenue"] - rec["variable_ci"] - rec["fixed_cost"] - rec["maintenance"]
    if not np.allclose(va, rec["value_added"], rtol=1e-12, atol=1e-6):
        raise InvariantError("CA − CI − maintenance ≠ VA")
    flow = rec["surplus"] - rec["interest"] - rec["severance"] - rec["hiring_cost"]
    if not np.allclose(rec["cash"] - rec["cash_start"], flow, rtol=1e-9, atol=1e-4):
        raise InvariantError("cash identity broken")
    if np.any(market_rec["sales"] > market_rec["budget"] * (1 + 1e-9)):
        raise InvariantError("sales above the budget")
    for m in state.markets:
        if not np.isclose(rec["share"][m.firms].sum(), 1.0):
            raise InvariantError("market shares do not sum to one")
