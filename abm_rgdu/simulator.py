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
    pop_seq, shock_seq, stock_seq = np.random.SeedSequence(seed).spawn(3)
    firms = population.build_firms(p, np.random.default_rng(pop_seq), np.random.default_rng(stock_seq))
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
        wages = payroll.negotiated_wages(branch.alpha, f.profile, state.params)
        f.set_costs(payroll.band_costs(branch.alpha, f.profile, state.params, wages))


def _reenter(state, t):
    """Closure variant: a closed firm is replaced by an entrant after a delay, with no staff."""
    v = state.params["variants"]
    for f in state.firms:
        if not f.active and t > f.closed_at + v["replacement_delay"]:
            f.active, f.closed_at = True, None
            f.staff, f.due = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
            f.price, f.cash, f.loan, f.loan_payment, f.distress = f.ref_price, f.entry_cash, 0.0, 0.0, 0
            f.machine = None
            f.demand_history = [f.initial_output] * state.params["demand"]["history"]


def budget_factor(state, branch):
    """Budgets are exogenous, unless the branch lets them follow last quarter's income.

    B_s(t) = B_s × [(1 − φ) + φ × R(t−1) / R_0], with R = gross wages + ARE and R_0
    taken in the last quarter before the reform. φ is an exploratory sensitivity,
    not an estimated propensity to consume.
    """
    if not branch.feedback:
        return 1.0
    phi = state.params["demand"]["feedback_phi"]
    return (1 - phi) + phi * state.income_last / state.income_ref


def step(state, branch, records, invest):
    p = state.params
    t = state.t + 1
    row = t + p["simulation"]["burn_in"] - 1
    firms, pools = state.firms, state.pools
    n = len(firms)
    dem, lab, planning, fin = p["demand"], p["labor"], p["planning"], p["finance"]
    if p["variants"]["closures"]:
        _reenter(state, t)
    active = np.array([f.active for f in firms])

    staff0 = np.array([f.staff for f in firms])
    cash0 = np.array([f.cash for f in firms])
    loan0 = np.array([f.loan for f in firms])
    machine_now = np.array([f.machine_at(t) for f in firms])
    maintenance = np.array([f.maintenance_at(t) for f in firms])
    transition = np.array([f.machine.transition if f.machine is not None and f.machine.active_from == t else 0.0
                           for f in firms])

    # 2–4. Budgets, demand, production.
    budgets = np.array([m.budget for m in state.markets]) * budget_factor(state, branch)
    demand, output, extra, cap, share, weight = (np.zeros(n) for _ in range(6))
    lost, sales = np.zeros(len(state.markets)), np.zeros(len(state.markets))
    for m, budget in zip(state.markets, budgets):
        idx = m.firms
        prices = np.array([firms[i].price for i in idx])
        refs = np.array([firms[i].ref_price for i in idx])
        cap[idx] = [fm.capacity(firms[i].staff[A], firms[i].staff[B], machine_now[i], firms[i].a, firms[i].b,
                                firms[i].theta) for i in idx]
        share[idx] = mk.shares(prices, refs, m.base_share, dem["eta"], active[idx])
        weight[idx] = np.where(active[idx], mk.weights(prices, refs, m.base_share, dem["eta"]), 0.0)
        demand[idx] = mk.allocate(budget, prices, share[idx])
        output[idx], extra[idx], lost[m.index] = mk.spillover(demand[idx], cap[idx], prices, share[idx],
                                                              dem["spillover"])
        sales[m.index] = float((output[idx] * prices).sum())
    orders = demand + extra

    # 5. Accounts of the quarter; interest on loan and overdraft carried into it, scheduled principal.
    acc = {k: np.zeros(n) for k in ("price", "revenue", "variable_ci", "fixed_cost", "maintenance", "value_added",
                                    "labor_cost", "surplus", "gross_wages", "contributions", "reduction_full",
                                    "interest", "principal", "transition")}
    for i, f in enumerate(firms):
        if not f.active:
            continue  # a closed firm's accounts are frozen
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
        acc["principal"][i] = min(f.loan, f.loan_payment)
        acc["transition"][i] = transition[i]
        f.cash += a["surplus"] - acc["interest"][i] - acc["principal"][i] - transition[i]
        f.loan -= acc["principal"][i]
    are = np.array([pool.are(p) for pool in pools])
    unemployed = np.array([pool.unemployed() for pool in pools])  # during t, like employment
    buckets = np.stack([pool.by_bucket() for pool in pools])

    # 6. Quits, among those not already leaving on notice.
    quits = np.zeros((n, 2), dtype=int)
    for i, f in enumerate(firms):
        staying = f.staff - f.due
        quits[i] = np.minimum(staying, labor_pool.stochastic_round(lab["quit_rate"] * staying, state.shocks.quits[row, i]))
        f.staff = f.staff - quits[i]

    # 7. Machine decision: economic profitability and financial feasibility, tested apart.
    for i, f in enumerate(firms):
        f.demand_history.append(float(orders[i]))
    tech = {k: np.zeros(n) for k in ("capex", "new_loan", "npv_gain")}
    feasible, ordered = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    evaluated = np.zeros(n, dtype=bool)
    if invest:
        unit_cost = fm.unit_cost_at(t, p)
        for m, budget in zip(state.markets, budgets):
            for j, i in enumerate(m.firms):
                f = firms[i]
                if f.machine is not None or f.machine_capacity <= 0 or not f.active:
                    continue
                view = fm.MarketView(budget, m.base_share[j], weight[m.firms].sum() - weight[i], dem["eta"])
                ev, machine = fm.evaluate_machine(f, t, unit_cost, view, p)
                evaluated[i], feasible[i], tech["npv_gain"][i] = True, ev.feasible, ev.gain
                if ev.gain > p["technology"]["npv_threshold"] and ev.feasible:
                    tech["new_loan"][i] = fm.order_machine(f, machine, p)
                    tech["capex"][i] = machine.cost
                    ordered[i] = True
    blocked = evaluated & (tech["npv_gain"] > p["technology"]["npv_threshold"]) & ~feasible

    # 8. Labor plan, executed at once; 9. price for t+1.
    adjustment = (planning["hiring_cost_months"], planning["severance_months"])
    fields = ("vacancies", "layoffs", "notices", "cancelled", "non_replacements")
    flows = {k: np.zeros((n, 2), dtype=int) for k in fields}
    severance, planned, expected_demand = np.zeros(n), np.zeros(n), np.zeros(n)
    allowed = np.zeros(n, dtype=bool)
    closed = np.zeros(n, dtype=bool)
    for i, f in enumerate(firms):
        if not f.active:
            continue
        if p["variants"]["closures"] and f.distress >= p["variants"]["closure_quarters"]:
            severance[i] = _close(f, t, planning["severance_months"])
            flows["layoffs"][i], closed[i] = staff0[i] - quits[i], True
            continue
        expected = float(np.mean(f.demand_history[-dem["history"]:]))
        machines = [f.machine_at(t + 1 + k) for k in range(planning["horizon"])]
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
        f.price = mk.adjust_price(f.price, fm.price_target(f, plan, f.maintenance_at(t + 2)), dem["lambda"])

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
        if not f.active:
            continue
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
        "machine_capacity": machine_now, "loan": np.array([f.loan for f in firms]), "loan_start": loan0,
        **tech, "evaluated": evaluated, "feasible": feasible, "blocked": blocked, "ordered": ordered,
        "active": active, "closed": closed,
        "machine_state": np.array([_machine_state(f, t) for f in firms]),
    }
    pool_rec = {"active": [pool.active for pool in pools], "unemployed": unemployed, "buckets": buckets,
                "are": are, "hires": hires.sum(axis=0), "separations": separations.sum(axis=0)}
    market_rec = {"budget": budgets, "sales": sales, "lost": lost}
    public = {"contributions": acc["contributions"].sum(), "reduction_applied": reduction_applied.sum(),
              "are": are.sum(), "balance": balance, "income": state.income_last}
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
    flow = (rec["surplus"] - rec["interest"] - rec["principal"] - rec["transition"] - rec["severance"]
            - rec["hiring_cost"] - rec["capex"] + rec["new_loan"])
    if not np.allclose(rec["cash"] - rec["cash_start"], flow, rtol=1e-9, atol=1e-4):
        raise InvariantError("cash identity broken")
    if not np.allclose(rec["loan"], rec["loan_start"] - rec["principal"] + rec["new_loan"], atol=1e-6):
        raise InvariantError("loan identity broken")
    inactive = rec["machine_state"] < 2
    if np.any(rec["machine_capacity"][inactive] != 0):
        raise InvariantError("a machine that is not installed adds capacity")
    if np.any(market_rec["sales"] > market_rec["budget"] * (1 + 1e-9)):
        raise InvariantError("sales above the budget")
    for m in state.markets:
        if rec["active"][m.firms].any() and not np.isclose(rec["share"][m.firms].sum(), 1.0):
            raise InvariantError("market shares do not sum to one")
        if np.any(rec["share"][m.firms][~rec["active"][m.firms]] != 0):
            raise InvariantError("a closed firm still has customers")


def _close(f, t, severance_months):
    """Closure variant: every post goes at once, with severance; accounts freeze from now on."""
    cost = 0.0
    for task in (A, B):
        s = int(f.staff[task])
        gross = f.table("gross", task, s)
        cost += severance_months / 3 * gross[s]
    f.cash -= cost
    f.staff, f.due = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    f.active, f.closed_at = False, t
    return cost


def _machine_state(firm, t):
    """0: none; 1: ordered or installing during the next quarter; 2: active in the quarter just run."""
    if firm.machine is None:
        return 0
    return 2 if firm.machine.active(t) else 1
