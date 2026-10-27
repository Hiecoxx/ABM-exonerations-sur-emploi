import copy

import numpy as np
import pytest

from abm_rgdu import firms, markets, params
from abm_rgdu import simulator as sim
from abm_rgdu.firms import Machine
from worlds import small_world


def warmed(p, seed=5):
    state = sim.initial_state(p, seed)
    sim.burn_in(state)
    return state


def view(state, i):
    p = state.params
    m = state.markets[state.firms[i].market]
    prices = np.array([state.firms[j].price for j in m.firms])
    refs = np.array([state.firms[j].ref_price for j in m.firms])
    w = markets.weights(prices, refs, m.base_share, p["demand"]["eta"])
    j = list(m.firms).index(i)
    return firms.MarketView(m.budget, m.base_share[j], w.sum() - w[j], p["demand"]["eta"])


def test_credit_financed_investment_is_accounted_for():
    p = small_world(**{"technology.unit_cost": 40000.0, "demand.lambda": 0.1, "finance.credit_months": [20.0]})
    state = warmed(p)
    for f in state.firms:
        f.cash = f.cash_buffer  # nothing to spare: the machine has to be borrowed for
    rec = sim.run(copy.deepcopy(state), sim.Branch(1.0, True))
    new_loan, principal = rec.get("firm", "new_loan"), rec.get("firm", "principal")
    assert new_loan.sum() > 0
    # What is still owed plus what was repaid is what was borrowed.
    assert rec.get("firm", "loan")[-1] + principal.sum(axis=0) == pytest.approx(new_loan.sum(axis=0))
    # Interest: one rate on the loan and the overdraft carried into the quarter.
    carried = rec.get("firm", "loan_start") + np.maximum(0, -rec.get("firm", "cash_start"))
    assert rec.get("firm", "interest") == pytest.approx(p["finance"]["loan_rate"] * carried)
    # K is paid once, in the order quarter, and nowhere else.
    capex, ordered = rec.get("firm", "capex"), rec.get("firm", "ordered")
    assert np.array_equal(capex > 0, ordered)
    assert np.all(ordered.sum(axis=0) <= 1)


def test_npv_contains_no_financing_flows():
    p = small_world(**{"technology.unit_cost": 20000.0})
    state = warmed(p)
    f = state.firms[0]
    one, _ = firms.evaluate_machine(f, 1, 20000.0, view(state, 0), p)
    two, _ = firms.evaluate_machine(f, 1, 20000.0, view(state, 0), params.override(p, {"finance.loan_rate": 0.08}))
    assert one.gain == two.gain


def test_a_profitable_project_can_be_blocked_by_financing():
    # K above one quarter's surplus, NPV still positive.
    p = small_world(**{"technology.unit_cost": 40000.0, "demand.lambda": 0.1})
    state = warmed(p)
    for f in state.firms:
        f.cash, f.cash_buffer, f.credit_ceiling = 0.0, 0.0, 0.0
    rec = sim.run(copy.deepcopy(state), sim.Branch(1.0, True))
    blocked = rec.get("firm", "blocked")[0]
    assert blocked.any()
    assert np.all(rec.get("firm", "npv_gain")[0][blocked] > 0)
    assert rec.get("firm", "ordered")[0].sum() == 0


def test_financing_path_is_checked_beyond_the_order_date():
    f = copy.deepcopy(warmed(small_world()).firms[0])
    f.cash, f.cash_buffer, f.credit_ceiling, f.loan, f.loan_payment = 150.0, 0.0, 60.0, 0.0, 0.0
    machine = Machine(capacity=1.0, cost=100.0, maintenance=0.0, transition=0.0, ordered_at=1, active_from=3)
    p = small_world()
    losing = np.array([-100.0, -50.0, -50.0, -50.0])
    assert not firms.financing_path_ok(f, machine, losing, p)
    steady = np.array([-100.0, 10.0, 10.0, 10.0])
    assert firms.financing_path_ok(f, machine, steady, p)
    f.credit_ceiling = 0.0
    f.cash = 90.0
    assert not firms.financing_path_ok(f, machine, steady, p)  # cannot even pay K
