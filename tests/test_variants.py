"""Robustness switches: all off by default, each one does what it says and nothing else."""

import copy
import math

import numpy as np
import pytest

from abm_rgdu import firms, params, payroll
from abm_rgdu import simulator as sim
from abm_rgdu.firms import A
from worlds import small_world

P = params.load()
ADOPT = {"technology.unit_cost": 20000.0, "demand.lambda": 0.1}


def warmed(p, seed=5):
    state = sim.initial_state(p, seed)
    sim.burn_in(state)
    return state


def test_all_switches_are_off_by_default():
    v = P["variants"]
    assert not v["closures"] and v["wage_passthrough"] == 0.0
    assert P["finance"]["credit_mode"] == "central" and not P["finance"]["investment_cash_only"]
    assert P["technology"]["cost_path"] == "stable" and P["technology"]["initial_stock_share"] == 0.0


def test_credit_regimes():
    central = sim.initial_state(small_world(), 5).firms
    free = sim.initial_state(small_world(**{"finance.credit_mode": "free"}), 5).firms
    rationed = sim.initial_state(small_world(**{"finance.credit_mode": "rationed",
                                                "finance.credit_months_rationed": [0.5]}), 5).firms
    for c, f, r in zip(central, free, rationed):
        assert math.isinf(f.credit_ceiling)
        assert r.credit_ceiling == pytest.approx(c.credit_ceiling * 0.5 / 3.0)


def test_investment_from_cash_only_never_borrows():
    p = small_world(**{"technology.unit_cost": 40000.0, "demand.lambda": 0.1,
                       "finance.investment_cash_only": True, "finance.credit_months": [20.0]})
    state = warmed(p)
    for f in state.firms:
        f.cash = f.cash_buffer
    rec = sim.run(copy.deepcopy(state), sim.Branch(1.0, True))
    assert rec.get("firm", "new_loan").sum() == 0
    assert rec.get("firm", "blocked")[0].any()


def test_tech_cost_path_is_exogenous_and_shared_by_branches():
    p = small_world(**ADOPT, **{"technology.cost_path": "fast"})
    assert firms.unit_cost_at(1, p) == 20000.0
    assert firms.unit_cost_at(5, p) == pytest.approx(20000.0 * 0.96**4)
    state = warmed(p)
    ref, pol = copy.deepcopy(state), copy.deepcopy(state)
    sim.run(ref, sim.Branch(1.0, True))
    sim.run(pol, sim.Branch(0.8, True))
    for f in ref.firms + pol.firms:
        if f.machine is not None:
            assert f.machine.cost == pytest.approx(firms.unit_cost_at(f.machine.ordered_at, p) * f.machine_capacity)


def test_negotiated_wages_absorb_part_of_the_cut_within_bounds():
    profile = payroll.Profile(large=False, size=0, at_mp=0.02)
    p = params.override(P, {"variants.wage_passthrough": 1.0, "variants.wage_cut_max": 0.02})
    base = payroll.band_wages(P)
    assert np.array_equal(payroll.negotiated_wages(0.8, profile, P), base)  # off by default
    assert np.array_equal(payroll.negotiated_wages(1.0, profile, p), base)  # nothing to absorb at α = 1
    cut = payroll.negotiated_wages(0.8, profile, p)
    assert cut[0] == pytest.approx(base[0] * 0.98)       # bound reached
    assert cut[3] == base[3]                             # no RGDU above 3 SMIC, nothing to pass on
    assert np.all(cut >= payroll.smic_current(P))
    assert np.all(cut >= base * 0.98 - 1e-9)


def test_closures_freeze_the_firm_and_an_entrant_comes_back_empty():
    p = small_world(**{"variants.closures": True, "sizes.margin_quartiles": [[0.02, 0.02, 0.02]],
                       "sizes.cash_months_quartiles": [[0.0, 0.0, 0.0]], "finance.credit_months": [0.1],
                       "demand.lambda": 0.0, "population.replications": 8})
    rec = sim.run(copy.deepcopy(warmed(p)), sim.Branch(0.0))  # invariants are checked inside
    active, closed = rec.get("firm", "active"), rec.get("firm", "closed")
    employment = rec.get("firm", "employment").sum(axis=2)
    assert closed.any()
    delay = p["variants"]["replacement_delay"]
    for i in np.flatnonzero(closed.any(axis=0)):
        t = int(np.flatnonzero(closed[:, i])[0])
        assert rec.get("firm", "distress")[t - 1, i] >= p["variants"]["closure_quarters"]
        assert employment[t + 1:t + 1 + delay, i].sum() == 0
        assert not active[t + 1:t + 1 + delay, i].any()
        assert np.all(rec.get("firm", "share")[t + 1:t + 1 + delay, i] == 0)
        if t + 1 + delay < len(rec.quarters):
            assert active[t + 1 + delay, i] and employment[t + 1 + delay, i] == 0


def test_initial_stock_of_machines_is_tagged_and_already_reflected_in_staff():
    none = sim.initial_state(small_world(), 5)
    stock = sim.initial_state(small_world(**{"technology.initial_stock_share": 1.0}), 5)
    for f0, f1 in zip(none.firms, stock.firms):
        assert f0.machine is None
        assert f1.machine.origin == "pre-existing" and f1.machine_at(1) == f1.machine_capacity
        assert f1.loan == 0.0
        assert f1.staff[A] < f0.staff[A]
        revenue = f1.price * f1.initial_output
        ci = f1.unit_ci * f1.initial_output + f1.fixed_cost + f1.machine.maintenance
        assert (revenue - ci) / revenue == pytest.approx(small_world()["markets"]["va_rate"][0])
    sim.burn_in(stock)
    rec = sim.run(copy.deepcopy(stock), sim.Branch(1.0, True))
    assert rec.get("firm", "ordered").sum() == 0  # nothing new to buy: the option is used
    assert np.all(rec.get("firm", "machine_state") == 2)
