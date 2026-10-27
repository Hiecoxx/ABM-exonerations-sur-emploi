import copy

import numpy as np
import pytest

from abm_rgdu import params
from abm_rgdu import simulator as sim
from worlds import small_world

P = params.load()


@pytest.fixture(scope="module")
def warmed():
    state = sim.initial_state(P, 3)
    sim.burn_in(state)
    return state


def _all_records(rec):
    out = {}
    for kind in ("firm", "pool", "market", "public"):
        for name in getattr(rec, kind):
            out[(kind, name)] = rec.get(kind, name)
    return out


def test_identities_hold_every_quarter_even_under_stress(warmed):
    # check_invariants is on: any broken identity raises inside the loop.
    for alpha in (1.0, 0.8, 0.0):
        sim.run(copy.deepcopy(warmed), sim.Branch(alpha))


def test_reference_against_itself_gives_exactly_zero_gaps(warmed):
    one = _all_records(sim.run(copy.deepcopy(warmed), sim.Branch(1.0)))
    two = _all_records(sim.run(copy.deepcopy(warmed), sim.Branch(1.0)))
    for key in one:
        assert np.array_equal(one[key], two[key]), key


def test_same_scenario_twice_gives_identical_results():
    runs = []
    for _ in range(2):
        state = sim.initial_state(P, 9)
        burn = sim.burn_in(state)
        runs.append((_all_records(burn), _all_records(sim.run(state, sim.Branch(0.9)))))
    for (b1, r1), (b2, r2) in zip(runs[:1], runs[1:]):
        for key in r1:
            assert np.array_equal(r1[key], r2[key]), key
            assert np.array_equal(b1[key], b2[key]), key


def test_drift_of_the_reference_is_accounted_for_by_flows(warmed):
    rec = sim.run(copy.deepcopy(warmed), sim.Branch(1.0))
    employment = rec.get("firm", "employment").sum(axis=(1, 2))
    hires = rec.get("firm", "hires").sum(axis=(1, 2))
    exits = rec.get("firm", "exits").sum(axis=(1, 2))
    assert np.array_equal(employment[1:], (employment + hires - exits)[:-1])
    unfilled = rec.get("firm", "vacancies") - rec.get("firm", "hires")
    assert np.all(unfilled >= 0)


def test_recorded_unemployment_is_that_of_the_same_quarter(warmed):
    rec = sim.run(copy.deepcopy(warmed), sim.Branch(0.8))
    employment = rec.get("firm", "employment").sum(axis=1)
    assert np.array_equal(employment + rec.get("pool", "unemployed"), rec.get("pool", "active"))


def test_partial_public_balance_is_built_from_its_parts(warmed):
    rec = sim.run(copy.deepcopy(warmed), sim.Branch(0.8))
    full = rec.get("firm", "reduction_full").sum(axis=1)
    assert rec.get("public", "reduction_applied") == pytest.approx(0.8 * full)
    balance = rec.get("public", "contributions") - rec.get("public", "reduction_applied") - rec.get("public", "are")
    assert rec.get("public", "balance") == pytest.approx(balance)
    assert rec.get("public", "are") == pytest.approx(rec.get("pool", "are").sum(axis=1))


def test_reform_without_technology_constructed_case():
    """Where the mechanism must work, it does: costs up, prices up, fewer sales, fewer posts.

    With a fixed nominal budget and prices following costs, quantities fall,
    and so does employment. No such sign is asserted for general runs.
    """
    p = small_world()
    state = sim.initial_state(p, 5)
    sim.burn_in(state)
    ref = sim.run(copy.deepcopy(state), sim.Branch(1.0))
    pol = sim.run(copy.deepcopy(state), sim.Branch(0.8))
    ref_price = np.array([f.ref_price for f in state.firms])
    price_ref = (ref.get("firm", "price")[-1] / ref_price).mean()
    price_pol = (pol.get("firm", "price")[-1] / ref_price).mean()
    assert price_pol > price_ref
    assert pol.get("firm", "employment")[-1].sum() < ref.get("firm", "employment")[-1].sum()
    assert pol.get("firm", "output")[-1].sum() < ref.get("firm", "output")[-1].sum()


def test_budgets_are_exogenous_unless_the_branch_feeds_them_back(warmed):
    base = np.array([m.budget for m in warmed.markets])
    off = sim.run(copy.deepcopy(warmed), sim.Branch(0.8, False, False))
    assert np.allclose(off.get("market", "budget"), base)
    on = sim.run(copy.deepcopy(warmed), sim.Branch(0.8, False, True))
    phi = P["demand"]["feedback_phi"]
    income = np.concatenate([[warmed.income_ref], on.get("public", "income")[:-1]])
    expected = base[None, :] * ((1 - phi) + phi * income / warmed.income_ref)[:, None]
    assert on.get("market", "budget") == pytest.approx(expected)
    # The first quarter of a feedback branch starts from the reference income exactly.
    assert on.get("market", "budget")[0] == pytest.approx(base)


def test_feedback_against_itself_is_exactly_zero(warmed):
    one = sim.run(copy.deepcopy(warmed), sim.Branch(1.0, False, True))
    two = sim.run(copy.deepcopy(warmed), sim.Branch(1.0, False, True))
    for kind, name in (("firm", "employment"), ("market", "budget"), ("public", "balance")):
        assert np.array_equal(one.get(kind, name), two.get(kind, name))
