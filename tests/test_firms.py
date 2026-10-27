import numpy as np
import pytest

from abm_rgdu import firms, payroll, params
from abm_rgdu.firms import A, B, BandLadder, capacity, requirements

P = params.load()


def test_requirements_make_output_feasible_and_are_minimal():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a = rng.uniform(0.2, 0.8)
        theta = rng.uniform(0.0, 0.95)
        q = rng.uniform(0.0, 300.0)
        machine = 0.0 if rng.random() < 0.3 else rng.uniform(0.0, 60.0)
        need_a, need_b = requirements(q, machine, a, 1 - a, theta)
        assert capacity(need_a, need_b, machine, a, 1 - a, theta) >= q - 1e-6
        if need_a > 0:
            assert capacity(need_a - 1, need_b, machine, a, 1 - a, theta) < q
        if need_b > 0:
            assert capacity(need_a, need_b - 1, machine, a, 1 - a, theta) < q


def test_floating_noise_does_not_create_a_post():
    assert requirements(2 / 0.45, 0.0, 0.45, 0.55, 0.3) == (2, 3)


def test_inactive_machine_adds_nothing():
    assert capacity(4, 6, 0.0, 0.4, 0.6, 0.5) == pytest.approx(min(4 / 0.4, 6 / 0.6))


def test_machine_never_does_more_than_the_automatable_share():
    a, theta = 0.5, 0.4
    # Plenty of B and a huge machine: A capacity is capped by the human (1 − θ) share.
    assert capacity(6, 100, 1e6, a, 0.5, theta) == pytest.approx(6 / ((1 - theta) * a))
    need_a, _ = requirements(20.0, 1e6, a, 0.5, theta)
    assert need_a == 6  # ceil((1 − 0.4) × 0.5 × 20)


def test_ladder_keeps_the_band_mix_whatever_the_path():
    ladder = BandLadder([0.5, 0.3, 0.15, 0.05])
    for n in range(0, 200):
        counts = ladder.counts(n)
        assert counts.sum() == n
        assert np.all(np.abs(counts - ladder.shares * n) < 1)
        assert np.all(ladder.counts(n + 1) - counts >= 0)


def test_firm_cost_is_the_sum_over_its_cells():
    profile = payroll.Profile(large=False, size=1, at_mp=0.02)
    bc = payroll.band_costs(0.8, profile, P)
    f = firms.Firm(
        id=0, market=0, size=1, replicate=0, profile=profile, a=0.5, b=0.5, theta=0.3,
        ladders=[BandLadder([0.4, 0.4, 0.15, 0.05]), BandLadder([0.1, 0.3, 0.4, 0.2])],
        staff=np.array([13, 17]), price=1.0, ref_price=1.0, markup=0.1, unit_ci=0.0,
        fixed_cost=0.0, cash=0.0, machine_capacity=3.9, initial_output=26.0,
    )
    f.set_costs(bc)
    assert f.amount("cost") == pytest.approx((f.cells * bc.cost).sum())
    assert f.amount("reduction") == pytest.approx((f.cells * bc.reduction).sum())
    # Tables grow when the staff outgrows them.
    big = np.array([400, 3])
    assert f.amount("gross", big) == pytest.approx(
        (np.stack([f.ladders[A].counts(400), f.ladders[B].counts(3)]) * bc.gross).sum()
    )


def test_accounts_identity():
    acc = firms.accounts(price=2.0, output=10.0, unit_ci=0.5, fixed_cost=3.0, maintenance=1.0, labor_cost=7.0)
    assert acc["revenue"] - acc["variable_ci"] - 3.0 - 1.0 == pytest.approx(acc["value_added"])
    assert acc["surplus"] == pytest.approx(acc["value_added"] - 7.0)


def _firm(staff, a=0.5, theta=0.4, due=(0, 0), alpha=1.0):
    profile = payroll.Profile(large=False, size=1, at_mp=0.02)
    f = firms.Firm(
        id=0, market=0, size=1, replicate=0, profile=profile, a=a, b=1 - a, theta=theta,
        ladders=[BandLadder([0.6, 0.3, 0.08, 0.02]), BandLadder([0.3, 0.4, 0.2, 0.1])],
        staff=np.array(staff), price=1.0, ref_price=1.0, markup=0.1, unit_ci=0.0,
        fixed_cost=0.0, cash=1e6, machine_capacity=theta * staff[0], initial_output=20.0,
    )
    f.due = np.array(due)
    f.set_costs(payroll.band_costs(alpha, profile, P))
    return f


ADJ = (0.5, 1.25)


def test_extra_demand_needing_both_tasks_hires_the_bundle():
    f = _firm([10, 10])
    unit_labor = f.amount("cost") / 20.0
    plan = firms.plan_staff(f, 24.0, 2.0 * unit_labor, [0.0] * 4, ADJ)
    assert list(plan.hires) == [2, 2]
    assert plan.output == pytest.approx(24.0)
    # One post alone adds nothing: B would bind at once. Valued post by post, the
    # firm would never grow; valued as a bundle, it does.
    assert capacity(11, 10, 0.0, 0.5, 0.5, 0.4) == capacity(10, 10, 0.0, 0.5, 0.5, 0.4)


def test_no_hiring_when_the_bundle_loses_money():
    f = _firm([10, 10])
    unit_labor = f.amount("cost") / 20.0
    plan = firms.plan_staff(f, 24.0, 0.9 * unit_labor, [0.0] * 4, ADJ)
    assert list(plan.hires) == [0, 0]


def test_durably_unprofitable_activity_is_cut_rather_than_kept():
    f = _firm([10, 10])
    unit_labor = f.amount("cost") / 20.0
    plan = firms.plan_staff(f, 20.0, 0.5 * unit_labor, [0.0] * 4, ADJ)
    assert plan.output < 20.0
    assert list(plan.hires) == [0, 0]
    assert plan.target.sum() < 20
    # Notified posts are still there, and paid, next quarter.
    assert list(plan.next_staff) == [10, 10]


def test_hiring_freeze_allows_no_new_post():
    f = _firm([10, 10])
    unit_labor = f.amount("cost") / 20.0
    plan = firms.plan_staff(f, 24.0, 2.0 * unit_labor, [0.0] * 4, ADJ, allow_hiring=False)
    assert list(plan.hires) == [0, 0]


def test_due_notices_are_cancelled_when_the_posts_are_needed_again():
    f = _firm([10, 10], due=(2, 0))
    unit_labor = f.amount("cost") / 20.0
    plan = firms.plan_staff(f, 20.0, 2.0 * unit_labor, [0.0] * 4, ADJ)
    assert list(plan.cancel) == [2, 0]
    assert list(plan.execute) == [0, 0]
    assert list(plan.hires) == [0, 0]


def test_due_notices_are_carried_out_when_demand_stays_low():
    f = _firm([10, 10], due=(2, 2))
    unit_labor = f.amount("cost") / 20.0
    plan = firms.plan_staff(f, 16.0, 2.0 * unit_labor, [0.0] * 4, ADJ)
    assert list(plan.execute) == [2, 2]
    assert list(plan.next_staff) == [8, 8]


def test_price_target_reproduces_a_calibrated_price():
    f = _firm([10, 10])
    f.markup = 0.2
    f.fixed_cost = 500.0
    plan = firms.plan_staff(f, 20.0, 10 * f.amount("cost") / 20.0, [0.0] * 4, ADJ)
    expected = 1.2 * (f.amount("cost") + 500.0) / 20.0
    assert firms.price_target(f, plan, 0.0) == pytest.approx(expected)


def test_financing_available_counts_an_overdraft_once():
    f = _firm([10, 10])
    f.cash_buffer, f.credit_ceiling = 50.0, 1000.0
    f.cash = -100.0
    assert f.financing_available() == pytest.approx(850.0)
    f.cash = 300.0
    assert f.financing_available() == pytest.approx(1250.0)
