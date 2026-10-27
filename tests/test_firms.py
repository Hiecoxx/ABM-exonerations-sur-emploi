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
