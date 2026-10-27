import numpy as np
import pytest

from abm_rgdu import params, population
from abm_rgdu.firms import A, B, requirements

P = params.load()


@pytest.fixture(scope="module")
def firms():
    return population.build_firms(P, np.random.default_rng(7))


def test_120_firms_ten_per_market_and_size(firms):
    assert len(firms) == 120
    cells = {}
    for f in firms:
        cells[(f.market, f.size)] = cells.get((f.market, f.size), 0) + 1
    assert set(cells.values()) == {10} and len(cells) == 12


def test_headcounts_stay_inside_their_size_class(firms):
    for f in firms:
        low, high = P["sizes"]["headcount"][f.size]
        assert low <= f.staff.sum() <= high
        assert f.profile.large == (f.size == 2)


def test_no_idle_post_at_the_start(firms):
    for f in firms:
        assert tuple(f.staff) == requirements(f.initial_output, 0.0, f.a, f.b, f.theta)


def test_cells_are_the_band_mix_of_each_task(firms):
    for f in firms:
        assert f.cells.shape == (2, 4)
        assert f.cells[A].sum() == f.staff[A] and f.cells[B].sum() == f.staff[B]


def test_initial_accounts_hit_the_margin_draw(firms):
    va_rate = P["markets"]["va_rate"]
    for f in firms:
        q = f.initial_output
        labor = f.amount("cost")
        revenue = f.price * q
        ci = f.unit_ci * q + f.fixed_cost
        value_added = revenue - ci
        assert value_added / revenue == pytest.approx(va_rate[f.market])
        assert (value_added - labor) / value_added == pytest.approx(f.target_margin)
        assert f.price == pytest.approx((1 + f.markup) * (f.unit_ci + (labor + f.fixed_cost) / q))
        assert f.cash >= 0


def test_margin_quartiles_follow_their_targets(firms):
    shifts = P["markets"]["margin_shift"]
    for size, target in enumerate(P["sizes"]["margin_quartiles"]):
        drawn = [f.target_margin - shifts[f.market] for f in firms if f.size == size]
        assert np.percentile(drawn, [25, 50, 75]) == pytest.approx(target, abs=0.035)


def test_machine_option_is_the_automatable_share_of_initial_a_work(firms):
    for f in firms:
        assert 0 < f.theta < 1
        assert f.machine_capacity == pytest.approx(f.theta * f.staff[A])


def test_same_seed_same_population():
    one = population.build_firms(P, np.random.default_rng(3))
    two = population.build_firms(P, np.random.default_rng(3))
    other = population.build_firms(P, np.random.default_rng(4))
    assert [f.price for f in one] == [f.price for f in two]
    assert [f.price for f in one] != [f.price for f in other]


def test_quantile_goes_through_the_quartiles():
    q = (0.05, 0.15, 0.30)
    assert population.quantile(0.25, q) == pytest.approx(0.05)
    assert population.quantile(0.5, q) == pytest.approx(0.15)
    assert population.quantile(0.75, q) == pytest.approx(0.30)
