import numpy as np
import pytest

from abm_rgdu import markets, params, population

P = params.load()


@pytest.fixture(scope="module")
def world():
    firms = population.build_firms(P, np.random.default_rng(11))
    return firms, markets.build_markets(firms, P)


def test_reference_prices_reproduce_the_initial_state(world):
    firms, mkts = world
    assert len(mkts) == 4
    for m in mkts:
        members = [firms[i] for i in m.firms]
        prices = np.array([f.price for f in members])
        ref = np.array([f.ref_price for f in members])
        assert m.base_share.sum() == pytest.approx(1.0)
        demand = markets.allocate(m.budget, prices, markets.shares(prices, ref, m.base_share, 2.0))
        assert demand == pytest.approx([f.initial_output for f in members])


def test_shares_are_normalised_and_budgets_spent():
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(30))
    ref = rng.uniform(0.5, 2.0, 30)
    for eta in (0.0, 2.0, 5.0):
        prices = ref * rng.uniform(0.8, 1.2, 30)
        s = markets.shares(prices, ref, base, eta)
        assert s.sum() == pytest.approx(1.0)
        assert (markets.allocate(1000.0, prices, s) * prices).sum() == pytest.approx(1000.0)


def test_eta_zero_means_no_reallocation():
    base = np.array([0.2, 0.3, 0.5])
    s = markets.shares([3.0, 1.0, 0.5], [1.0, 1.0, 1.0], base, 0.0)
    assert s == pytest.approx(base)


def test_a_dearer_firm_loses_share():
    base = np.array([0.5, 0.5])
    s = markets.shares([1.1, 1.0], [1.0, 1.0], base, 2.0)
    assert s[0] < 0.5 < s[1]


def test_inactive_firm_gets_nothing_and_the_rest_is_reshared():
    base = np.array([0.2, 0.3, 0.5])
    s = markets.shares([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], base, 2.0, active=np.array([True, False, True]))
    assert s == pytest.approx([0.2 / 0.7, 0.0, 0.5 / 0.7])


def test_spillover_zero_changes_nothing():
    demand = np.array([10.0, 5.0])
    cap = np.array([6.0, 20.0])
    sales, extra, lost = markets.spillover(demand, cap, [1.0, 1.0], np.array([0.6, 0.4]), 0.0)
    assert sales == pytest.approx([6.0, 5.0])
    assert extra == pytest.approx([0.0, 0.0])
    assert lost == pytest.approx(4.0)


def test_full_spillover_with_ample_capacity_spends_the_whole_budget():
    prices = np.array([2.0, 1.0, 1.5])
    s = np.array([0.5, 0.3, 0.2])
    demand = markets.allocate(300.0, prices, s)
    cap = np.array([demand[0] / 2, 1e6, 1e6])
    sales, extra, lost = markets.spillover(demand, cap, prices, s, 1.0)
    assert (sales * prices).sum() == pytest.approx(300.0)
    assert lost == pytest.approx(0.0)
    assert extra[0] == 0.0


def test_spillover_never_exceeds_the_budget_or_spare_capacity():
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = 8
        prices = rng.uniform(0.5, 2.0, n)
        s = rng.dirichlet(np.ones(n))
        demand = markets.allocate(500.0, prices, s)
        cap = demand * rng.uniform(0.3, 1.5, n)
        sales, extra, lost = markets.spillover(demand, cap, prices, s, rng.uniform(0, 1))
        assert (sales * prices).sum() <= 500.0 + 1e-9
        assert np.all(sales <= cap + 1e-9)
        assert lost >= -1e-9


def test_price_adjustment_speed():
    assert markets.adjust_price(10.0, 12.0, 0.0) == 10.0
    assert markets.adjust_price(10.0, 12.0, 1.0) == 12.0
    assert markets.adjust_price(10.0, 12.0, 0.25) == pytest.approx(10.5)


def test_own_demand_agrees_with_the_market_allocation():
    base = np.array([0.2, 0.3, 0.5])
    ref = np.array([1.0, 2.0, 1.5])
    prices = np.array([1.1, 1.9, 1.5])
    s = markets.shares(prices, ref, base, 2.0)
    expected = markets.allocate(100.0, prices, s)[0]
    rest = markets.weights(prices[1:], ref[1:], base[1:], 2.0).sum()
    assert markets.own_demand(prices[0], ref[0], base[0], rest, 100.0, 2.0) == pytest.approx(expected)
