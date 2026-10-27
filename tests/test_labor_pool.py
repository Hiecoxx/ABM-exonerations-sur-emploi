import numpy as np
import pytest

from abm_rgdu import labor_pool, params, population
from abm_rgdu.firms import A, B
from abm_rgdu.labor_pool import LaborPool, apportion, match

P = params.load()


def test_apportion_sums_respects_caps_and_proportions():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = rng.integers(1, 12)
        w = rng.uniform(0, 3, n)
        caps = rng.integers(0, 30, n)
        total = int(rng.integers(0, 200))
        out = apportion(total, w, caps, rng.random(n))
        assert out.sum() == min(total, caps.sum())
        assert np.all(out <= caps) and np.all(out >= 0)
    assert list(apportion(10, [1, 1, 2], [10, 10, 10])) == [3, 2, 5]
    assert list(apportion(10, [1, 1, 2], [10, 10, 10], tiebreak=[1, 0, 0])) == [2, 3, 5]
    assert list(apportion(10, [1, 1, 8], [10, 10, 3])) == [4, 3, 3]


def test_initial_pools_close_the_field():
    firms = population.build_firms(P, np.random.default_rng(2))
    pools = labor_pool.build_pools(firms, P)
    for task, pool in zip((A, B), pools):
        employed = sum(f.staff[task] for f in firms)
        assert employed + pool.unemployed() == pool.active
        u0 = P["labor"]["initial_unemployment"][task]
        assert pool.unemployed() / pool.active == pytest.approx(u0, abs=0.002)
        shares = pool.by_bucket() / pool.unemployed()
        assert shares == pytest.approx(P["labor"]["initial_duration_shares"], abs=0.01)
        assert pool.reference_wage > 0


def test_flows_keep_employment_plus_unemployment_constant():
    rng = np.random.default_rng(1)
    pool = LaborPool(A, active=500, slots=np.array([10, 10, 8, 6, 6, 20]), reference_wage=30000.0)
    employed = 500 - pool.unemployed()
    for _ in range(50):
        out = int(rng.integers(0, 15))
        pool.receive(out)
        employed -= out
        hired = pool.hire(int(rng.integers(0, 25)), P["labor"]["hire_weights"])
        employed += hired
        pool.age()
        assert employed + pool.unemployed() == 500
        assert np.all(pool.slots >= 0)


def test_ageing_moves_everyone_one_quarter_on():
    pool = LaborPool(B, active=100, slots=np.array([1, 2, 3, 4, 5, 6]), reference_wage=1.0)
    pool.receive(7)
    pool.age()
    assert list(pool.slots) == [7, 1, 2, 3, 4, 11]


def test_hires_favour_the_recently_unemployed():
    pool = LaborPool(A, active=1000, slots=np.array([100, 100, 100, 100, 100, 100]), reference_wage=1.0)
    before = pool.slots.copy()
    pool.hire(120, [1.0, 0.7, 0.4])
    taken = before - pool.slots
    assert taken[0] > taken[3] > taken[5]
    assert pool.hire(10_000, [1.0, 0.7, 0.4]) == 600 - 120


def test_matching_is_bounded_and_rationed():
    vac = np.array([5, 0, 12, 3])
    u = np.array([0.1, 0.5, 0.9, 0.3])
    filled = match(vac, 1.0, u, available=1000, tiebreak=u)
    assert list(filled) == [5, 0, 12, 3]
    filled = match(vac, 0.8, u, available=1000, tiebreak=u)
    assert np.all(filled <= vac)
    assert list(filled) == list(np.floor(vac * 0.8 + u).astype(int))
    short = match(vac, 1.0, u, available=6, tiebreak=u)
    assert short.sum() == 6 and np.all(short <= vac)


def test_are_proxy_by_bucket():
    pool = LaborPool(A, active=100, slots=np.array([10, 0, 5, 0, 0, 4]), reference_wage=24000.0)
    a = P["are"]
    monthly = a["replacement_rate"] * 2000.0
    expected = sum(n * s * f * monthly * 3 for n, s, f in zip([10, 5, 4], a["share_compensated"], a["amount_factor"]))
    assert pool.are(P) == pytest.approx(expected)
