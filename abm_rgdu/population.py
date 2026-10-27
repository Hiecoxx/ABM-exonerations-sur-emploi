"""The initial states: markets × sizes × replications, one set of rules.

Heterogeneity is drawn conditionally on market and size from common
parameters. Prices are then set so that each firm's margin (EBE/VA) hits its
draw: productivity is given, the price adapts, never the other way round.
"""

import math
from statistics import NormalDist

import numpy as np

from . import payroll
from .firms import A, BandLadder, Firm, capacity, requirements

_NORMAL = NormalDist()


def quantile(u, quartiles):
    """Piecewise-linear quantile function through Q1, median and Q3, tails extended."""
    q1, med, q3 = quartiles
    if u < 0.5:
        return med - (0.5 - u) * (med - q1) / 0.25
    return med + (u - 0.5) * (q3 - med) / 0.25


def copula_draws(n, rho, rng):
    """n pairs of uniforms with Gaussian correlation rho; the first one stratified.

    Stratifying the margin draw over the replications of a cell keeps ten
    firms from all landing in the same quartile by chance.
    """
    u1 = (rng.permutation(n) + rng.random(n)) / n
    u1 = np.clip(u1, 1e-6, 1 - 1e-6)
    z1 = np.array([_NORMAL.inv_cdf(u) for u in u1])
    z2 = rho * z1 + math.sqrt(1 - rho**2) * rng.standard_normal(n)
    return u1, np.array([_NORMAL.cdf(z) for z in z2])


def _headcount(low, high, routine, rng):
    """Whole staff in both tasks, with no idle post, inside the size class."""
    for _ in range(1000):
        n = int(math.exp(rng.uniform(math.log(low), math.log(high + 1))))
        n = min(max(n, low), high)
        n_a = min(max(int(round(routine * n)), 1), n - 1)
        q = capacity(n_a, n - n_a, 0.0, routine, 1 - routine, 0.0)
        staff = requirements(q, 0.0, routine, 1 - routine, 0.0)
        if low <= sum(staff) <= high:
            return np.array(staff), float(q)
    raise RuntimeError("no headcount found in the size class")


def build_firms(p, rng):
    m, s, pop = p["markets"], p["sizes"], p["population"]
    payroll.check_band_wages(p)
    firms = []
    for market in range(len(m["names"])):
        routine = m["routine_share"][market]
        for size in range(len(s["names"])):
            u_margin, u_cash = copula_draws(pop["replications"], pop["margin_cash_correlation"], rng)
            for rep in range(pop["replications"]):
                staff, q0 = _headcount(*s["headcount"][size], routine, rng)
                ladders = [
                    BandLadder(rng.dirichlet(m["band_mix_concentration"] * np.array(m["band_mix"][task][market])))
                    for task in ("A", "B")
                ]
                k = m["automatable_concentration"]
                theta = float(rng.beta(k * m["automatable_mean"][market], k * (1 - m["automatable_mean"][market])))
                profile = payroll.Profile(s["large"][size], size, m["at_mp_rate"][market])
                margin = float(np.clip(quantile(u_margin[rep], s["margin_quartiles"][size]) + m["margin_shift"][market],
                                       *pop["margin_bounds"]))
                cash_months = float(np.clip(quantile(u_cash[rep], s["cash_months_quartiles"][size]),
                                            0.0, pop["cash_months_max"]))
                firm = Firm(
                    id=len(firms), market=market, size=size, replicate=rep, profile=profile,
                    a=routine, b=1 - routine, theta=theta, ladders=ladders, staff=staff,
                    price=1.0, ref_price=1.0, markup=0.0, unit_ci=0.0, fixed_cost=0.0, cash=0.0,
                    machine_capacity=theta * staff[A], initial_output=q0,
                )
                firm.set_costs(payroll.band_costs(1.0, profile, p))
                calibrate(firm, margin, cash_months, p)
                firms.append(firm)
    return firms


def calibrate(firm, margin, cash_months, p):
    """Set price, intermediate consumption and cash from the margin and liquidity draws.

    VA = W / (1 − m) so that EBE / VA = m; CA = VA / (VA rate of the market);
    CI = CA − VA, split into a fixed part and a variable part per unit.
    """
    q0 = firm.initial_output
    labor = firm.amount("cost")
    value_added = labor / (1 - margin)
    revenue = value_added / p["markets"]["va_rate"][firm.market]
    ci = revenue - value_added
    firm.fixed_cost = p["costs"]["fixed_share_ci"] * ci
    firm.unit_ci = (ci - firm.fixed_cost) / q0
    firm.price = firm.ref_price = revenue / q0
    firm.markup = revenue / (labor + ci) - 1
    monthly_charges = (labor + ci) / 3
    firm.cash = cash_months * monthly_charges
    firm.credit_ceiling = p["finance"]["credit_months"][firm.size] * monthly_charges
    firm.cash_buffer = p["finance"]["cash_buffer_months"] * monthly_charges
    firm.demand_history = [q0] * p["demand"]["history"]
    firm.target_margin = margin
