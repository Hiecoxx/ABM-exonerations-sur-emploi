"""Product markets: one budget per market, shared out by inherited share and relative price.

No household is simulated. Customers move between firms only through the
price term, and, when it is switched on, through the spillover of demand a
firm could not serve. Competition creates no customer outside these two
explicit mechanisms.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Market:
    index: int
    name: str
    firms: np.ndarray       # indices of its firms in the population
    base_share: np.ndarray  # inherited shares: initial revenue shares
    budget: float           # exogenous spending per quarter


def build_markets(firms, p):
    markets = []
    for s, name in enumerate(p["markets"]["names"]):
        members = [f for f in firms if f.market == s]
        revenue = np.array([f.price * f.initial_output for f in members])
        markets.append(Market(s, name, np.array([f.id for f in members]), revenue / revenue.sum(), revenue.sum()))
    return markets


def weights(prices, ref_prices, base_share, eta):
    return base_share * (np.asarray(prices) / np.asarray(ref_prices)) ** (-eta)


def shares(prices, ref_prices, base_share, eta, active=None):
    """share_i ∝ share0_i × (p_i / p_ref_i)^(−η) over active firms, summing to one."""
    w = weights(prices, ref_prices, base_share, eta)
    if active is not None:
        w = np.where(active, w, 0.0)
    return w / w.sum()


def allocate(budget, prices, market_shares):
    """Quantities ordered: the budget each firm attracts, divided by its price."""
    return budget * market_shares / np.asarray(prices)


def spillover(demand, capacity, prices, market_shares, psi):
    """Serve what can be served, then offer a share ψ of the unserved budget once.

    Rivals with spare capacity take it in proportion to their shares, at their
    own prices and within their spare capacity. Whatever is left is lost and
    reported. Sales can therefore never exceed the budget.
    """
    prices = np.asarray(prices, dtype=float)
    served = np.minimum(demand, capacity)
    unserved = float(((demand - served) * prices).sum())
    spare = capacity - served
    extra = np.zeros_like(served)
    takers = market_shares * (spare > 1e-9)
    if psi > 0 and unserved > 0 and takers.sum() > 0:
        extra = np.minimum(spare, psi * unserved * takers / takers.sum() / prices)
    lost = unserved - float((extra * prices).sum())
    return served + extra, extra, lost


def own_demand(price, ref_price, base_share, rest_weight, budget, eta):
    """Demand a firm can expect at its own price, its rivals' prices held fixed."""
    w = base_share * (price / ref_price) ** (-eta)
    return budget * w / (w + rest_weight) / price


def adjust_price(price, target, lam):
    """Partial adjustment: λ = 0 means rigid prices, λ = 1 an immediate move to target."""
    return price + lam * (target - price)
