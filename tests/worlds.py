"""Small constructed worlds for tests where a mechanism must show."""

from abm_rgdu import params

P = params.load()


def small_world(**changes):
    """One market, large firms paid mostly in the first band, ample margins and cash."""
    base = {
        "markets.names": ["manufacturing"], "markets.va_rate": [0.3], "markets.routine_share": [0.5],
        "markets.automatable_mean": [0.4], "markets.at_mp_rate": [0.02], "markets.margin_shift": [0.0],
        "markets.band_mix": {"A": [[0.94, 0.02, 0.02, 0.02]], "B": [[0.94, 0.02, 0.02, 0.02]]},
        "sizes.names": ["50-250"], "sizes.headcount": [[80, 120]], "sizes.large": [True],
        "sizes.margin_quartiles": [[0.3, 0.3, 0.3]], "sizes.cash_months_quartiles": [[6.0, 6.0, 6.0]],
        "population.replications": 6, "finance.credit_months": [3.0], "demand.lambda": 0.5,
    }
    base.update(changes)
    return params.override(P, base)
