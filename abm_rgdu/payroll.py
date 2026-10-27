"""RGDU 2026 and employer contributions, computed post by post.

Only what the model simulates: full-time posts, the two FNAL profiles, the
band wages. The reduction is never applied to an average wage. Each band is a
homogeneous fictitious post; amounts are summed over posts afterwards, because
the mean of a degressive reduction is not the reduction of the mean.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    """What the employer side of a payslip depends on, besides the wage."""

    large: bool   # 50 employees or more: FNAL at 0.50 % and Tdelta = 0.3821
    size: int     # size class, for the levies outside the RGDU
    at_mp: float  # AT/MP rate notified to the firm


def smic_reference(p):
    """Annual SMIC of the RGDU formula, frozen at its 1 January 2026 value."""
    r = p["rgdu"]
    return r["smic_hourly"] * r["annual_hours"]


def smic_current(p):
    """Annual SMIC in force: the legal floor for gross wages, nothing more."""
    return p["smic_current"]["hourly"] * p["rgdu"]["annual_hours"]


def coefficient(gross, large, p):
    """RGDU coefficient of an annual full-time gross wage.

    C = Tmin + Tdelta × [½ (3 × SMIC / gross − 1)]^P, capped at Tmin + Tdelta,
    rounded to four decimals, and zero from 3 SMIC: just below the threshold
    the reduction is still Tmin × gross, at the threshold it is nothing.
    """
    r = p["rgdu"]
    smic = smic_reference(p)
    if gross >= r["exit_multiple"] * smic:
        return 0.0
    t_delta = r["t_delta_large"] if large else r["t_delta_small"]
    x = 0.5 * (r["exit_multiple"] * smic / gross - 1.0)
    c = min(r["t_min"] + t_delta * x ** r["exponent"], r["t_min"] + t_delta)
    scale = 10 ** r["decimals"]
    return math.floor(c * scale + 0.5) / scale


def _bases(gross, ceiling):
    return {
        "total": gross,
        "capped": min(gross, ceiling),
        "capped_4": min(gross, 4 * ceiling),
        "tranche_2": max(0.0, min(gross, 8 * ceiling) - ceiling),
        "above_ceiling": min(gross, 8 * ceiling) if gross > ceiling else 0.0,
    }


def contributions(gross, profile, p):
    """Annual employer contributions before the RGDU, and the part it may reduce."""
    c = p["contributions"]
    bases = _bases(gross, c["pass_annual"])
    total = eligible = 0.0
    for line in c["lines"]:
        if line.get("large", profile.large) != profile.large:
            continue
        amount = line["rate"] * bases[line["base"]]
        total += amount
        if line["eligible"]:
            eligible += amount
    total += (profile.at_mp + c["other_levies"][profile.size]) * gross
    eligible += min(profile.at_mp, c["at_mp_eligible_cap"]) * gross
    return total, eligible


def reduction(gross, profile, p):
    """Annual RGDU of one post, capped by the contributions it may reduce."""
    _, eligible = contributions(gross, profile, p)
    return min(coefficient(gross, profile.large, p) * gross, eligible)


def employer_cost(gross, alpha, profile, p):
    """Annual employer cost of one post: gross + contributions before RGDU − α × RGDU."""
    total, _ = contributions(gross, profile, p)
    return gross + total - alpha * reduction(gross, profile, p)


def band_wages(p):
    """Annual gross wage of the representative post of each band."""
    return np.array(p["bands"]["gross_smic"], dtype=float) * smic_reference(p)


def check_band_wages(p, wages=None):
    """Band wages must be legal (current SMIC) and lie inside their band (reference SMIC)."""
    wages = band_wages(p) if wages is None else np.asarray(wages, dtype=float)
    smic = smic_reference(p)
    bounds = list(p["bands"]["bounds_smic"]) + [math.inf]
    for g, wage in enumerate(wages):
        if wage < smic_current(p):
            raise ValueError(f"band {g}: {wage:.2f} € is below the current SMIC")
        if not bounds[g] * smic <= wage < bounds[g + 1] * smic:
            raise ValueError(f"band {g}: {wage:.2f} € lies outside its band")


@dataclass(frozen=True)
class BandCosts:
    """Quarterly amounts per post, band by band, for one profile and one α."""

    gross: np.ndarray
    contributions: np.ndarray  # before the RGDU
    reduction: np.ndarray      # the full RGDU, i.e. at α = 1
    cost: np.ndarray           # gross + contributions − α × reduction
    alpha: float


def band_costs(alpha, profile, p):
    """Annual amounts are computed per post, then put on a quarter by dividing by four."""
    wages = band_wages(p)
    contrib = np.array([contributions(w, profile, p)[0] for w in wages])
    red = np.array([reduction(w, profile, p) for w in wages])
    return BandCosts(wages / 4, contrib / 4, red / 4, (wages + contrib - alpha * red) / 4, alpha)
