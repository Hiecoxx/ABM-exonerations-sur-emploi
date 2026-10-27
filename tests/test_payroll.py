"""Payroll checks. Expected values were computed by hand, in decimal arithmetic,
independently of the module."""

import pytest

from abm_rgdu import params, payroll

P = params.load()
SMALL = payroll.Profile(large=False, size=0, at_mp=0.022)
LARGE = payroll.Profile(large=True, size=2, at_mp=0.010)
SMIC = 21876.40


def test_reference_and_current_smic():
    assert payroll.smic_reference(P) == pytest.approx(SMIC)
    assert payroll.smic_current(P) == pytest.approx(22404.20)


def test_maximum_coefficient_at_the_reference_smic_of_1_january_2026():
    assert payroll.coefficient(SMIC, False, P) == 0.3981
    assert payroll.coefficient(SMIC, True, P) == 0.4021


def test_a_post_paid_the_current_smic_gets_less_than_the_maximum():
    current = payroll.smic_current(P)
    assert payroll.coefficient(current, False, P) == 0.3750
    assert payroll.coefficient(current, True, P) == 0.3788


def test_spec_example_and_the_danger_of_the_mean():
    low, high = 1.05 * SMIC, 3.00 * SMIC
    assert low == pytest.approx(22970.22)
    assert high == pytest.approx(65629.20)
    correct = payroll.reduction(low, SMALL, P) + payroll.reduction(high, SMALL, P)
    assert correct == pytest.approx(8087.81, abs=0.005)
    mean = (low + high) / 2
    assert mean == pytest.approx(44299.71)
    at_mean = 2 * payroll.reduction(mean, SMALL, P)
    assert at_mean == pytest.approx(4545.15, abs=0.005)
    assert at_mean / correct - 1 == pytest.approx(-0.438, abs=0.0005)


def test_reduction_stops_dead_at_three_smic():
    below = 3 * SMIC - 0.01
    assert payroll.coefficient(below, False, P) == 0.02
    assert payroll.reduction(below, SMALL, P) == pytest.approx(1312.58, abs=0.005)
    assert payroll.reduction(3 * SMIC, SMALL, P) == 0.0
    assert payroll.reduction(3 * SMIC + 1000, LARGE, P) == 0.0


def test_threshold_uses_the_reference_smic_not_the_current_one():
    # 3 current SMIC is above the threshold: the RGDU is already gone there.
    assert payroll.reduction(3 * payroll.smic_current(P) - 1, SMALL, P) == 0.0
    assert payroll.coefficient(3 * SMIC - 1, False, P) > 0


@pytest.mark.parametrize(
    "gross_smic, small, large",
    [(1.12, 0.2983, 0.3012), (1.6, 0.1090, 0.1099), (2.4, 0.0299, 0.0300), (3.6, 0.0, 0.0)],
)
def test_simulated_band_wages(gross_smic, small, large):
    gross = gross_smic * SMIC
    assert payroll.coefficient(gross, False, P) == small
    assert payroll.coefficient(gross, True, P) == large


def test_coefficient_falls_as_the_wage_rises():
    wages = [SMIC * (1 + k / 50) for k in range(100)]
    coefs = [payroll.coefficient(w, False, P) for w in wages]
    assert all(a >= b for a, b in zip(coefs, coefs[1:]))


def test_eligible_rates_add_up_to_the_maximum_coefficient():
    for profile, cap in ((SMALL, 0.3981), (LARGE, 0.4021)):
        _, eligible = payroll.contributions(SMIC, profile, P)
        assert eligible / SMIC == pytest.approx(cap)
        assert payroll.reduction(SMIC, profile, P) == pytest.approx(cap * SMIC)


def test_employer_cost_of_two_posts():
    band1 = 1.12 * SMIC
    total, eligible = payroll.contributions(band1, SMALL, P)
    assert total == pytest.approx(10535.67424)
    assert eligible == pytest.approx(9754.0742208)
    assert payroll.employer_cost(band1, 1.0, SMALL, P) == pytest.approx(27728.4245056)
    assert payroll.employer_cost(band1, 0.8, SMALL, P) == pytest.approx(29190.18805248)

    band3 = 2.4 * SMIC
    total, eligible = payroll.contributions(band3, LARGE, P)
    assert total == pytest.approx(23527.199952)
    assert eligible == pytest.approx(20464.64784)
    assert payroll.employer_cost(band3, 1.0, LARGE, P) == pytest.approx(74455.459152)


def test_alpha_scales_the_reduction_and_nothing_else():
    for gross in payroll.band_wages(P):
        full = payroll.employer_cost(gross, 1.0, SMALL, P)
        none = payroll.employer_cost(gross, 0.0, SMALL, P)
        assert none - full == pytest.approx(payroll.reduction(gross, SMALL, P))


def test_band_wages_are_legal_and_inside_their_bands():
    payroll.check_band_wages(P)
    with pytest.raises(ValueError, match="current SMIC"):
        payroll.check_band_wages(P, [1.01 * SMIC, 1.6 * SMIC, 2.4 * SMIC, 3.6 * SMIC])
    with pytest.raises(ValueError, match="outside"):
        payroll.check_band_wages(P, [1.12 * SMIC, 2.1 * SMIC, 2.4 * SMIC, 3.6 * SMIC])


def test_quarterly_band_costs_are_annual_amounts_over_four():
    costs = payroll.band_costs(0.9, SMALL, P)
    for g, gross in enumerate(payroll.band_wages(P)):
        assert costs.gross[g] == pytest.approx(gross / 4)
        assert costs.reduction[g] == pytest.approx(payroll.reduction(gross, SMALL, P) / 4)
        assert costs.cost[g] == pytest.approx(payroll.employer_cost(gross, 0.9, SMALL, P) / 4)
