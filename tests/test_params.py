import argparse
import json

import pytest

from abm_rgdu import experiments as ex
from abm_rgdu import figures
from abm_rgdu import params
from abm_rgdu import simulator as sim
from worlds import small_world


def test_assignments_are_read_as_json_when_they_can_be():
    changes = params.parse_assignments(["demand.eta=5", "variants.closures=true", "finance.credit_mode=free",
                                        "technology.cost_decline={\"stable\": 0.0, \"moderate\": 0.01, \"fast\": 0.03}",
                                        "markets.va_rate=[0.2, 0.2, 0.4, 0.5]"])
    assert changes["demand.eta"] == 5
    assert changes["variants.closures"] is True
    assert changes["finance.credit_mode"] == "free"
    assert changes["technology.cost_decline"]["moderate"] == 0.01
    assert changes["markets.va_rate"] == [0.2, 0.2, 0.4, 0.5]
    with pytest.raises(ValueError):
        params.parse_assignments(["demand.eta"])


def test_a_typo_in_a_parameter_name_is_refused():
    with pytest.raises(KeyError, match="demand.etta"):
        params.override(params.load(), {"demand.etta": 5})


def test_command_line_options_change_parameters_and_time():
    parser = argparse.ArgumentParser()
    params.add_arguments(parser)
    p = params.from_arguments(parser.parse_args(["--set", "demand.eta=4", "--quarters", "12", "--burn-in", "0"]))
    assert p["demand"]["eta"] == 4
    assert p["simulation"]["quarters"] == 12 and p["simulation"]["burn_in"] == 0
    assert params.load()["simulation"]["quarters"] == 8  # the file itself is untouched


def test_reported_quarters_follow_the_horizon():
    p = params.load()
    assert ex.report_quarters(params.override(p, {"simulation.quarters": 4})) == [2, 4]
    assert ex.report_quarters(params.override(p, {"simulation.quarters": 12})) == [2, 4, 8, 12]
    assert ex.report_quarters(params.override(p, {"simulation.quarters": 1})) == [1]


def test_impossible_horizons_are_refused():
    with pytest.raises(ValueError, match="quarters"):
        sim.initial_state(params.override(small_world(), {"simulation.quarters": 0}), 1)


def test_a_short_run_without_burn_in(tmp_path):
    p = small_world(**{"technology.unit_cost": 20000.0, "demand.lambda": 0.1, "experiments.alphas": [1.0, 0.8]})
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(p), encoding="utf-8")
    ex.main(["--seeds", "1", "--out", str(tmp_path / "out"), "--params", str(param_file), "--workers", "1",
             "--quarters", "3", "--burn-in", "0", "--set", "demand.eta=3"])
    rows = figures.read(tmp_path / "out" / "paths.csv")
    assert {r["quarter"] for r in rows} == {1.0, 2.0, 3.0}
    drift = (tmp_path / "out" / "drift.csv").read_text(encoding="utf-8")
    assert "burn-in" not in drift
