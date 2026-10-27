import json

import numpy as np
import pytest

from abm_rgdu import experiments as ex
from abm_rgdu import params
from abm_rgdu import simulator as sim
from worlds import small_world

FAST = {"technology.unit_cost": 20000.0, "demand.lambda": 0.1}
BRANCHES = [sim.Branch(a, tech) for tech in (True, False) for a in (1.0, 0.8)]


@pytest.fixture(scope="module")
def seed_result():
    return ex.run_seed(small_world(**FAST), 4, BRANCHES)


def _level(res, alpha, tech, quarter, market=ex.ALL, size=ex.ALL):
    (row,) = [r for r in res["levels"] if r["alpha"] == alpha and r["technology"] == tech
              and r["quarter"] == quarter and r["market"] == market and r["size"] == size]
    return row


def test_paired_gaps_and_interaction(seed_result):
    for e in seed_result["effects"]:
        q = e["quarter"]
        assert e["dE_tech"] == _level(seed_result, 0.8, True, q, e["market"], e["size"])["employment"] - \
            _level(seed_result, 1.0, True, q, e["market"], e["size"])["employment"]
        assert e["dE_without"] == _level(seed_result, 0.8, False, q, e["market"], e["size"])["employment"] - \
            _level(seed_result, 1.0, False, q, e["market"], e["size"])["employment"]
        assert e["interaction"] == e["dE_tech"] - e["dE_without"]


def test_partial_public_balance_splits_into_mechanical_gain_and_employment_adjustment(seed_result):
    totals = [e for e in seed_result["effects"] if e["market"] == ex.ALL and e["size"] == ex.ALL]
    assert len(totals) == 3
    for e in totals:
        for tag, tech in (("tech", True), ("without", False)):
            ref = _level(seed_result, 1.0, tech, e["quarter"])
            pol = _level(seed_result, 0.8, tech, e["quarter"])
            assert e[f"dBalance_{tag}"] == pytest.approx(pol["partial_public_balance"] - ref["partial_public_balance"])
            assert e[f"mechanical_gain_{tag}"] == pytest.approx(0.2 * ref["reduction_full"])
            assert e[f"mechanical_gain_{tag}"] + e[f"employment_adjustment_{tag}"] == pytest.approx(e[f"dBalance_{tag}"])


def test_sample_totals_add_up_from_market_and_size_cells(seed_result):
    p = small_world(**FAST)
    for row in (r for r in seed_result["levels"] if r["market"] == ex.ALL and r["size"] == ex.ALL):
        parts = [r for r in seed_result["levels"] if r["alpha"] == row["alpha"] and r["technology"] == row["technology"]
                 and r["quarter"] == row["quarter"] and r["market"] != ex.ALL and r["size"] != ex.ALL]
        assert len(parts) == len(p["markets"]["names"]) * len(p["sizes"]["names"])
        assert sum(r["employment"] for r in parts) == row["employment"]
        assert sum(r["share"] for r in parts) == pytest.approx(len(p["markets"]["names"]))


def test_the_same_seed_gives_the_same_experiment(seed_result):
    again = ex.run_seed(small_world(**FAST), 4, BRANCHES)
    assert again["levels"] == seed_result["levels"]
    assert again["effects"] == seed_result["effects"]


def test_drift_rows_are_explained_by_flows(seed_result):
    rows = seed_result["drift"]
    assert {r["phase"] for r in rows} == {"burn-in", "reference"}
    by_phase = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r)
    for phase in by_phase.values():
        for now, nxt in zip(phase, phase[1:]):
            assert now["next_employment"] == nxt["employment"]


def test_validation_reports_unverified_observations_as_such():
    obs = {
        "x": {"low": 0.1, "high": 0.2, "verified": True, "source": "s"},
        "y": {"low": 0.1, "high": 0.2, "verified": True, "source": "s"},
        "z": {"low": None, "high": None, "verified": False, "source": "s"},
    }
    rows = {r["observation"]: r["status"] for r in ex.validate({"x": 0.15, "y": 0.5, "z": 0.3}, obs)}
    assert rows == {"x": "pass", "y": "fail", "z": "not verified"}


def test_seed_summary_says_what_its_spread_means():
    rows = [{"k": 1, "v": 1.0}, {"k": 1, "v": 3.0}, {"k": 2, "v": 5.0}]
    out = {r["k"]: r for r in ex.summarize(rows, ["k"], ["v"])}
    assert out[1]["v_mean"] == 2.0 and out[1]["v_sd"] == pytest.approx(np.sqrt(2))
    assert out[1]["seeds"] == 2 and out[2]["v_sd"] == 0.0
    assert all(r["uncertainty"] == ex.SEEDS_ONLY for r in out.values())


def test_command_line_writes_labelled_outputs(tmp_path):
    p = small_world(**FAST, **{"experiments.alphas": [1.0, 0.8], "experiments.report_at": [8]})
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(p), encoding="utf-8")
    ex.main(["--seeds", "1", "--out", str(tmp_path / "out"), "--params", str(param_file), "--workers", "1"])
    for name in ("registry", "levels", "effects", "effects_summary", "drift", "validation"):
        lines = (tmp_path / "out" / f"{name}.csv").read_text(encoding="utf-8").splitlines()
        assert lines[0] == f"# {params.status(p)}"
    validation = (tmp_path / "out" / "validation.csv").read_text(encoding="utf-8")
    assert "not verified" in validation


def test_feedback_branches_are_paired_with_their_own_reference():
    branch_list = [sim.Branch(a, tech, fb) for fb in (False, True) for tech in (True, False) for a in (1.0, 0.8)]
    res = ex.run_seed(small_world(**FAST, **{"experiments.report_at": [8]}), 2, branch_list)
    totals = {e["feedback"]: e for e in res["effects"] if e["market"] == ex.ALL and e["size"] == ex.ALL}
    assert set(totals) == {False, True}
    for fb, e in totals.items():
        ref = _level(res | {"levels": [r for r in res["levels"] if r["feedback"] == fb]}, 1.0, False, 8)
        pol = _level(res | {"levels": [r for r in res["levels"] if r["feedback"] == fb]}, 0.8, False, 8)
        assert e["dE_without"] == pol["employment"] - ref["employment"]


def test_all_six_figures_are_drawn_and_labelled(tmp_path):
    from abm_rgdu import figures

    p = small_world(**FAST, **{"experiments.alphas": [1.0, 0.9, 0.8], "experiments.grid.seeds": 1,
                               "experiments.grid.unit_cost": [20000.0, 80000.0],
                               "experiments.grid.automatable_scale": [1.0], "experiments.grid.eta": [0.0, 2.0]})
    param_file = tmp_path / "params.json"
    param_file.write_text(json.dumps(p), encoding="utf-8")
    ex.main(["--seeds", "1", "--out", str(tmp_path / "out"), "--params", str(param_file), "--workers", "1",
             "--grid", "--figures"])
    drawn = sorted(path.name for path in (tmp_path / "out" / "figures").glob("*.png"))
    assert drawn == ["1_employment.png", "2_adoptions.png", "3_frontier.png", "4_competition_income.png",
                     "5_balance.png", "6_sign_maps.png"]
    grid = (tmp_path / "out" / "grid.csv").read_text(encoding="utf-8")
    assert grid.startswith(f"# {params.status(p)}") and ex.HYPOTHESES in grid
    costs = figures.break_even_costs(p, [1.0, 0.8], figures.payroll.Profile(False, 0, 0.02))
    assert np.all(costs[1, :3] > costs[0, :3]) and costs[1, 3] == pytest.approx(costs[0, 3])
