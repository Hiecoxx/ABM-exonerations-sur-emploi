import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from abm_rgdu import live
from worlds import small_world

FAST = {"technology.unit_cost": 20000.0, "demand.lambda": 0.1, "simulation.quarters": 3, "simulation.burn_in": 2}


def test_a_session_streams_burn_in_then_both_branches():
    events = list(live.Session(small_world(**FAST), alpha=0.8, technology=True, seed=3, delay=0).events())
    kinds = [e["type"] for e in events]
    assert kinds == ["start"] + ["quarter"] * 5 + ["done"]
    quarters = [e for e in events if e["type"] == "quarter"]
    assert [(e["phase"], e["t"]) for e in quarters] == [("burn-in", -1), ("burn-in", 0), ("reform", 1),
                                                       ("reform", 2), ("reform", 3)]
    for e in quarters:
        for branch in ("reference", "reform"):
            assert sum(c[branch] for c in e["cells"]) == e[branch]["employment"]
        if e["phase"] == "burn-in":
            assert e["reference"] == e["reform"]


def test_the_reference_watched_against_itself_never_moves_apart():
    events = live.Session(small_world(**FAST), alpha=1.0, technology=True, seed=3, delay=0).events()
    for e in events:
        if e["type"] == "quarter":
            assert e["reference"] == e["reform"]


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    srv = live.serve(small_world(**FAST), "test parameters", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def _read_events(url, into):
    with urllib.request.urlopen(url) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: "):
                into.append(json.loads(line[6:]))


def test_page_parameters_run_and_no_restart_after_the_end(server):
    page = urllib.request.urlopen(server + "/").read().decode()
    assert "<title>ABM RGDU live</title>" in page
    params = json.loads(urllib.request.urlopen(server + "/api/params").read())
    assert params["source"] == "test parameters"
    assert _post(server + "/api/start", {"alpha": 0.8})[0] == 400
    status, out = _post(server + "/api/start", {"params": params["params"], "alpha": 0.8, "delay": 0})
    assert status == 200
    events = []
    _read_events(f"{server}/api/stream?id={out['id']}", events)
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    assert urllib.request.urlopen(f"{server}/api/stream?id={out['id']}").status == 204


def test_pause_step_and_resume(server):
    params = json.loads(urllib.request.urlopen(server + "/api/params").read())["params"]
    sid = _post(server + "/api/start", {"params": params, "delay": 0})[1]["id"]
    _post(server + "/api/control", {"id": sid, "action": "pause"})
    events = []
    reader = threading.Thread(target=_read_events, args=(f"{server}/api/stream?id={sid}", events), daemon=True)
    reader.start()
    time.sleep(0.4)
    assert events == []
    _post(server + "/api/control", {"id": sid, "action": "step"})
    time.sleep(0.6)
    assert [e["type"] for e in events] == ["start", "quarter"]
    _post(server + "/api/control", {"id": sid, "action": "resume"})
    reader.join(timeout=30)
    assert events[-1]["type"] == "done"


def test_a_model_error_reaches_the_page(server):
    params = json.loads(urllib.request.urlopen(server + "/api/params").read())["params"]
    params["simulation"]["quarters"] = 0
    sid = _post(server + "/api/start", {"params": params, "delay": 0})[1]["id"]
    events = []
    _read_events(f"{server}/api/stream?id={sid}", events)
    assert events[-1]["type"] == "error" and "quarters" in events[-1]["message"]


def test_saving_a_copy_stays_under_results(server, tmp_path):
    params = json.loads(urllib.request.urlopen(server + "/api/params").read())["params"]
    status, out = _post(server + "/api/save", {"params": params, "name": "../../elsewhere/my set"})
    assert status == 200
    saved = tmp_path / "results" / "params" / "elsewhere_my_set.json"
    assert saved.exists() and json.loads(saved.read_text()) == params
    assert "--params" in out["command"]
