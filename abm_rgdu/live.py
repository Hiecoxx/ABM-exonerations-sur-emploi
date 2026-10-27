"""Watch one seed unfold, quarter by quarter, in a browser.

    python -m abm_rgdu.live --open

Two branches advance side by side, from the same burn-in and the same draws:
the reference, at α = 1, and the reform, at the α you choose. Every parameter
can be changed on the page before a run, and time too: the horizon, the
burn-in, and the pace at which quarters are shown. Nothing is written back to
the parameter file; a copy can be saved under results/params/ and reused with
--params.

One seed only: what moves on the screen includes random draws. Anything meant
to be reported goes through the experiments command, over many seeds.
"""

import argparse
import copy
import json
import re
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import params as prm
from . import simulator as sim

PAGE = Path(__file__).with_name("live.html")
SAVE_DIR = Path("results") / "params"
MAX_BODY = 4 * 2**20
KEEP_ALIVE = 15.0


class Session:
    """One run being watched: two branches advanced in lockstep, which can pause, step or stop."""

    def __init__(self, p, alpha=0.8, technology=True, feedback=False, seed=0, delay=0.4):
        self.p = p
        self.alpha, self.technology, self.feedback = float(alpha), bool(technology), bool(feedback)
        self.seed = int(seed)
        self.delay = max(0.0, float(delay))
        self.paused, self.stopped, self.finished = False, False, False
        self.steps = 0
        self.lock = threading.Lock()  # one reader at a time, even across reconnections
        self.iterator = self.events()

    def events(self):
        p = self.p
        state = sim.initial_state(p, self.seed)
        firms = state.firms
        yield {"type": "start", "status": prm.status(p), "seed": self.seed, "alpha": self.alpha,
               "technology": self.technology, "feedback": self.feedback, "firms": len(firms),
               "quarters": p["simulation"]["quarters"], "burn_in": p["simulation"]["burn_in"],
               "markets": p["markets"]["names"], "sizes": p["sizes"]["names"]}
        burn = sim.Records()
        for t in sim.burn_in_steps(state, burn):
            shot = snapshot(burn, state)
            yield {"type": "quarter", "phase": "burn-in", "t": t, "reference": shot, "reform": shot,
                   "cells": cells(burn, burn, state), "events": happenings(burn, state, "both")}
        ref_state, pol_state = state, copy.deepcopy(state)
        ref, pol = sim.Records(), sim.Records()
        steps = zip(sim.run_steps(ref_state, sim.Branch(1.0, self.technology, self.feedback), ref),
                    sim.run_steps(pol_state, sim.Branch(self.alpha, self.technology, self.feedback), pol))
        for t, _ in steps:
            yield {"type": "quarter", "phase": "reform", "t": t,
                   "reference": snapshot(ref, ref_state), "reform": snapshot(pol, pol_state),
                   "cells": cells(ref, pol, ref_state),
                   "events": happenings(ref, ref_state, "reference") + happenings(pol, pol_state, "reform")}
        yield {"type": "done"}


def _last(records):
    return ({k: v[-1] for k, v in records.firm.items()}, {k: v[-1] for k, v in records.pool.items()},
            {k: v[-1] for k, v in records.public.items()})


def snapshot(records, state):
    """The whole sample in the quarter just run."""
    f, pool, public = _last(records)
    output = f["output"]
    at_initial_prices = float((np.array([x.ref_price for x in state.firms]) * output).sum())
    return {
        "employment": int(f["employment"].sum()),
        "unemployment_rate": float(pool["unemployed"].sum() / pool["active"].sum()),
        "price_index": float((f["price"] * output).sum() / at_initial_prices) if at_initial_prices > 0 else 1.0,
        "output": float(output.sum()),
        "hires": int(f["hires"].sum()), "exits": int(f["exits"].sum()), "layoffs": int(f["layoffs"].sum()),
        "notices": int(f["notices"].sum()), "vacancies": int(f["vacancies"].sum()),
        "unfilled": int((f["vacancies"] - f["hires"]).sum()),
        "machines_ordered": int(sum(o.sum() for o in records.firm["ordered"])),
        "machines_active": int((f["machine_state"] == 2).sum()),
        "blocked": int(f["blocked"].sum()), "active_firms": int(f["active"].sum()),
        "balance": float(public["balance"]) / 1e6,
        "reduction": float(public["reduction_applied"]) / 1e6,
    }


def cells(ref, pol, state):
    """Employment by market × size in both branches."""
    p = state.params
    market = np.array([f.market for f in state.firms])
    size = np.array([f.size for f in state.firms])
    e_ref = ref.firm["employment"][-1].sum(axis=1)
    e_pol = pol.firm["employment"][-1].sum(axis=1)
    out = []
    for m, mname in enumerate(p["markets"]["names"]):
        for s, sname in enumerate(p["sizes"]["names"]):
            mask = (market == m) & (size == s)
            out.append({"market": mname, "size": sname, "reference": int(e_ref[mask].sum()),
                        "reform": int(e_pol[mask].sum())})
    return out


def happenings(records, state, branch, limit=12):
    """What a reader would want to notice in the quarter: orders, refusals, layoffs, closures."""
    p = state.params
    f, _, _ = _last(records)
    who = lambda i: (f"firm {i} ({p['markets']['names'][state.firms[i].market].replace('_', ' ')}, "  # noqa: E731
                     f"{p['sizes']['names'][state.firms[i].size]} posts)")
    out = []
    for i in np.flatnonzero(f["ordered"]):
        borrowed = f["new_loan"][i]
        out.append(f"{who(i)} orders its machine, {f['capex'][i] / 1e3:,.0f} k€"
                   + (f", {borrowed / 1e3:,.0f} k€ borrowed" if borrowed > 0 else ", paid in cash"))
    for i in np.flatnonzero(f["transition"] > 0):
        out.append(f"{who(i)}: its machine starts working")
    if f["blocked"].any():
        out.append(f"{int(f['blocked'].sum())} profitable machine(s) refused for want of financing")
    for i in np.flatnonzero(f["closed"]):
        out.append(f"{who(i)} closes")
    if len(records.quarters) > 1:
        before = records.firm["active"][-2]
        for i in np.flatnonzero(f["active"] & ~before):
            out.append(f"{who(i)} is replaced by an entrant")
    staff = f["employment"].sum(axis=1)
    for i in np.flatnonzero(f["notices"].sum(axis=1) >= np.maximum(3, 0.1 * staff)):
        out.append(f"{who(i)} gives notice to {int(f['notices'][i].sum())} posts")
    for i in np.flatnonzero(f["layoffs"].sum(axis=1) >= np.maximum(3, 0.1 * staff)):
        out.append(f"{who(i)} lets {int(f['layoffs'][i].sum())} posts go")
    rows = [{"branch": branch, "text": text} for text in out[:limit]]
    if len(out) > limit:
        rows.append({"branch": branch, "text": f"and {len(out) - limit} more"})
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "abm-rgdu-live"

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/params":
            self._json({"params": self.server.params, "source": str(self.server.params_source)})
        elif url.path == "/api/stream":
            self._stream(parse_qs(url.query).get("id", [""])[0])
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        body = self._body()
        if body is None:
            return
        if url.path == "/api/start":
            if not isinstance(body.get("params"), dict):
                return self._json({"error": "a run needs a full parameter set"}, 400)
            for old in self.server.sessions.values():
                old.stopped = True
            sid = uuid.uuid4().hex
            self.server.sessions = {sid: Session(body["params"], body.get("alpha", 0.8), body.get("technology", True),
                                                 body.get("feedback", False), body.get("seed", 0),
                                                 body.get("delay", 0.4))}
            self._json({"id": sid})
        elif url.path == "/api/control":
            session = self.server.sessions.get(body.get("id", ""))
            if session is None:
                return self._json({"error": "unknown run"}, 404)
            action = body.get("action")
            if action == "pause":
                session.paused = True
            elif action == "resume":
                session.paused, session.steps = False, 0
            elif action == "step":
                session.paused, session.steps = True, session.steps + 1
            elif action == "stop":
                session.stopped = True
            elif action == "pace":
                session.delay = max(0.0, float(body.get("delay", session.delay)))
            self._json({"ok": True})
        elif url.path == "/api/save":
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(body.get("name") or "my_params")).strip("._")[:60] or "my_params"
            path = SAVE_DIR / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body["params"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._json({"path": str(path), "command": f"python -m abm_rgdu.experiments --params {path}"})
        else:
            self._json({"error": "not found"}, 404)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json({"error": "request too large"}, 413)
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._json({"error": f"invalid JSON: {exc}"}, 400)
            return None

    def _send(self, code, payload, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, value, code=200):
        self._send(code, json.dumps(value).encode("utf-8"), "application/json")

    def _event(self, value):
        self.wfile.write(f"data: {json.dumps(value)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream(self, sid):
        session = self.server.sessions.get(sid)
        if session is None or session.finished or session.stopped:
            # 204 tells the browser not to reconnect.
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        if not session.lock.acquire(blocking=False):
            return
        try:
            quiet_since = time.monotonic()
            while not session.stopped:
                if session.paused and session.steps == 0:
                    if time.monotonic() - quiet_since > KEEP_ALIVE:
                        self.wfile.write(b": still here\n\n")
                        self.wfile.flush()
                        quiet_since = time.monotonic()
                    time.sleep(0.05)
                    continue
                try:
                    event = next(session.iterator)
                except StopIteration:
                    session.finished = True
                    break
                except Exception as exc:  # a parameter the model cannot work with
                    session.finished = True
                    self._event({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                    break
                self._event(event)
                quiet_since = time.monotonic()
                if event["type"] != "quarter":
                    continue
                if session.steps > 0:
                    session.steps -= 1
                    continue
                waited = 0.0
                while waited < session.delay and not session.stopped and not session.paused:
                    time.sleep(0.05)
                    waited += 0.05
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            session.lock.release()


def serve(p, source, host="127.0.0.1", port=8050):
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.params, server.params_source, server.sessions = p, source, {}
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch one seed run quarter by quarter in a browser.")
    prm.add_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this computer only)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)
    p = prm.from_arguments(args)
    server = serve(p, args.params, args.host, args.port)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.server_address[1]}/"
    print(f"{prm.status(p)}\nLive view on {url}  (Ctrl-C to stop)")
    if args.open:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
