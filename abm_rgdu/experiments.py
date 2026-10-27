"""Paired experiments: one seed, one starting state, one set of shocks, several branches.

    ΔE_tech    = E(α, technology possible)   − E(1, technology possible)
    ΔE_without = E(α, technology impossible) − E(1, technology impossible)
    interaction = ΔE_tech − ΔE_without

The same decomposition is applied to the partial public balance, which is
also split into the mechanical gain of the cut, (1 − α) × RGDU on the
employment of the reference, and what the change in employment does to it.
That balance leaves out employee contributions, CSG, taxes and most benefits:
it is never the effect on public finances as a whole.

Results are given by market × size, with sample totals labelled as such: the
120 firms are an experimental sample, not France. Two spreads are reported
and never pooled: across seeds, which reflects random draws only, and across
the parameter grid, which reflects hypotheses nobody has identified yet.
"""

import argparse
import csv
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import params as prm
from . import simulator as sim
from .firms import A

ALL = "all"
SAMPLE = "synthetic sample, not representative"
SEEDS_ONLY = "random draws only"
HYPOTHESES = "hypothesis sensitivity"
VALIDATION_PATH = Path(__file__).resolve().parent.parent / "data" / "validation.json"


def branches(p, feedback=(False,)):
    e = p["experiments"]
    main = [sim.Branch(a, tech, fb) for fb in feedback for tech in (True, False) for a in e["alphas"]]
    stress = [sim.Branch(e["stress_alpha"], tech, False) for tech in (True, False)]
    return main + stress


def cells(state):
    """Firm masks: each market × size, each market, each size, and the whole sample."""
    p = state.params
    market = np.array([f.market for f in state.firms])
    size = np.array([f.size for f in state.firms])
    out = {(ALL, ALL): np.ones(len(state.firms), dtype=bool)}
    for m, name in enumerate(p["markets"]["names"]):
        out[(name, ALL)] = market == m
    for s, name in enumerate(p["sizes"]["names"]):
        out[(ALL, name)] = size == s
    for (m, mname), (s, sname) in itertools.product(enumerate(p["markets"]["names"]), enumerate(p["sizes"]["names"])):
        out[(mname, sname)] = (market == m) & (size == s)
    return out


def series(rec, state):
    """Quarterly series of one branch, cell by cell. Pool and public series exist for the whole sample only."""
    get = lambda name: rec.get("firm", name)  # noqa: E731
    by_task = {k: get(k).sum(axis=2) for k in ("employment", "hires", "exits", "layoffs", "non_replacements",
                                                "vacancies", "quits")}
    by_task["employment_a"] = get("employment")[:, :, A]
    flat = {k: get(k) for k in ("output", "revenue", "value_added", "contributions", "reduction_applied",
                                "reduction_full", "gross_wages", "share")}
    ordered = get("ordered").astype(float)
    flat["machines_ordered"] = np.cumsum(ordered, axis=0)
    flat["machines_installed"] = (get("machine_state") == 2).astype(float)
    flat["blocked_by_financing"] = get("blocked").astype(float)
    ref_price = np.array([f.ref_price for f in state.firms])
    price, output = get("price"), get("output")
    out = {}
    for key, mask in cells(state).items():
        s = {k: v[:, mask].sum(axis=1) for k, v in {**by_task, **flat}.items()}
        s["price_index"] = (price[:, mask] * output[:, mask]).sum(axis=1) / (ref_price[mask] * output[:, mask]).sum(axis=1)
        s["net_contributions"] = s["contributions"] - s["reduction_applied"]
        if ALL in key:
            del s["share"]  # a share means something for a cell within its market, not for a total
        out[key] = s
    total = out[(ALL, ALL)]
    unemployed = rec.get("pool", "unemployed").sum(axis=1)
    active = rec.get("pool", "active").sum(axis=1)
    total["unemployed"] = unemployed.astype(float)
    total["unemployment_rate"] = unemployed / active
    total["unemployed_quarters"] = np.cumsum(unemployed).astype(float)
    total["are"] = rec.get("public", "are")
    total["partial_public_balance"] = rec.get("public", "balance")
    total["lost_demand"] = rec.get("market", "lost").sum(axis=1)
    return out


def report_quarters(p):
    """Quarters reported: those asked for within the horizon, and always the last one."""
    last = p["simulation"]["quarters"]
    return sorted({q for q in p["experiments"]["report_at"] if 1 <= q <= last} | {last})


def first_order(rec):
    """Quarter at which each firm ordered its machine, or 0."""
    ordered = rec.get("firm", "ordered")
    quarters = np.array(rec.quarters)[:, None]
    return np.where(ordered.any(axis=0), (quarters * ordered).max(axis=0), 0)


def run_seed(p, seed, branch_list):
    """Everything one seed produces: levels, paired effects, drift, validation statistics."""
    state = sim.initial_state(p, seed)
    burn = sim.burn_in(state)
    records = sim.run_branches(state, branch_list)
    ser = {b: series(r, state) for b, r in records.items()}
    report = report_quarters(p)
    quarters = records[branch_list[0]].quarters
    at = [quarters.index(q) for q in report]

    paths = []
    for b, by_cell in ser.items():
        total = by_cell[(ALL, ALL)]
        for i, q in enumerate(quarters):
            paths.append({"seed": seed, **_branch(b), "quarter": q,
                          **{k: float(total[k][i]) for k in ("employment", "machines_ordered", "price_index",
                                                             "unemployment_rate", "partial_public_balance")}})

    levels = []
    for b, by_cell in ser.items():
        for (market, size), s in by_cell.items():
            for q, i in zip(report, at):
                levels.append({"seed": seed, **_branch(b), "quarter": q, "market": market, "size": size,
                               **{k: float(v[i]) for k, v in s.items()}})

    effects = []
    for alpha in sorted({b.alpha for b in branch_list if b.alpha != 1.0}, reverse=True):
        for fb in sorted({b.feedback for b in branch_list}):
            pairs = {tech: (sim.Branch(1.0, tech, fb), sim.Branch(alpha, tech, fb)) for tech in (True, False)}
            if not all(r in ser and q in ser for r, q in pairs.values()):
                continue
            ref_t, pol_t = pairs[True]
            ref_w, pol_w = pairs[False]
            orders_ref, orders_pol = first_order(records[ref_t]), first_order(records[pol_t])
            for key in ser[ref_t]:
                for q, i in zip(report, at):
                    d_tech = ser[pol_t][key]["employment"][i] - ser[ref_t][key]["employment"][i]
                    d_without = ser[pol_w][key]["employment"][i] - ser[ref_w][key]["employment"][i]
                    row = {"seed": seed, "alpha": alpha, "feedback": fb, "quarter": q, "market": key[0],
                           "size": key[1], "employment_ref_tech": ser[ref_t][key]["employment"][i],
                           "employment_ref_without": ser[ref_w][key]["employment"][i],
                           "dE_tech": d_tech, "dE_without": d_without, "interaction": d_tech - d_without,
                           "additional_adoptions": ser[pol_t][key]["machines_ordered"][i]
                           - ser[ref_t][key]["machines_ordered"][i]}
                    if key == (ALL, ALL):
                        quarter = quarters[i]
                        accelerated = (orders_pol > 0) & (orders_ref > 0) & (orders_pol < orders_ref)
                        row["accelerated_adoptions"] = float(((orders_pol <= quarter) & accelerated & (orders_pol > 0)).sum())
                        for tech, (r, pol) in pairs.items():
                            tag = "tech" if tech else "without"
                            d_bal = ser[pol][key]["partial_public_balance"][i] - ser[r][key]["partial_public_balance"][i]
                            mech = (1 - alpha) * ser[r][key]["reduction_full"][i]
                            row[f"dBalance_{tag}"] = d_bal
                            row[f"mechanical_gain_{tag}"] = mech
                            row[f"employment_adjustment_{tag}"] = d_bal - mech
                        row["interaction_balance"] = row["dBalance_tech"] - row["dBalance_without"]
                    effects.append({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                    for k, v in row.items()})

    drift = _drift(seed, burn, records.get(sim.Branch(1.0, False, False)))
    stats = _validation_stats(records.get(sim.Branch(1.0, False, False)), state)
    return {"levels": levels, "paths": paths, "effects": effects, "drift": drift, "stats": stats}


def _branch(b):
    return {"alpha": b.alpha, "technology": b.technology, "feedback": b.feedback}


def _drift(seed, burn, ref):
    """The reference is not forced to stand still; this says why it moves."""
    rows = []
    for phase, rec in (("burn-in", burn), ("reference", ref)):
        if rec is None or not rec.quarters:
            continue
        get = lambda k: rec.get("firm", k).sum(axis=(1, 2))  # noqa: E731
        emp, hires, quits, layoffs, vac, nonrep = (get(k) for k in ("employment", "hires", "quits", "layoffs",
                                                                     "vacancies", "non_replacements"))
        unemployed = rec.get("pool", "unemployed").sum(axis=1)
        for i, q in enumerate(rec.quarters):
            rows.append({"seed": seed, "phase": phase, "quarter": q, "employment": int(emp[i]),
                         "vacancies": int(vac[i]), "hires": int(hires[i]), "unfilled": int(vac[i] - hires[i]),
                         "quits": int(quits[i]), "layoffs": int(layoffs[i]), "non_replacements": int(nonrep[i]),
                         "unemployed": int(unemployed[i]),
                         "next_employment": int(emp[i] + hires[i] - quits[i] - layoffs[i])})
    return rows


def _validation_stats(ref, state):
    if ref is None:
        return {}
    emp = ref.get("firm", "employment").sum(axis=(1, 2))
    return {
        "quarterly_hire_rate": float((ref.get("firm", "hires").sum(axis=(1, 2)) / emp).mean()),
        "quarterly_exit_rate": float((ref.get("firm", "exits").sum(axis=(1, 2)) / emp).mean()),
        "unemployment_rate": float((ref.get("pool", "unemployed").sum(axis=1) / ref.get("pool", "active").sum(axis=1)).mean()),
        "initial_margin_median": float(np.median([f.target_margin for f in state.firms])),
    }


def validate(stats, observations):
    """Compare model statistics with held-out observations; unverified ones are reported as such."""
    rows = []
    for name, obs in observations.items():
        value = stats.get(name)
        if not obs.get("verified") or obs.get("low") is None or obs.get("high") is None or value is None:
            status = "not verified"
        else:
            status = "pass" if obs["low"] <= value <= obs["high"] else "fail"
        rows.append({"observation": name, "model": value, "low": obs.get("low"), "high": obs.get("high"),
                     "source": obs.get("source"), "status": status})
    return rows


def summarize(rows, keys, values):
    """Mean and spread across seeds. This spread is random draws only."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, group in groups.items():
        entry = dict(zip(keys, key))
        entry["seeds"] = len(group)
        for v in values:
            data = np.array([g[v] for g in group if g.get(v) is not None], dtype=float)
            if len(data) == 0:
                continue
            entry[f"{v}_mean"] = float(data.mean())
            entry[f"{v}_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
            entry[f"{v}_min"], entry[f"{v}_max"] = float(data.min()), float(data.max())
        entry["uncertainty"] = SEEDS_ONLY
        out.append(entry)
    return out


def run_many(p, seeds, branch_list, workers=1):
    jobs = [(p, s, branch_list) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(_job, jobs))
    return [_job(j) for j in jobs]


def _job(args):
    return run_seed(*args)


def grid(p, seeds, workers=1):
    """Signs and sizes of effects over the hypotheses nobody has identified yet."""
    g = p["experiments"]["grid"]
    alpha = g["alpha"]
    branch_list = [sim.Branch(a, tech, False) for tech in (True, False) for a in (1.0, alpha)]
    rows = []
    for cost, scale, eta in itertools.product(g["unit_cost"], g["automatable_scale"], g["eta"]):
        q = prm.override(p, {
            "technology.unit_cost": cost, "demand.eta": eta,
            "markets.automatable_mean": [min(0.95, scale * x) for x in p["markets"]["automatable_mean"]],
            "experiments.alphas": [1.0, alpha], "experiments.report_at": [p["simulation"]["quarters"]],
        })
        results = run_many(q, seeds, branch_list, workers)
        effects = [e for r in results for e in r["effects"] if e["market"] == ALL and e["size"] == ALL]
        d_tech = np.array([e["dE_tech"] / e["employment_ref_tech"] for e in effects])
        d_without = np.array([e["dE_without"] / e["employment_ref_without"] for e in effects])
        rows.append({"unit_cost": cost, "automatable_scale": scale, "eta": eta, "alpha": alpha,
                     "quarter": p["simulation"]["quarters"], "seeds": len(effects),
                     "dE_tech_pct": 100 * float(d_tech.mean()), "dE_without_pct": 100 * float(d_without.mean()),
                     "interaction_pct": 100 * float((d_tech - d_without).mean()),
                     "sign_dE_tech": int(np.sign(np.round(d_tech.mean(), 6))),
                     "additional_adoptions": float(np.mean([e["additional_adoptions"] for e in effects])),
                     "uncertainty": HYPOTHESES})
    return rows


def competition_income(p, seeds, workers=1):
    """The reform with and without income feedback, at three levels of price sensitivity."""
    alpha = p["experiments"]["grid"]["alpha"]
    branch_list = [sim.Branch(a, tech, fb) for fb in (False, True) for tech in (True, False) for a in (1.0, alpha)]
    rows = []
    for eta in p["experiments"]["grid"]["eta"]:
        q = prm.override(p, {"demand.eta": eta, "experiments.alphas": [1.0, alpha],
                             "experiments.report_at": [p["simulation"]["quarters"]]})
        effects = [e for r in run_many(q, seeds, branch_list, workers) for e in r["effects"]
                   if e["market"] == ALL and e["size"] == ALL]
        for fb in (False, True):
            sel = [e for e in effects if e["feedback"] == fb]
            rows.append({"eta": eta, "feedback": fb, "alpha": alpha, "quarter": p["simulation"]["quarters"],
                         "seeds": len(sel),
                         "dE_tech_pct": 100 * float(np.mean([e["dE_tech"] / e["employment_ref_tech"] for e in sel])),
                         "dE_without_pct": 100 * float(np.mean([e["dE_without"] / e["employment_ref_without"]
                                                                for e in sel])),
                         "uncertainty": HYPOTHESES})
    return rows


def write_csv(path, rows, p, note=None):
    """Every output file opens with the status of the parameters behind it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {prm.status(p)}\n")
        if note:
            f.write(f"# {note}\n")
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired RGDU experiments (EXPERIMENTAL parameters).")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds (default from the parameter file)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    prm.add_arguments(parser)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--grid", action="store_true", help="also run the hypothesis grid (slow)")
    parser.add_argument("--figures", action="store_true", help="draw the figures from the outputs")
    args = parser.parse_args(argv)

    p = prm.from_arguments(args)
    e = p["experiments"]
    seeds = range(args.seeds if args.seeds is not None else e["seeds"])
    results = run_many(p, seeds, branches(p, feedback=(False, True)), args.workers)
    levels = [r for res in results for r in res["levels"]]
    effects = [r for res in results for r in res["effects"]]
    out = args.out
    write_csv(out / "registry.csv", prm.registry_rows(p), p, "status of every parameter family: R rule, E checked estimate, H hypothesis")
    write_csv(out / "levels.csv", levels, p, f"rows with market or size 'all' are totals of a {SAMPLE}")
    write_csv(out / "effects.csv", effects, p, "paired gaps, seed by seed; balance = partial public balance")
    write_csv(out / "paths.csv", [r for res in results for r in res["paths"]], p,
              f"quarter by quarter, totals of a {SAMPLE}")
    numeric = [k for k in effects[0] if k not in ("seed", "alpha", "feedback", "quarter", "market", "size")]
    write_csv(out / "effects_summary.csv", summarize(effects, ["alpha", "feedback", "quarter", "market", "size"], numeric),
              p, f"spread across seeds: {SEEDS_ONLY}; it says nothing of the hypotheses")
    write_csv(out / "drift.csv", [r for res in results for r in res["drift"]], p,
              "why the reference moves: vacancies, hires, quits, layoffs, non-replacements")
    with open(VALIDATION_PATH, encoding="utf-8") as f:
        observations = json.load(f)["observations"]
    stats = {k: float(np.mean([res["stats"][k] for res in results])) for k in results[0]["stats"]}
    write_csv(out / "validation.csv", validate(stats, observations), p, "held-out observations, never used to calibrate")
    if args.grid:
        grid_seeds = range(e["grid"]["seeds"])
        write_csv(out / "grid.csv", grid(p, grid_seeds, args.workers), p,
                  f"{HYPOTHESES}: each row is one set of unidentified parameters")
        write_csv(out / "competition_income.csv", competition_income(p, grid_seeds, args.workers), p,
                  f"{HYPOTHESES}: price sensitivity of customers × income feedback")
    if args.figures:
        from . import figures
        figures.make_all(out, p)
    print(f"{prm.status(p)}: {len(seeds)} seeds written to {out}/")


if __name__ == "__main__":
    main()
