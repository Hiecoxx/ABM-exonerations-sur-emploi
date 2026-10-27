"""The six graphs of the spec, drawn from the CSV files written by experiments.py.

1. employment, in levels and as a gap with the reference
2. machine adoptions, and those added by the reform
3. break-even technology cost against labor cost, by wage band
4. the role of competition and of income feedback
5. the partial public balance and its two parts
6. sign maps over the hypothesis grid: where, if anywhere, the sign turns

Every figure carries the status of its parameters, and says which kind of
spread it shows. None of them is a picture of France.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import params as prm  # noqa: E402
from . import payroll  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ALPHA_COLOR = {1.0: MUTED, 0.9: "#86b6ef", 0.8: "#2a78d6"}  # reference in gray, then one blue, light to dark
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
STYLE = {
    "font.family": "sans-serif", "font.size": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
}


def read(path):
    """Rows of an output CSV, comment lines skipped, numbers and booleans restored."""
    def convert(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v
    with open(path, encoding="utf-8") as f:
        return [{k: convert(v) for k, v in row.items()} for row in csv.DictReader(l for l in f if not l.startswith("#"))]


def _mean(rows, value, **where):
    sel = [r[value] for r in rows if all(r[k] == v for k, v in where.items())]
    return float(np.mean(sel)) if sel else np.nan


def _finish(fig, p, title, note, path):
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.text(0.01, 0.01, f"{prm.status(p)} · {note}", color=MUTED, fontsize=7.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def employment(paths, p, out):
    quarters = sorted({r["quarter"] for r in paths})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, constrained_layout=True)
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.94))
        for row, tech in enumerate((True, False)):
            ref = np.array([_mean(paths, "employment", alpha=1.0, technology=tech, feedback=False, quarter=q)
                            for q in quarters])
            for alpha, color in ALPHA_COLOR.items():
                level = np.array([_mean(paths, "employment", alpha=alpha, technology=tech, feedback=False, quarter=q)
                                  for q in quarters])
                label = "reference (α = 1)" if alpha == 1.0 else f"α = {alpha:g}"
                axes[row, 0].plot(quarters, level, color=color, label=label, marker="o", markersize=4)
                if alpha != 1.0:
                    axes[row, 1].plot(quarters, level - ref, color=color, label=label, marker="o", markersize=4)
            axes[row, 1].axhline(0, color=AXIS, linewidth=0.8)
            which = "technology possible" if tech else "technology impossible"
            axes[row, 0].set_title(f"Employment, {which}", loc="left")
            axes[row, 1].set_title(f"Gap with the reference, {which}", loc="left")
            axes[row, 0].set_ylabel("persons")
            axes[row, 1].set_ylabel("persons")
        for ax in axes[1]:
            ax.set_xlabel("quarter after the reform")
        axes[0, 0].legend(loc="lower left")
        axes[0, 1].legend(loc="lower left")
        return _finish(fig, p, "1 · Employment of the synthetic sample", "mean across seeds; their spread is random draws only",
                       out / "1_employment.png")


def adoptions(paths, p, out):
    quarters = sorted({r["quarter"] for r in paths})
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.9))
        ref = np.array([_mean(paths, "machines_ordered", alpha=1.0, technology=True, feedback=False, quarter=q)
                        for q in quarters])
        for alpha, color in ALPHA_COLOR.items():
            level = np.array([_mean(paths, "machines_ordered", alpha=alpha, technology=True, feedback=False, quarter=q)
                              for q in quarters])
            label = "reference (α = 1)" if alpha == 1.0 else f"α = {alpha:g}"
            left.plot(quarters, level, color=color, label=label, marker="o", markersize=4, drawstyle="steps-post")
            if alpha != 1.0:
                right.plot(quarters, level - ref, color=color, label=label, marker="o", markersize=4, drawstyle="steps-post")
        right.axhline(0, color=AXIS, linewidth=0.8)
        left.set_title("Machines ordered since the reform", loc="left")
        right.set_title("Added by the reform", loc="left")
        for ax in (left, right):
            ax.set_xlabel("quarter after the reform")
            ax.set_ylabel("firms")
            ax.legend(loc="lower right")
        return _finish(fig, p, "2 · Adoptions: the reference may already invest",
                       "mean across seeds; machines present before the reform are not counted",
                       out / "2_adoptions.png")


def break_even_costs(p, alphas, profile):
    """Unit cost at which a machine replacing one post of each band breaks even.

    One post saved from activation to the horizon, severance when it leaves,
    against capex, transition, maintenance and the residual value. Integer staff,
    prices and demand are left out: a frontier, not the firm's decision rule.
    """
    tech, fin = p["technology"], p["finance"]
    r, horizon, delay = fin["discount_rate"], tech["npv_horizon"], tech["delay"]
    active = (1 + r) ** -np.arange(delay, horizon + 1)
    residual = max(0.0, 1 - (horizon - delay + 1) / tech["lifetime"]) * (1 + r) ** -horizon
    per_unit = 1 + tech["transition_rate"] * (1 + r) ** -delay + tech["maintenance_rate"] * active.sum() - residual
    out = []
    for alpha in alphas:
        bc = payroll.band_costs(alpha, profile, p)
        severance = p["planning"]["severance_months"] / 3 * bc.gross * (1 + r) ** -(delay - 1)
        out.append((bc.cost * active.sum() - severance) / per_unit)
    return np.array(out)


def frontier(p, out):
    profile = payroll.Profile(large=False, size=1, at_mp=float(np.mean(p["markets"]["at_mp_rate"])))
    alphas = list(ALPHA_COLOR)
    costs = break_even_costs(p, alphas, profile) / 1000
    bounds = p["bands"]["bounds_smic"]
    labels = [f"{lo:g}–{hi:g} SMIC" for lo, hi in zip(bounds, bounds[1:])] + [f"≥ {bounds[-1]:g} SMIC"]
    x = np.arange(len(labels))
    width = 0.17
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.9))
        for k, (alpha, color) in enumerate(ALPHA_COLOR.items()):
            label = "reference (α = 1)" if alpha == 1.0 else f"α = {alpha:g}"
            ax.bar(x + (k - 1) * width, costs[k], width * 0.8, color=color, label=label)
        central = p["technology"]["unit_cost"] / 1000
        ax.axhline(central, color=INK2, linewidth=0.8)
        ax.text(x[0] - 0.45, central * 1.04, f"central tech cost, {central:g} k€", color=INK2, fontsize=7.5,
                va="bottom", ha="left")
        ax.set_xticks(x, labels)
        ax.set_ylabel("break-even cost, k€ per A post of capacity")
        ax.set_title("A machine pays where the bar rises above the line; the cut raises bars only below 3 SMIC",
                     loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
        return _finish(fig, p, "3 · Technology cost against labor cost, band by band",
                       "10–49 profile; one post saved, no price or demand effect; tech costs are hypotheses",
                       out / "3_frontier.png")


def competition_income(rows, p, out):
    etas = sorted({r["eta"] for r in rows})
    x = np.arange(len(etas))
    width = 0.14
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
        fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.9))
        for k, (fb, color, label) in enumerate(((False, BLUE, "exogenous budgets"),
                                                (True, ORANGE, "budgets follow income"))):
            values = [_mean(rows, "dE_tech_pct", eta=e, feedback=fb) for e in etas]
            ax.bar(x + (k - 0.5) * width, values, width * 0.8, color=color, label=label)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks(x, [f"η = {e:g}" for e in etas])
        ax.set_ylabel("employment gap, % of the reference")
        alpha = rows[0]["alpha"]
        ax.set_title(f"α = {alpha:g}, technology possible, after {int(rows[0]['quarter'])} quarters", loc="left")
        ax.legend(loc="lower left")
        return _finish(fig, p, "4 · Competition and income", "hypothesis sensitivity, not an estimate",
                       out / "4_competition_income.png")


def balance(summary, p, out):
    rows = [r for r in summary if r["market"] == "all" and r["size"] == "all" and r["feedback"] is False
            and r["alpha"] in (0.9, 0.8)]
    keys = sorted({(r["alpha"], r["quarter"]) for r in rows}, key=lambda k: (-k[0], k[1]))
    x = np.arange(len(keys))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True, constrained_layout=True)
        fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.9))
        for ax, tag, which in zip(axes, ("tech", "without"), ("technology possible", "technology impossible")):
            get = lambda name: np.array([_mean(rows, f"{name}_{tag}_mean", alpha=a, quarter=q) for a, q in keys]) / 1e6  # noqa: E731
            ax.bar(x, get("mechanical_gain"), 0.3, color=BLUE, label="mechanical gain")
            ax.bar(x, get("employment_adjustment"), 0.3, color=ORANGE, label="employment adjustment")
            ax.margins(y=0.3)
            ax.plot(x, get("dBalance"), "o", color=INK, markersize=5, label="net change")
            ax.axhline(0, color=AXIS, linewidth=0.8)
            ax.set_xticks(x, [f"α {a:g}\nq{int(q)}" for a, q in keys])
            ax.set_title(which, loc="left")
        axes[0].set_ylabel("M€ per quarter")
        axes[0].legend(loc="upper left")
        return _finish(fig, p, "5 · Partial public balance: employer contributions net of the RGDU, minus the ARE proxy",
                       "leaves out employee contributions, CSG, taxes and other benefits; mean across seeds",
                       out / "5_balance.png")


def sign_maps(rows, p, out):
    etas = sorted({r["eta"] for r in rows})
    costs = sorted({r["unit_cost"] for r in rows})
    scales = sorted({r["automatable_scale"] for r in rows})
    values = np.array([[[_mean(rows, "dE_tech_pct", eta=e, unit_cost=c, automatable_scale=s) for c in costs]
                        for s in scales] for e in etas])
    limit = max(float(np.nanmax(np.abs(values))), 0.01)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, len(etas), figsize=(3.2 * len(etas) + 1, 3.6), constrained_layout=True)
        fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.9))
        axes = np.atleast_1d(axes)
        for ax, eta, grid in zip(axes, etas, values):
            image = ax.imshow(grid, cmap=DIVERGING, vmin=-limit, vmax=limit, origin="lower", aspect="auto")
            for (i, j), v in np.ndenumerate(grid):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(v) > 0.6 * limit else INK)
            ax.set_xticks(range(len(costs)), [f"{c / 1000:g}k" for c in costs])
            ax.set_yticks(range(len(scales)), [f"×{s:g}" for s in scales])
            ax.set_xlabel("tech cost per A post (€)")
            ax.set_title(f"η = {eta:g}", loc="left")
        axes[0].set_ylabel("automatable share")
        fig.colorbar(image, ax=axes, shrink=0.85, label="employment gap, % of reference")
        alpha = rows[0]["alpha"]
        return _finish(fig, p, f"6 · Sign and size of the effect of α = {alpha:g} across the grid, technology possible",
                       "hypothesis sensitivity; blue: fewer jobs, red: more", out / "6_sign_maps.png")


def make_all(results, p):
    """Draw what the outputs allow; figures 4 and 6 need the --grid runs."""
    results = Path(results)
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    paths = read(results / "paths.csv")
    written += [employment(paths, p, out), adoptions(paths, p, out), frontier(p, out)]
    written.append(balance(read(results / "effects_summary.csv"), p, out))
    for name, draw in (("competition_income.csv", competition_income), ("grid.csv", sign_maps)):
        if (results / name).exists():
            written.append(draw(read(results / name), p, out))
        else:
            print(f"{name} missing: run with --grid to draw it")
    return written


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    for path in make_all(folder, prm.load(sys.argv[2]) if len(sys.argv) > 2 else prm.load()):
        print(path)
