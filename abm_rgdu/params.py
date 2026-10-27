"""Parameters, read once from the versioned JSON file.

A run never goes back to the file. Each family of parameters carries a status
in the registry: R for an official rule, E for an estimate checked against a
source, H for a hypothesis that has to be tested rather than believed.

Any parameter can be changed for one run without touching the file:
`--set demand.eta=5`, `--quarters 12`, `--burn-in 0`.
"""

import copy
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "params_v1_5.json"


def load(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def override(params, changes):
    """Copy of `params` with dotted keys replaced, e.g. {"demand.eta": 0.0}.

    Unknown keys raise, so that a typo in a sensitivity grid cannot pass
    silently for a result.
    """
    out = copy.deepcopy(params)
    for dotted, value in changes.items():
        node = out
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        if last not in node:
            raise KeyError(f"unknown parameter: {dotted}")
        node[last] = value
    return out


def status(params):
    """The label every output carries."""
    return params["meta"]["status"]


def registry_rows(params):
    return [{"family": family, **entry} for family, entry in params["registry"].items()]


def parse_assignments(items):
    """["demand.eta=5", "markets.names=[\"x\"]"] → {"demand.eta": 5, …}.

    Values are read as JSON when they can be (numbers, lists, true/false),
    and kept as text otherwise.
    """
    changes = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        changes[key.strip()] = value
    return changes


def add_arguments(parser):
    """Options shared by every command: which file, and what to change in it."""
    parser.add_argument("--params", type=Path, default=DEFAULT_PATH, help="parameter file (JSON)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="change one parameter for this run, e.g. --set demand.eta=5 (repeatable)")
    parser.add_argument("--quarters", type=int, help="quarters after the reform (simulation.quarters)")
    parser.add_argument("--burn-in", type=int, help="shared quarters at α = 1 before it (simulation.burn_in)")


def from_arguments(args):
    changes = parse_assignments(args.set)
    if args.quarters is not None:
        changes["simulation.quarters"] = args.quarters
    if args.burn_in is not None:
        changes["simulation.burn_in"] = args.burn_in
    return override(load(args.params), changes)
